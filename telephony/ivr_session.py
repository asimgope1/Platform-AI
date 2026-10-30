"""
IVR call session — one instance per active WebSocket connection.

Manages:
  • Structured Multilingual IVR Keypad Menu (English, Hindi, Odia)
  • Dynamic DTMF event processing & barge-in interruption
  • accumulation of incoming PCM frames (VAD-gated only during AI_CONVERSATION)
  • Speech-to-Text (Whisper local with language locking)
  • LLM turn via ChatOrchestrator with pgvector RAG & Gemma 4 fallback
  • Text-to-Speech → raw s16le 8 kHz PCM
  • chunked binary sending back on the WebSocket
  • conversation history (system prompt + rolling turns)
"""
from __future__ import annotations

import os
import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import WebSocket

from audio.service import AudioService
from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.contracts.requests import ChatMessage, NormalizedChatRequest
from platform_core.security.tenant_context import TenantContext
from storage.database import async_session
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService
from retrieval.rag_pipeline import RAGPipeline
from telephony.pcm_utils import (
    PBX_CHUNK_BYTES,
    PBX_SAMPLE_RATE,
    audio_bytes_to_pcm8k,
    chunk_pcm,
    is_silence,
    pcm_bytes_to_float32,
    wav_bytes_to_pcm8k,
)
from telephony.voice_learning import learn_from_call_summary, learn_from_voice_turn
from telephony.ivr_menu import (
    IVRState,
    IVRMenuStateMachine,
    SUPPORTED_LANGUAGES,
    SERVICE_CATEGORIES,
    SERVICE_BY_KEY,
    MENU_TIMEOUT_SECONDS,
)
from telephony.ivr_prompts import (
    WELCOME_AND_LANGUAGE,
    SERVICE_MENUS,
    SERVICE_NAMES,
    SERVICE_ACKS,
    INVALID_INPUT,
    TIMEOUT_PROMPTS,
    MAX_RETRIES_EXCEEDED,
    HUMAN_TRANSFER,
    get_service_ack_prompt,
)
from telephony.ivr_audio_cache import get_cached_prompt_pcm, store_cached_prompt_pcm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ──────────────────────────────────────────────────────────────────────────────

# How much PCM to accumulate before running STT (avoid sending tiny slivers)
MIN_SPEECH_BYTES = int(PBX_SAMPLE_RATE * 2 * 0.4)    # 0.4 second minimum (6,400 bytes)
MAX_SPEECH_BYTES = PBX_SAMPLE_RATE * 2 * 8           # 8 seconds hard cap

# Silence gap before we "close" an utterance and run STT
SILENCE_GAP_SECONDS = 0.8

# Delay between sending 20 ms PCM reply frames (throttles playback to real-time)
FRAME_INTERVAL_SECONDS = 0.020   # 20 ms

# Model used for chat; override via env / config
DEFAULT_CHAT_MODEL = "mock-chat"
GEMMA_MODEL_ALIAS = os.getenv("GEMMA_MODEL_ALIAS", "gemma-gpu")

# Ignore PBX transcript events by default; rely on local high-accuracy Whisper STT
PBX_IGNORE_TRANSCRIPT = os.getenv("PBX_IGNORE_TRANSCRIPT", "true").lower() in ("true", "1", "yes")

BASE_SYSTEM_PROMPT = (
    "You are a friendly, polite, and warm customer support specialist on a live phone call. "
    "Guidelines for natural phone conversation:\n"
    "1. Speak naturally like a real human customer care agent — never say you are an AI, bot, virtual assistant, or language model.\n"
    "2. Always respond in conversational {language_name}.\n"
    "3. Keep all responses brief and natural (1 to 2 spoken sentences maximum, under 35 words).\n"
    "4. Be polite, warm, direct, and empathetic.\n"
    "5. Do NOT use markdown, bullet points, asterisks, numbered lists, or URLs. Speak plain conversational sentences.\n"
    "6. If the caller asks to talk to a person or representative, warmly inform them that you can transfer them.\n"
    "7. Female Speaker Grammatical Gender: You are speaking with a female voice. In Hindi and any gendered language, ALWAYS use feminine first-person grammatical forms for yourself (e.g. 'कर सकती हूँ', 'सुन रही हूँ', 'मदद करूँगी', 'समझ रही हूँ', 'चेक कर रही हूँ' — NEVER use masculine forms like 'सकता हूँ' or 'रहा हूँ')."
)


# ──────────────────────────────────────────────────────────────────────────────
# IVR session
# ──────────────────────────────────────────────────────────────────────────────

class IVRCallSession:
    """
    Handles one phone call end-to-end with structured keypad IVR navigation
    and seamless transition into RAG-powered AI conversation.
    """

    def __init__(
        self,
        ws: WebSocket,
        orchestrator: ChatOrchestrator,
        audio_service: AudioService,
        chat_model: str = DEFAULT_CHAT_MODEL,
        language: str = "en",
    ):
        self._ws = ws
        self._orchestrator = orchestrator
        self._audio_service = audio_service
        self._chat_model = chat_model
        self._language = language
        self.tenant_id = os.getenv("IVR_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")

        # IVR Keypad Menu State Machine
        self.menu_machine = IVRMenuStateMachine()
        self.selected_language: str = "en"
        self.selected_service: Optional[str] = None
        self._menu_timeout_task: Optional[asyncio.Task] = None

        # Call metadata & conversation history
        self.call_uuid: Optional[str] = None
        self.caller_number: Optional[str] = None
        self._history: list[ChatMessage] = []

        # PCM accumulator (active only in AI_CONVERSATION)
        self._pcm_buffer: bytearray = bytearray()
        self._last_voice_ts: float = 0.0
        self._silence_task: Optional[asyncio.Task] = None

        # Concurrency & Playback controls
        self._pipeline_lock = asyncio.Lock()
        self._playback_task: Optional[asyncio.Task] = None
        self._is_playing: bool = False

    # ── call lifecycle handlers ───────────────────────────────────────────────

    def on_call_start(self, event: dict[str, Any]) -> None:
        """Invoked when PBX notifies of incoming call connection."""
        self.call_uuid = event.get("uuid") or event.get("call_uuid") or "unknown"
        self.caller_number = (
            event.get("caller")
            or event.get("caller_id")
            or event.get("caller_num")
            or event.get("caller_number")
            or event.get("from")
            or event.get("channel")
            or "unknown"
        )
        self.selected_language = "en"
        self.selected_service = None
        self._language = "en"
        self._history = [
            ChatMessage(role="system", content=BASE_SYSTEM_PROMPT.format(language_name="English"))
        ]
        self._pcm_buffer.clear()
        self._last_voice_ts = time.monotonic()

        # Start menu state machine in LANGUAGE_SELECTION
        self.menu_machine.start_call()
        logger.info(
            f"[IVR] 📞 Call started: uuid={self.call_uuid} | Caller: {self.caller_number} | State: {self.menu_machine.state}"
        )

        # Launch the initial Welcome & Language Selection menu flow
        asyncio.create_task(self._play_welcome_and_language_menu())

    def on_call_end(self, event: dict[str, Any]) -> None:
        """Invoked when PBX or caller hangs up."""
        logger.info(f"[IVR] 📵 Call ended: uuid={self.call_uuid}")
        self.menu_machine.state = IVRState.CALL_END
        self._is_playing = False
        self._cancel_menu_timeout()

        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        self._pcm_buffer.clear()

        # Continuous Learning: summarize and store call dialogue in pgvector
        if self._history and len(self._history) > 1:
            asyncio.create_task(
                learn_from_call_summary(
                    tenant_id=self.tenant_id,
                    call_uuid=self.call_uuid,
                    history=list(self._history),
                )
            )

        self._history.clear()

    # ── DTMF & Menu Navigation ───────────────────────────────────────────────

    async def on_dtmf(self, digit: str, event: Optional[dict[str, Any]] = None) -> None:
        """
        Processes a keypad digit pressed by the caller.
        Supports instant barge-in interruption of active menu prompts.
        """
        digit = digit.strip()
        logger.info(
            f"[IVR] 🔢 on_dtmf digit='{digit}' | Current State={self.menu_machine.state} | Retries={self.menu_machine.retries}"
        )

        # Stop pending menu timeout timer
        self._cancel_menu_timeout()

        # Interrupt active audio prompt playback immediately (barge-in)
        if self._playback_task and not self._playback_task.done():
            logger.info("[IVR] Barge-in: interrupting active prompt playback for DTMF.")
            self._playback_task.cancel()
            self._is_playing = False

        new_state, action = self.menu_machine.process_dtmf(digit)
        logger.info(f"[IVR] Menu Transition -> State={new_state}, Action={action}")

        if action == "LANGUAGE_SELECTED":
            self.selected_language = self.menu_machine.language
            self._language = self.selected_language
            logger.info(f"[IVR] Language set to: '{self.selected_language}'")
            asyncio.create_task(self._play_service_menu())

        elif action == "SERVICE_SELECTED":
            self.selected_service = self.menu_machine.service_key
            logger.info(f"[IVR] Service category set to: '{self.selected_service}'")
            asyncio.create_task(self._play_service_ack_and_start_ai())

        elif action == "HUMAN_TRANSFER":
            logger.info("[IVR] Caller requested human assistance.")
            asyncio.create_task(self._play_transfer_notice())

        elif action == "REPEAT_MENU":
            logger.info("[IVR] Caller requested menu repeat (key 0).")
            if self.menu_machine.state == IVRState.LANGUAGE_SELECTION:
                asyncio.create_task(self._play_welcome_and_language_menu())
            else:
                asyncio.create_task(self._play_service_menu())

        elif action == "INVALID_INPUT":
            logger.warning(f"[IVR] Invalid digit '{digit}' for state {self.menu_machine.state}.")
            asyncio.create_task(self._play_invalid_input_and_repeat())

        elif action == "MAX_RETRIES_EXCEEDED":
            logger.warning("[IVR] Max invalid retries exceeded; applying default route.")
            if self.menu_machine.state == IVRState.SERVICE_SELECTION:
                # Defaulted to English, now present service menu
                asyncio.create_task(self._play_service_menu())
            elif self.menu_machine.state == IVRState.AI_CONVERSATION:
                # Defaulted to Customer Support, enter AI conversation
                self.selected_service = self.menu_machine.service_key
                asyncio.create_task(self._play_service_ack_and_start_ai())

        elif action == "IGNORED":
            logger.debug(f"[IVR] Digit '{digit}' ignored in state {self.menu_machine.state}.")

    def _schedule_menu_timeout(self, seconds: float = MENU_TIMEOUT_SECONDS) -> None:
        """Schedules a timeout callback if caller does not press a key."""
        self._cancel_menu_timeout()
        self._menu_timeout_task = asyncio.create_task(self._on_menu_timeout(seconds))

    def _cancel_menu_timeout(self) -> None:
        """Cancels any active menu timeout task."""
        if self._menu_timeout_task and not self._menu_timeout_task.done():
            self._menu_timeout_task.cancel()
            self._menu_timeout_task = None

    async def _on_menu_timeout(self, seconds: float) -> None:
        """Fires when no keypad input is received within timeout window."""
        try:
            await asyncio.sleep(seconds)
            if self.menu_machine.state in (IVRState.LANGUAGE_SELECTION, IVRState.SERVICE_SELECTION):
                state, action = self.menu_machine.handle_timeout()
                logger.info(f"[IVR] ⏱️ Menu Timeout (retries={self.menu_machine.retries}) -> Action={action}")

                if action == "MAX_RETRIES_EXCEEDED":
                    if state == IVRState.SERVICE_SELECTION:
                        asyncio.create_task(self._play_service_menu())
                    else:
                        self.selected_service = self.menu_machine.service_key
                        asyncio.create_task(self._play_service_ack_and_start_ai())
                else:
                    asyncio.create_task(self._play_timeout_and_repeat())
        except asyncio.CancelledError:
            pass

    # ── Menu Playback Flows ──────────────────────────────────────────────────

    async def _play_welcome_and_language_menu(self) -> None:
        """Plays the initial multilingual greeting and language selection prompt."""
        try:
            await self._play_prompt(WELCOME_AND_LANGUAGE)
            self._schedule_menu_timeout(MENU_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _play_service_menu(self) -> None:
        """Plays the service category menu in the chosen language."""
        try:
            prompt = SERVICE_MENUS.get(self.selected_language, SERVICE_MENUS["en"])
            await self._play_prompt(prompt)
            self._schedule_menu_timeout(MENU_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            pass

    async def _play_service_ack_and_start_ai(self) -> None:
        """
        Acknowledges service selection, sets specialized system prompt & RAG context,
        and transitions call into AI_CONVERSATION.
        """
        self._cancel_menu_timeout()
        service_info = SERVICE_BY_KEY.get(self.selected_service or "customer_support", SERVICE_CATEGORIES["1"])
        service_name = service_info.get(f"name_{self.selected_language}", service_info.get("name_en", "Customer Support"))

        # Configure specialized system prompt for natural human customer conversation
        lang_name = {"en": "English", "hi": "Hindi", "or": "Odia"}.get(self.selected_language, "English")
        service_system_prompt = (
            f"You are a friendly, polite, and empathetic customer support specialist on a live phone call specializing in {service_name}. "
            f"Guidelines for natural telephone conversation:\n"
            f"1. Speak naturally like a real human customer care agent — never say you are an AI, bot, virtual assistant, or language model.\n"
            f"2. Use conversational openings when appropriate (e.g. 'Sure, I can check that for you', 'Don't worry', 'Ji bilkul', 'Aap chinta mat kijiye', 'Samajh gaya', 'Aagnya').\n"
            f"3. Always respond in conversational {lang_name}.\n"
            f"4. Keep responses brief: exactly 1 to 2 spoken sentences (under 35 words) so it sounds natural over the phone.\n"
            f"5. Never use markdown, bullet points, asterisks, numbered steps, or URLs. Speak in plain spoken sentences.\n"
            f"6. If the caller asks to speak to an agent or representative, warmly say you are connecting them.\n"
            f"7. Female Speaker Grammatical Gender: You are a female support specialist speaking with a female voice. In Hindi and any gendered language, ALWAYS use feminine first-person grammatical forms for yourself (e.g. 'कर सकती हूँ', 'सुन रही हूँ', 'मदद करूँगी', 'समझ रही हूँ', 'चेक कर रही हूँ' — NEVER use masculine forms like 'सकता हूँ' or 'रहा हूँ')."
        )
        self._history = [ChatMessage(role="system", content=service_system_prompt)]
        self._language = self.selected_language

        ack_text = get_service_ack_prompt(self.selected_language, self.selected_service or "customer_support")
        try:
            await self._play_prompt(ack_text)
            logger.info(f"[IVR] 🚀 Handoff to AI_CONVERSATION complete. Listening for caller voice...")
        except asyncio.CancelledError:
            pass
        finally:
            self._pcm_buffer.clear()
            self._last_voice_ts = time.monotonic()

    async def _play_transfer_notice(self) -> None:
        """Announces call transfer to a customer support representative."""
        self._cancel_menu_timeout()
        prompt = HUMAN_TRANSFER.get(self.selected_language, HUMAN_TRANSFER["en"])
        try:
            await self._play_prompt(prompt)
            logger.info(f"[IVR] Transfer announcement played for call uuid={self.call_uuid}.")
        except asyncio.CancelledError:
            pass

    async def _play_invalid_input_and_repeat(self) -> None:
        """Informs caller of invalid selection and replays current menu."""
        prompt = INVALID_INPUT.get(self.selected_language, INVALID_INPUT["en"])
        try:
            await self._play_prompt(prompt)
            if self.menu_machine.state == IVRState.LANGUAGE_SELECTION:
                await self._play_welcome_and_language_menu()
            else:
                await self._play_service_menu()
        except asyncio.CancelledError:
            pass

    async def _play_timeout_and_repeat(self) -> None:
        """Informs caller of timeout and replays current menu."""
        prompt = TIMEOUT_PROMPTS.get(self.selected_language, TIMEOUT_PROMPTS["en"])
        try:
            await self._play_prompt(prompt)
            if self.menu_machine.state == IVRState.LANGUAGE_SELECTION:
                await self._play_welcome_and_language_menu()
            else:
                await self._play_service_menu()
        except asyncio.CancelledError:
            pass

    async def _play_prompt(self, text: str) -> None:
        """
        Synthesizes text prompt into PCM 8kHz and streams to PBX.
        Checks in-memory/disk PCM cache first for instant 0ms playback!
        Managed as a playback task so it can be cleanly cancelled on DTMF barge-in.
        """
        self._playback_task = asyncio.current_task()
        self._is_playing = True
        try:
            pcm_reply = get_cached_prompt_pcm(text)
            if pcm_reply:
                logger.info(f"[IVR] ⚡ Instant prompt from cache ({len(pcm_reply)} bytes PCM, 0ms latency)")
            else:
                logger.info(f"[IVR] 🔊 Cache miss, synthesizing prompt ({len(text)} chars): '{text[:60]}...'")
                tts_bytes = await self._audio_service.synthesize(
                    text=text,
                    model="tts-1",
                    voice="alloy",
                    response_format="wav",
                )
                pcm_reply = _tts_output_to_pcm8k(tts_bytes)
                if pcm_reply:
                    store_cached_prompt_pcm(text, pcm_reply)

            await self._send_pcm_reply(pcm_reply)
        except asyncio.CancelledError:
            logger.info("[IVR] Audio playback cancelled by barge-in or hangup.")
            raise
        except Exception as exc:
            logger.error(f"[IVR] Error during prompt playback: {exc}")
        finally:
            self._is_playing = False

    # ── Audio Frame Processing (AI_CONVERSATION Only) ─────────────────────────

    async def on_pcm_frame(self, data: bytes) -> None:
        """
        Receive one binary WebSocket frame (raw PCM from Asterisk).
        IMPORTANT: Speech is gated and accumulated ONLY when caller is in AI_CONVERSATION.
        During menu navigation (WELCOME, LANGUAGE_SELECTION, SERVICE_SELECTION),
        caller microphone audio is ignored to prevent false STT triggers.
        """
        if self.menu_machine.state != IVRState.AI_CONVERSATION:
            return

        # When bot is actively speaking back to user, ignore echoed audio frames
        if self._is_playing:
            return

        silent = is_silence(data)

        if not silent:
            self._pcm_buffer.extend(data)
            self._last_voice_ts = time.monotonic()

            # Cancel any pending silence-triggered flush
            if self._silence_task and not self._silence_task.done():
                self._silence_task.cancel()

            # Hard cap — avoid unbounded accumulation
            if len(self._pcm_buffer) >= MAX_SPEECH_BYTES:
                await self._flush_utterance()
        else:
            # Schedule a flush after SILENCE_GAP_SECONDS of quiet
            gap = time.monotonic() - self._last_voice_ts
            if (
                len(self._pcm_buffer) >= MIN_SPEECH_BYTES
                and (self._silence_task is None or self._silence_task.done())
            ):
                self._silence_task = asyncio.create_task(
                    self._delayed_flush(SILENCE_GAP_SECONDS)
                )

    async def on_transcript(self, event: dict[str, Any]) -> None:
        """
        PBX already ran STT and sent us the transcript text.
        If PBX_IGNORE_TRANSCRIPT is enabled, ignore external PBX transcript
        and use local high-accuracy Whisper STT instead.
        """
        if self.menu_machine.state != IVRState.AI_CONVERSATION:
            return

        if PBX_IGNORE_TRANSCRIPT:
            text = (event.get("text") or event.get("transcript") or "").strip()
            logger.info(f"[IVR] PBX external transcript ignored: '{text}' (using local Whisper STT)")
            return

        if self._is_playing:
            logger.debug("[IVR] Ignoring transcript while AI is speaking.")
            return

        text = (event.get("text") or event.get("transcript") or "").strip()
        if text:
            self._pcm_buffer.clear()
            if self._silence_task and not self._silence_task.done():
                self._silence_task.cancel()

            logger.info(f"[IVR] PBX transcript: '{text}'")
            async with self._pipeline_lock:
                await self._run_ai_and_reply(text)

    # ── Conversational AI Pipeline ───────────────────────────────────────────

    async def _delayed_flush(self, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._flush_utterance()

    async def _flush_utterance(self) -> None:
        """Run STT → AI → TTS for whatever is in the buffer."""
        if len(self._pcm_buffer) < MIN_SPEECH_BYTES:
            self._pcm_buffer.clear()
            return

        pcm_snapshot = bytes(self._pcm_buffer)
        self._pcm_buffer.clear()

        async with self._pipeline_lock:
            await self._run_stt_ai_tts(pcm_snapshot)

    async def _run_stt_ai_tts(self, pcm: bytes) -> None:
        # 1. STT — wrap raw PCM as minimal WAV so AudioService can read it
        wav_bytes = _raw_pcm_to_wav(pcm, sample_rate=PBX_SAMPLE_RATE)
        try:
            stt_result = await self._audio_service.transcribe(
                audio_bytes=wav_bytes,
                filename="call_audio.wav",
                language=self._language,
            )
            user_text = (stt_result.get("text") or "").strip()
        except Exception as exc:
            logger.error(f"[IVR] STT error: {exc}")
            return

        if not user_text:
            logger.debug("[IVR] STT returned empty/silence, skipping turn.")
            return

        logger.info(f"[IVR] 🎙️ Caller Voice Transcribed (Whisper lang={self._language}): '{user_text}'")
        await self._run_ai_and_reply(user_text)

    async def _run_ai_and_reply(self, user_text: str) -> None:
        # 2. Build message history and call LLM with RAG retrieval from vector store
        self._history.append(ChatMessage(role="user", content=user_text))

        request = NormalizedChatRequest(
            model_alias=self._chat_model,
            messages=self._history,
            stream=False,
            max_tokens=120,      # short for IVR
            temperature=0.7,
        )

        try:
            # 1. Query pgvector RAG chunks for IVR knowledge
            async with async_session() as session:
                doc_repo = DocumentRepository(session)
                emb_service = EmbeddingService(provider_id="semantic-default")
                search_service = SemanticSearchService(doc_repo=doc_repo, embedding_service=emb_service)
                rag_pipeline = RAGPipeline(search_service=search_service)

                ivr_tenant = os.getenv("IVR_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")
                tenant_ctx = TenantContext(tenant_id=ivr_tenant, tenant_name="tenant-ivr-support")

                # Check if RAG has high-confidence domain knowledge chunks
                rag_threshold = float(os.getenv("RAG_MATCH_THRESHOLD", "0.28"))
                search_results = await search_service.search(tenant_id=ivr_tenant, query=user_text, top_k=3)
                top_score = max((score for _, score in search_results), default=0.0)
                has_rag_match = any(score >= rag_threshold for _, score in search_results)

                if has_rag_match:
                    logger.info(f"[IVR] ✓ Found {len(search_results)} relevant RAG chunks in pgvector (top score: {top_score:.3f} >= {rag_threshold}).")
                    rag_orchestrator = ChatOrchestrator(rag_pipeline=rag_pipeline)
                    response = await rag_orchestrator.execute(request, context=tenant_ctx, enable_rag=True)
                    reply_text = response.choices[0].message.content or ""
                else:
                    # RAG does not have this! Take help of Gemma to respond intelligently
                    logger.info(
                        f"[IVR] 🌐 Query '{user_text[:40]}' not in RAG chunks (top score: {top_score:.3f} < {rag_threshold}). "
                        f"Taking help of Gemma ({GEMMA_MODEL_ALIAS})..."
                    )
                    gemma_request = request.model_copy(update={"model_alias": GEMMA_MODEL_ALIAS})
                    try:
                        rag_orchestrator = ChatOrchestrator(rag_pipeline=rag_pipeline)
                        # Tight 3.5s timeout: IVR callers cannot wait in silence if Colab hangs/crashes
                        response = await asyncio.wait_for(
                            rag_orchestrator.execute(gemma_request, context=tenant_ctx, enable_rag=False),
                            timeout=3.5,
                        )
                        reply_text = response.choices[0].message.content or ""
                    except Exception as gemma_err:
                        logger.warning(f"[IVR] Gemma model unavailable or timed out ({gemma_err}); using instant fallback orchestrator.")
                        response = await self._orchestrator.execute(request)
                        reply_text = response.choices[0].message.content or ""

        except Exception as exc:
            logger.error(f"[IVR] LLM/RAG error: {exc}")
            try:
                response = await self._orchestrator.execute(request)
                reply_text = response.choices[0].message.content or ""
            except Exception:
                reply_text = "Sorry, I could not process your request. Please try again."

        reply_text = reply_text.strip()
        logger.info(f"[IVR] AI reply: '{reply_text}'")

        # Keep rolling history
        self._history.append(ChatMessage(role="assistant", content=reply_text))

        # Continuous Learning: learn novel Q&A pairs from voice turns into pgvector
        asyncio.create_task(
            learn_from_voice_turn(
                tenant_id=self.tenant_id,
                call_uuid=self.call_uuid,
                user_text=user_text,
                reply_text=reply_text,
            )
        )

        # Trim to last 10 turns (+ system) to avoid context explosion
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

        # 3. TTS → raw PCM 8 kHz (clean spoken text of markdown/bullets)
        spoken_reply = _clean_spoken_text(reply_text)
        try:
            logger.info(f"[IVR] Synthesizing speech for reply ({len(spoken_reply)} chars)...")
            tts_bytes = await self._audio_service.synthesize(
                text=spoken_reply,
                model="tts-1",
                voice="alloy",
                response_format="wav",
            )
            pcm_reply = _tts_output_to_pcm8k(tts_bytes)
            logger.info(f"[IVR] TTS ready: {len(pcm_reply)} bytes PCM ({len(pcm_reply)/16000:.1f}s)")
        except Exception as exc:
            logger.error(f"[IVR] TTS error: {exc}")
            return

        # 4. Send reply over WebSocket
        self._is_playing = True
        try:
            await self._send_pcm_reply(pcm_reply)
        finally:
            self._is_playing = False
            self._pcm_buffer.clear()
            self._last_voice_ts = time.monotonic()

    async def _send_pcm_reply(self, pcm: bytes) -> None:
        """
        Stream raw PCM frames to PBX over WebSocket with jitter-buffer pre-filling
        and high-precision drift compensation to eliminate glitches and network stutter.
        """
        try:
            logger.info(f"[IVR] Sending audio reply to Asterisk (uuid={self.call_uuid})...")
            if self.call_uuid:
                header = json.dumps({"event": "play", "uuid": self.call_uuid})
                await self._ws.send_text(header)

            chunks = list(chunk_pcm(pcm, PBX_CHUNK_BYTES))
            total_chunks = len(chunks)
            if total_chunks == 0:
                return

            # PREBUFFERING: Send initial burst of 5 frames (100 ms) immediately to fill
            # Asterisk/MizoDroid jitter buffer so event loop delays never cause underruns.
            PREBUFFER_FRAMES = min(5, total_chunks)
            for i in range(PREBUFFER_FRAMES):
                await self._ws.send_bytes(chunks[i])

            chunks_sent = PREBUFFER_FRAMES
            start_time = time.monotonic()

            for i in range(PREBUFFER_FRAMES, total_chunks):
                frame = chunks[i]
                await self._ws.send_bytes(frame)
                chunks_sent += 1

                # Calculate target wall-clock time for subsequent frames
                target_time = start_time + (i - PREBUFFER_FRAMES + 1) * FRAME_INTERVAL_SECONDS
                now = time.monotonic()
                sleep_duration = target_time - now

                # If behind by more than 100ms due to a system freeze, resync to avoid a huge burst
                if sleep_duration < -0.100:
                    start_time = now - (i - PREBUFFER_FRAMES + 1) * FRAME_INTERVAL_SECONDS
                    sleep_duration = 0

                if sleep_duration > 0.002:
                    await asyncio.sleep(sleep_duration)

            logger.info(f"[IVR] Finished sending audio: {chunks_sent} frames sent.")

        except asyncio.CancelledError:
            logger.info(f"[IVR] PCM stream cancelled (uuid={self.call_uuid}). Sending soft fade-out.")
            # Send 2 frames (40ms) of comfort silence to Asterisk so speaker DAC ramps down cleanly
            try:
                silence_frame = b"\x00" * PBX_CHUNK_BYTES
                await self._ws.send_bytes(silence_frame)
                await self._ws.send_bytes(silence_frame)
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.warning(f"[IVR] Send error (call may have ended): {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_spoken_text(text: str) -> str:
    """
    Cleans text of markdown markers, asterisks, bullet points, brackets, and URLs
    so that TTS speaks smoothly and naturally without robotic pauses or pronouncing punctuation.
    """
    import re
    # Remove markdown bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove bullet markers, numbered lists, and header hashes
    text = re.sub(r"^\s*[-*#\d\.]+\s*", "", text, flags=re.MULTILINE)
    # Remove robotic bracket tags like [Source 1], (Note: ...)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(Note:.*?\)", "", text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse multiple spaces and newlines into natural sentence flow
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _raw_pcm_to_wav(pcm: bytes, sample_rate: int = PBX_SAMPLE_RATE) -> bytes:
    """Wrap raw s16le mono PCM bytes in a minimal RIFF WAV container."""
    import struct
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _tts_output_to_pcm8k(tts_bytes: bytes) -> bytes:
    """
    Convert whatever the TTS provider returns (WAV / MP3 / raw) to
    raw s16le mono 8 kHz PCM.
    """
    if not tts_bytes:
        return b""

    # WAV magic: "RIFF"
    if tts_bytes[:4] == b"RIFF":
        return wav_bytes_to_pcm8k(tts_bytes)

    # MP3 magic: 0xFF 0xFB or ID3
    if tts_bytes[:3] == b"ID3" or tts_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return audio_bytes_to_pcm8k(tts_bytes, ext=".mp3")

    # M4A / AAC
    if tts_bytes[4:8] == b"ftyp":
        return audio_bytes_to_pcm8k(tts_bytes, ext=".m4a")

    # Assume WAV as default
    return wav_bytes_to_pcm8k(tts_bytes)
