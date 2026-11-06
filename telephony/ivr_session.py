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
from collections import deque
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket

from providers.chat.ivr_mini_model import ivr_mini_brain
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

# How much PCM to accumulate before running STT (avoid dropping crisp short words like 'yes'/'no')
MIN_SPEECH_BYTES = int(PBX_SAMPLE_RATE * 2 * 0.25)    # 0.25 second minimum (4,000 bytes)
MAX_SPEECH_BYTES = PBX_SAMPLE_RATE * 2 * 8           # 8 seconds hard cap

# Silence gap before we "close" an utterance and run STT (gives natural breathing headroom)
SILENCE_GAP_SECONDS = 0.9

# Delay between sending 20 ms PCM reply frames (throttles playback to real-time)
FRAME_INTERVAL_SECONDS = 0.020   # 20 ms

# Model used for chat; override via env / config
DEFAULT_CHAT_MODEL = os.getenv("IVR_CHAT_MODEL", "ivr-rag-mini")
GEMMA_MODEL_ALIAS = os.getenv("GEMMA_MODEL_ALIAS", "ivr-rag-mini")

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
    "7. Female Speaker Grammatical Gender: You are speaking with a female voice. In Hindi and any gendered language, ALWAYS use feminine first-person grammatical forms for yourself (e.g. 'कर सकती हूँ', 'सुन रही हूँ', 'मदद करूँगी', 'समझ रही हूँ', 'चेक कर रही हूँ' — NEVER use masculine forms like 'सकता हूँ' or 'रहा हूँ').\n"
    "8. Native Script Pronunciation: When speaking in Hindi or Odia, ALWAYS write technical terms, banking acronyms, and loan words in the native script itself (e.g. in Hindi: 'केवाईसी', 'यूपीआई', 'ओटीपी', 'पैन कार्ड', 'आधार'; in Odia: 'କେୱାଇସି', 'ୟୁପିଆଇ', 'ଓଟିପି', 'ପ୍ୟାନ୍ କାର୍ଡ', 'ଆଧାର'). NEVER write English/Latin letters or parenthetical translations like '(Video KYC)' in Hindi or Odia responses."
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
        self._preroll_buffer: deque[bytes] = deque(maxlen=4)   # 80ms speech-onset preservation buffer
        self._last_voice_ts: float = 0.0
        self._silence_task: Optional[asyncio.Task] = None

        # Concurrency & Playback controls
        self._pipeline_lock = asyncio.Lock()
        self._playback_task: Optional[asyncio.Task] = None
        self._is_playing: bool = False
        self._latest_pbx_transcript: str = ""

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
        """Invoked when PBX or caller hangs up, or upon automated closing."""
        if self.menu_machine.state == IVRState.CALL_END:
            return  # Already concluded
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

    def _stop_playback(self, reason: str = "barge_in") -> None:
        """Immediately interrupts active audio playback to Asterisk."""
        if self._playback_task and not self._playback_task.done():
            logger.info(f"[IVR] ⚡ Barge-in: cancelling playback task ({reason})")
            self._playback_task.cancel()
        self._is_playing = False

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
        self._stop_playback(reason="dtmf_keypad")

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
            f"7. Female Speaker Grammatical Gender: You are a female support specialist speaking with a female voice. In Hindi and any gendered language, ALWAYS use feminine first-person grammatical forms for yourself (e.g. 'कर सकती हूँ', 'सुन रही हूँ', 'मदद करूँगी', 'समझ रही हूँ', 'चेक कर रही हूँ' — NEVER use masculine forms like 'सकता हूँ' or 'रहा हूँ').\n"
            f"8. Native Script Pronunciation: When speaking in Hindi or Odia, write all technical terms and acronyms in the native script itself (e.g. in Hindi: 'केवाईसी', 'यूपीआई', 'ओटीपी', 'पैन कार्ड', 'आधार'; in Odia: 'କେୱାଇସି', 'ୟୁପିଆଇ', 'ଓଟିପି', 'ପ୍ୟାନ୍ କାର୍ଡ', 'ଆଧାର'). NEVER write Latin/English characters or parenthetical translations like '(Video KYC)' in Hindi or Odia responses."
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

        # When bot is actively speaking back to user, detect voice barge-in interruption
        if self._is_playing:
            if len(data) >= 4:
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                if rms > 650.0:
                    self._barge_in_voice_frames = getattr(self, "_barge_in_voice_frames", 0) + 1
                    if self._barge_in_voice_frames >= 2:  # ~40 ms of active speech
                        logger.info(f"[IVR] ⚡ Voice Barge-in detected (RMS={rms:.1f}): interrupting playback!")
                        self._stop_playback(reason="voice_barge_in")
                        self._barge_in_voice_frames = 0
                        self._pcm_buffer.clear()
                        self._pcm_buffer.extend(data)
                        self._last_voice_ts = time.monotonic()
                else:
                    self._barge_in_voice_frames = 0
            return

        silent = is_silence(data)

        if not silent:
            # If starting a new utterance, prepend pre-roll frames to preserve leading consonants (/p/, /t/, /k/, /s/)
            if len(self._pcm_buffer) == 0 and self._preroll_buffer:
                for frame in self._preroll_buffer:
                    self._pcm_buffer.extend(frame)
                self._preroll_buffer.clear()

            self._pcm_buffer.extend(data)
            self._last_voice_ts = time.monotonic()

            # Cancel any pending silence-triggered flush
            if self._silence_task and not self._silence_task.done():
                self._silence_task.cancel()

            # Hard cap — avoid unbounded accumulation
            if len(self._pcm_buffer) >= MAX_SPEECH_BYTES:
                await self._flush_utterance()
        else:
            # Maintain rolling 80ms pre-roll buffer when caller is paused/idle
            if len(self._pcm_buffer) == 0:
                self._preroll_buffer.append(data)

            # Schedule a flush after SILENCE_GAP_SECONDS of quiet
            if (
                len(self._pcm_buffer) >= MIN_SPEECH_BYTES
                and (self._silence_task is None or self._silence_task.done())
            ):
                self._silence_task = asyncio.create_task(
                    self._delayed_flush(SILENCE_GAP_SECONDS)
                )

    async def on_transcript(self, event: dict[str, Any]) -> None:
        """
        Handle external transcript from Asterisk PBX.
        If local PCM buffer has speech, local Whisper STT takes precedence.
        If local PCM buffer is empty or short, use PBX transcript directly
        so caller query is never dropped!
        """
        if self.menu_machine.state != IVRState.AI_CONVERSATION:
            return

        if self._is_playing:
            logger.info("[IVR] ⚡ Barge-in by PBX transcript while playing: interrupting audio!")
            self._stop_playback(reason="transcript_barge_in")

        text = (event.get("text") or event.get("transcript") or "").strip()
        if not text:
            return

        # Clear PCM buffer and cancel silence timer since PBX already gave us the full transcript
        self._pcm_buffer.clear()
        self._preroll_buffer.clear()
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()

        logger.info(f"[IVR] 🎙️ Processing PBX transcript directly: '{text}'")
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
        self._preroll_buffer.clear()

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
            if self._latest_pbx_transcript:
                user_text = self._latest_pbx_transcript
                self._latest_pbx_transcript = ""
                logger.info(f"[IVR] 🎙️ Local STT returned empty; fell back to PBX transcript: '{user_text}'")
            else:
                logger.debug("[IVR] STT returned empty/silence, skipping turn.")
                return

        logger.info(f"[IVR] 🎙️ Caller Voice Transcribed (Whisper lang={self._language}): '{user_text}'")
        await self._run_ai_and_reply(user_text)

    async def _run_ai_and_reply(self, user_text: str) -> None:
        """
        Conversational Phone Agent Core:
        1. Fast Telephony Protocols: handles pure greetings, closing farewells, hearing checks,
           and agent transfers instantly (<1ms).
        2. Dynamic Conversational Brain (Gemma 2B): conducts warm, empathetic, multi-turn
           conversation with full dialogue history and injected RAG banking knowledge.
        3. Resilient Fallback: if remote GPU/gateway is unavailable, falls back gracefully to
           the local IVR mini-model so the caller never experiences silence.
        """
        clean_user = (user_text or "").strip()
        lower_user = clean_user.lower()
        self._history.append(ChatMessage(role="user", content=clean_user))

        from providers.chat.ivr_mini_model import AUTONOMOUS_KNOWLEDGE, ivr_mini_brain

        # ── 1. Fast Telephony Protocol Signals (instant response) ─────────────
        # A. Appreciation / Closing turn (auto-hangup trigger)
        if ivr_mini_brain._is_closing_intent(lower_user):
            reply_text = AUTONOMOUS_KNOWLEDGE["appreciation"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["appreciation"]["en"])
            logger.info(f"[IVR] 📞 Closing farewell turn detected: '{clean_user}' -> '{reply_text}'")
            return await self._finalize_and_speak(user_text, reply_text)

        # B. Voice clarity / Hearing check
        if any(w in lower_user for w in ["hear me", "loud and clear", "awaaz", "awaz", "sound", "आवाज़", "slow down", "dheere", "धीरे", "ଶୁଣା"]):
            reply_text = AUTONOMOUS_KNOWLEDGE["voice_clarification"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["voice_clarification"]["en"])
            logger.info(f"[IVR] 🔊 Voice clarity check: '{clean_user}' -> '{reply_text}'")
            return await self._finalize_and_speak(user_text, reply_text)

        # C. Representative / Live human agent transfer
        if any(w in lower_user for w in ["representative", "agent", "executive", "human", "person", "operator", "talk to", "speak to", "एजेंट", "प्रतिनिधि", "ପ୍ରତିନିଧି", "ଏଜେଣ୍ଟ"]):
            reply_text = AUTONOMOUS_KNOWLEDGE["human_agent"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["human_agent"]["en"])
            logger.info(f"[IVR] 👤 Live human agent requested: '{clean_user}' -> '{reply_text}'")
            return await self._finalize_and_speak(user_text, reply_text)

        # D. Opening greeting on very first turn
        user_turn_count = len([m for m in self._history if m.role == "user"])
        if ivr_mini_brain._is_greeting_intent(lower_user) and user_turn_count <= 1:
            reply_text = AUTONOMOUS_KNOWLEDGE["greeting"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["greeting"]["en"])
            logger.info(f"[IVR] 👋 Opening greeting turn: '{clean_user}' -> '{reply_text}'")
            return await self._finalize_and_speak(user_text, reply_text)

        # ── 2. Conversational LLM Brain (Gemma with Multi-Turn Memory & RAG) ───
        reply_text = ""
        rag_context_str = ""

        # Step 2A: Query domain knowledge from pgvector
        try:
            async with async_session() as session:
                doc_repo = DocumentRepository(session)
                emb_service = EmbeddingService(provider_id="semantic-default")
                search_service = SemanticSearchService(doc_repo=doc_repo, embedding_service=emb_service)

                ivr_tenant = os.getenv("IVR_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")
                rag_threshold = float(os.getenv("RAG_MATCH_THRESHOLD", "0.55"))
                search_results = await search_service.search(tenant_id=ivr_tenant, query=clean_user, top_k=2)
                relevant_chunks = [c.content for c, score in search_results if score >= rag_threshold]
                if relevant_chunks:
                    rag_context_str = "\n\n".join(relevant_chunks)
                    logger.info(f"[IVR] ✓ Injected {len(relevant_chunks)} pgvector RAG chunks into prompt.")
        except Exception as rag_err:
            logger.debug(f"[IVR] pgvector lookup skipped: {rag_err}")

        # Ingest autonomous banking reference hint if relevant
        inferred_topic = ivr_mini_brain._infer_intent_with_history(lower_user, self._history)
        if inferred_topic and inferred_topic in AUTONOMOUS_KNOWLEDGE:
            topic_fact = AUTONOMOUS_KNOWLEDGE[inferred_topic].get(self.selected_language, AUTONOMOUS_KNOWLEDGE[inferred_topic]["en"])
            rag_context_str = f"{rag_context_str}\n\nBank Knowledge: {topic_fact}".strip()

        # Step 2B: Build Conversational Multi-Turn Prompt
        lang_name = {"en": "English", "hi": "Hindi", "or": "Odia"}.get(self.selected_language, "English")
        reference_block = f"\nRelevant Bank Reference:\n{rag_context_str}\nUse the above information to answer accurately." if rag_context_str else ""

        conversational_system_prompt = (
            f"You are a friendly, warm, and natural customer support specialist on a live phone call.\n"
            f"Guidelines for natural phone conversation:\n"
            f"1. Respond in conversational {lang_name} naturally like a real human customer care representative.\n"
            f"2. Keep responses brief, natural, and helpful (1 to 2 spoken sentences, under 28 words).\n"
            f"3. Use natural conversational markers when appropriate (e.g. 'Sure!', 'I understand', 'I would be glad to help', 'Ji bilkul', 'Haanji', 'Aap chinta mat kijiye').\n"
            f"4. Never use markdown, asterisks, bullet points, or numbering. Speak plain conversational sentences.\n"
            f"5. Maintain multi-turn context: refer to what the caller said earlier in the conversation.\n"
            f"6. When speaking Hindi or Odia, use feminine first-person grammatical forms for yourself (कर सकती हूँ, सुन रही हूँ) and write in the native script.\n"
            f"{reference_block}"
        )

        # Build rolling conversation history (last 6 turns) so Gemma has multi-turn memory!
        dialogue_messages = [ChatMessage(role="system", content=conversational_system_prompt)]
        recent_turns = [m for m in self._history if m.role in ("user", "assistant")][-6:]
        dialogue_messages.extend(recent_turns)

        gemma_req = NormalizedChatRequest(
            model_alias="gemma",
            messages=dialogue_messages,
            stream=False,
            max_tokens=45,
            temperature=0.6,
        )

        # Step 2C: Execute Conversational Turn via Gemma Gateway
        enable_gemma = os.getenv("ENABLE_GEMMA_OPEN_DOMAIN", "true").lower() in ("1", "true", "yes")
        if enable_gemma:
            try:
                logger.info(f"[IVR] 🗣️ Conversational turn with Gemma (history_len={len(recent_turns)}): '{clean_user[:60]}'...")
                gemma_res = await asyncio.wait_for(
                    self._orchestrator.execute(gemma_req),
                    timeout=3.5,
                )
                candidate = (gemma_res.choices[0].message.content or "").strip()
                if candidate and "mock response" not in candidate.lower():
                    reply_text = candidate
                    logger.info(f"[IVR] 🌟 Gemma conversational reply: '{reply_text}'")
            except Exception as gemma_err:
                logger.warning(f"[IVR] Gemma conversational turn error ({gemma_err}); falling back to mini-brain.")

        # Step 2D: Resilient Fallback to local mini-brain
        if not reply_text:
            reply_text = ivr_mini_brain.synthesize(
                user_query=clean_user,
                messages=self._history,
                rag_context=rag_context_str,
                language=self.selected_language,
                service_category=self.selected_service,
            )
            logger.info(f"[IVR] 🧠 Mini-brain fallback reply: '{reply_text}'")

        await self._finalize_and_speak(user_text, reply_text)

    async def _finalize_and_speak(self, user_text: str, reply_text: str) -> None:
        """Sanitizes text, updates history, triggers learning, and streams PCM audio."""
        reply_text = (reply_text or "").strip()
        if not reply_text or "mock response" in reply_text.lower():
            if self.selected_language == "hi":
                reply_text = "माफ़ कीजिये, आपकी आवाज़ साफ़ नहीं आई। क्या आप वीडियो केवाईसी या यूपीआई समस्या के बारे में पूछ रहे हैं?"
            elif self.selected_language == "or":
                reply_text = "କ୍ଷମା କରିବେ, ଆପଣଙ୍କ କଥା ସ୍ପଷ୍ଟ ଶୁଣାଗଲା ନାହିଁ। ଆପଣ ଭିଡିଓ କେୱାଇସି କିମ୍ବା ୟୁପିଆଇ ବିଷୟରେ ପଚାରୁଛନ୍ତି କି?"
            else:
                reply_text = "I'm sorry, I couldn't clearly hear that. Could you please repeat your query about Video KYC or UPI?"

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

        # Trim history to avoid context explosion
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

        # 3. TTS → raw PCM 8 kHz (clean spoken text of markdown/bullets/emojis)
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

        # 4. Send reply over WebSocket as cancellable playback task (supports voice barge-in)
        self._is_playing = True
        self._playback_task = asyncio.create_task(self._send_pcm_reply(pcm_reply))
        try:
            await self._playback_task
        except asyncio.CancelledError:
            logger.info(f"[IVR] ⚡ AI reply playback interrupted by caller barge-in (uuid={self.call_uuid}).")
        finally:
            self._is_playing = False
            self._playback_task = None
            self._pcm_buffer.clear()
            self._last_voice_ts = time.monotonic()

        # 5. Auto Cut after Thank You / Closing turn
        if is_closing_turn(user_text, reply_text):
            logger.info(
                f"[IVR] 📞 Closing farewell turn detected ('{user_text}' → '{reply_text[:50]}...'). "
                f"Auto-cutting call (uuid={self.call_uuid})..."
            )
            await asyncio.sleep(0.8)
            try:
                if self.call_uuid:
                    await self._ws.send_text(json.dumps({"event": "hangup", "uuid": self.call_uuid}))
                    await self._ws.send_text(json.dumps({"command": "hangup", "uuid": self.call_uuid}))
                    await self._ws.send_text(json.dumps({"action": "hangup", "uuid": self.call_uuid}))
                else:
                    await self._ws.send_text(json.dumps({"event": "hangup"}))
            except Exception as hangup_err:
                logger.warning(f"[IVR] Failed to send hangup signal to PBX: {hangup_err}")

            self.on_call_end({"uuid": self.call_uuid, "reason": "auto_cut_after_closing"})

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

def is_closing_turn(user_text: str, reply_text: str) -> bool:
    """
    Detects if this conversational turn is a closing/farewell turn
    (e.g. caller saying 'thank you', 'thanks', 'dhanyawad', 'bye', or AI delivering closing greeting).
    """
    u = (user_text or "").lower().strip()
    r = (reply_text or "").lower().strip()

    closing_user_keywords = [
        "thank", "thanks", "thx", "dhanyawad", "dhanyavad", "shukriya", "dhanyabad",
        "धन्यवाद", "शुक्रिया", "bye", "goodbye", "alvida", "अलविदा", "ଧନ୍ୟବାଦ",
        "nothing else", "no more", "that is all", "thats all", "that's all",
        "kuch nahi", "kuch nhi", "aur nahi", "bas itna", "bas thank", "aur kuch nahi",
        "ଆଉ କିଛି ନାହିଁ", "କିଛି ନାହିଁ", "कुछ नहीं", "बस इतना ही", "नो थैंक", "थैंक यू"
    ]
    if any(k in u for k in closing_user_keywords):
        return True

    closing_reply_keywords = [
        "have a wonderful day", "have a great day",
        "दिन शुभ हो", "शुभ दिन",
        "ଦିନଟି ଶୁଭମୟ",
        "हमारी ग्राहक सेवा में संपर्क करने के लिए आभार",
        "thank you for contacting our customer support",
        "ଆମ ଗ୍ରାହକ ସେବା ସହିତ ଯୋଗାଯୋଗ କରିଥିବାରୁ କୃତଜ୍ଞତା",
        "you are most welcome"
    ]
    if any(k.lower() in r for k in closing_reply_keywords):
        return True

    return False


def _clean_spoken_text(text: str) -> str:
    """
    Cleans text of markdown markers, asterisks, bullet points, brackets, URLs,
    and parenthetical English translations in Indian language sentences
    so that TTS speaks smoothly, naturally, and with flawless native pronunciation.
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
    # Remove parenthetical English translations like (Video KYC), (Physical PAN Card)
    text = re.sub(r"\([A-Za-z0-9\s/\-_\.]+\)", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove emojis and miscellaneous symbols so TTS speaks cleanly
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Remove technical metadata headers from spoken speech
    text = re.sub(r"(?i)Audio\s*Prompt\s*ID\s*:\s*[A-Z0-9\-_]+", "", text)
    text = re.sub(r"(?i)(?:English|Hindi|Odia)\s*Voice\s*Transcript\s*:\s*", "", text)
    text = re.sub(r"(?i)(?:Document\s*ID|Category|Source|Title)\s*:\s*[^\n\.,]+", "", text)

    # Strip quotes and stray numbering anywhere in the speech
    text = text.replace('"', '').replace("'", '')
    text = re.sub(r'(?:^|\s+)\d+[\.\)]\s*', ' ', text).strip()

    # Deduplicate consecutive identical sentences (e.g. "Sentence A. Sentence A." -> "Sentence A.")
    pattern = r'([^.!?।\n]+[.!?।])\s*(?:\1\s*)+'
    text = re.sub(pattern, r'\1 ', text).strip()

    # Phonetic normalization for Hindi speech (convert Latin acronyms to Devanagari)
    has_hindi = any('\u0900' <= ch <= '\u097f' for ch in text)
    if has_hindi:
        hindi_loanwords = [
            (r"\bvideo\s*kyc\b", "वीडियो केवाईसी"),
            (r"\bkyc\b", "केवाईसी"),
            (r"\bupi\b", "यूपीआई"),
            (r"\botp\b", "ओटीपी"),
            (r"\bpan\s*card\b", "पैन कार्ड"),
            (r"\bpan\b", "पैन"),
            (r"\baadhaar\b", "आधार"),
            (r"\baadhar\b", "आधार"),
            (r"\batm\b", "एटीएम"),
            (r"\bpin\b", "पिन"),
            (r"\bsms\b", "एसएमएस"),
            (r"\butr\b", "यूटीआर"),
            (r"\bemi\b", "ईएमआई"),
            (r"\bnetbanking\b", "नेटबैंकिंग"),
            (r"\bapp\b", "ऐप"),
            (r"\bpassword\b", "पासवर्ड"),
            (r"\brefund\b", "रिफंड"),
            (r"\bpayment\b", "पेमेंट"),
        ]
        for pat, rep in hindi_loanwords:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

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
