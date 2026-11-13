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
import re
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
from services.conversation.response_generator import RAGFirstResponseGenerator
from services.knowledge.knowledge_learning_service import KnowledgeLearningService
from storage.database import async_session
from telephony.pcm_utils import (
    PBX_CHUNK_BYTES,
    PBX_SAMPLE_RATE,
    audio_bytes_to_pcm8k,
    chunk_pcm,
    generate_processing_pulse_pcm,
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
from audio_intelligence import AudioIntelligencePipeline, AudioQualityReport, EnhancedAudioResult
from conversation_intelligence.transcript_confidence import mask_sensitive_text
from decision_engine.decision_manager import Action
from telephony.intelligence import ConversationState, TelephonyIntelligencePipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ──────────────────────────────────────────────────────────────────────────────

# How much PCM to accumulate before running STT (avoid dropping crisp short words like 'yes'/'no')
MIN_SPEECH_BYTES = int(PBX_SAMPLE_RATE * 2 * 0.25)    # 0.25 second minimum (4,000 bytes)
MIN_SHORT_REPLY_BYTES = int(PBX_SAMPLE_RATE * 2 * 0.12)  # 0.12 second minimum for crisp 1–5 digits / yes-no
MAX_SPEECH_BYTES = PBX_SAMPLE_RATE * 2 * 8           # 8 seconds hard cap

# Silence gap before closing an utterance for STT (0.65s feels responsive without chopping brief pauses)
SILENCE_GAP_SECONDS = float(os.getenv("IVR_SILENCE_GAP_SECONDS", "0.65"))

# Delay between sending 20 ms PCM reply frames (throttles playback to real-time)
FRAME_INTERVAL_SECONDS = 0.020   # 20 ms

# Barge-in speech protection & responsive ducking ("slowdown AI voice and start listening")
BARGE_IN_COOLDOWN_SECONDS = float(os.getenv("IVR_BARGE_IN_COOLDOWN_SECONDS", "0.25"))
BARGE_IN_DUCK_RMS_THRESHOLD = float(os.getenv("IVR_BARGE_IN_DUCK_RMS_THRESHOLD", "420.0"))
BARGE_IN_RMS_THRESHOLD = float(os.getenv("IVR_BARGE_IN_RMS_THRESHOLD", "650.0"))
BARGE_IN_MIN_CONSECUTIVE_FRAMES = int(os.getenv("IVR_BARGE_IN_MIN_CONSECUTIVE_FRAMES", "2"))  # ~40 ms strong speech
BARGE_IN_DUCK_GAIN = float(os.getenv("IVR_BARGE_IN_DUCK_GAIN", "0.22"))
BARGE_IN_DUCK_SLOWDOWN_FACTOR = float(os.getenv("IVR_BARGE_IN_DUCK_SLOWDOWN_FACTOR", "1.35"))

# Gentle waiting / processing chime played while STT -> RAG/LLM -> TTS is running
PROCESSING_SOUND_ENABLED = os.getenv("IVR_PROCESSING_SOUND_ENABLED", "true").lower() in ("1", "true", "yes")
PROCESSING_SOUND_INITIAL_DELAY_SECONDS = float(os.getenv("IVR_PROCESSING_SOUND_DELAY_SECONDS", "0.35"))
PROCESSING_SOUND_INTERVAL_SECONDS = float(os.getenv("IVR_PROCESSING_SOUND_INTERVAL_SECONDS", "1.4"))

# Model used for chat; override via env / config
DEFAULT_CHAT_MODEL = os.getenv("IVR_CHAT_MODEL", "ivr-rag-mini")
GEMMA_MODEL_ALIAS = os.getenv("GEMMA_MODEL_ALIAS", "gemma4")

_LEGACY_LOCAL_TOOLS_REPLACED_BY_REMOTE = frozenset(
    {"count_open_purchase_requests", "get_customer_info"}
)
_CORE_IVR_TOOLS = frozenset(
    {
        "search_customer",
        "get_customer_details",
        "get_order_status",
        "get_customer_orders",
        "search_existing_complaints",
        "create_complaint",
        "register_complaint",
        "get_complaint_status",
        "get_customer_complaints",
    }
)
_CONDITIONAL_IVR_TOOL_KEYWORDS = (
    ("get_order_tracking", ("track", "tracking", "awb", "courier", "shipment", "shipped", "delivery", "dispatch", "ট্র্যাক", "ट्रैक")),
    ("get_payment_status", ("payment", "paid", "refund", "upi", "card", "transaction", "money", "charge", "पेमेंट", "रिफंड")),
    ("get_invoice_details", ("invoice", "bill", "receipt", "gst", "tax", "बिल", "इनवॉइस")),
    ("get_customer_order_summary", ("summary", "all orders", "history", "total spend", "recent orders")),
    ("escalate_complaint", ("escalate", "escalation", "supervisor", "manager", "urgent", "unresolved", "no callback", "delay")),
    ("add_complaint_note", ("note", "comment", "remark", "follow up", "followup", "add info")),
    ("update_complaint", ("update", "modify", "change", "edit", "resolution", "close complaint")),
    ("call_external_api", ("api", "webhook", "external", "sync", "crm", "trigger")),
    ("search_knowledge_documents", ("policy", "rule", "faq", "how to", "warranty", "return window", "document", "नियम")),
    ("search_database_records", ("search database", "find record", "lookup table", "product sku", "catalog")),
)


def _supports_tool_calling_safe(alias: str) -> Optional[bool]:
    """Return True/False if provider_registry knows the alias, or None if uninitialized/unknown."""
    try:
        from platform_core.contracts.capabilities import ModelCapability
        from providers.registry import provider_registry

        if not provider_registry.models:
            return None
        meta = provider_registry.get_model_metadata(alias)
        if meta.provider_id == "windows_ollama" and "gemma2" in (meta.underlying_model or "").lower():
            return False
        return meta.supports(ModelCapability.TOOL_CALLING)
    except Exception:
        return None


def _resolve_ivr_tool_model_alias(
    preferred_alias: str,
    has_tools: bool,
    settings_alias: Optional[str] = None,
) -> str:
    """Resolve an IVR model alias that is guaranteed tool-capable when tools are attached."""
    if not has_tools:
        return preferred_alias

    candidates: list[str] = []
    for cand in (preferred_alias, settings_alias, "gemma4", "gemma", "general-chat"):
        if cand and cand not in candidates:
            candidates.append(cand)

    # First respect explicit non-gemma2 preferred alias if tool-capable
    pref_ok = _supports_tool_calling_safe(preferred_alias)
    if pref_ok is True:
        return preferred_alias
    if pref_ok is None:
        if preferred_alias and preferred_alias != "ivr-rag-mini" and "gemma2" not in preferred_alias.lower():
            return preferred_alias
        fallback = settings_alias or "gemma4"
        return "gemma4" if (fallback == "ivr-rag-mini" or "gemma2" in fallback.lower()) else fallback

    # Registry initialized and preferred_alias does not support tools -> pick first tool-capable alias
    for cand in candidates[1:]:
        if _supports_tool_calling_safe(cand) is True:
            logger.info(
                f"[IVR] Switching model alias from '{preferred_alias}' to tool-capable '{cand}'."
            )
            return cand

    return settings_alias or "gemma4"


def _select_ivr_session_tools(
    all_tool_schemas: list[dict[str, Any]],
    user_text: str,
    history: list[ChatMessage],
    service_category: Optional[str],
    remote_tools_active: bool,
) -> Optional[list[dict[str, Any]]]:
    """Keep live call tool schemas focused so local Gemma keeps high accuracy and low latency."""
    if not all_tool_schemas:
        return None

    filtered = all_tool_schemas
    if remote_tools_active:
        filtered = [
            s
            for s in all_tool_schemas
            if (s.get("function") or {}).get("name") not in _LEGACY_LOCAL_TOOLS_REPLACED_BY_REMOTE
        ]

    if len(filtered) <= 10:
        return filtered

    recent_text = " ".join(
        [user_text or "", service_category or "", *((m.content or "") for m in history[-4:])]
    ).lower()

    selected_names = set(_CORE_IVR_TOOLS)
    for tool_name, keywords in _CONDITIONAL_IVR_TOOL_KEYWORDS:
        if any(kw in recent_text for kw in keywords):
            selected_names.add(tool_name)

    focused = [s for s in filtered if (s.get("function") or {}).get("name") in selected_names]
    return focused or filtered

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
        self.ignore_pbx_transcript: bool = PBX_IGNORE_TRANSCRIPT
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
        self._preroll_buffer: deque[bytes] = deque(maxlen=10)   # 200ms speech-onset preservation buffer
        self._last_voice_ts: float = 0.0
        self._silence_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None

        # Concurrency & Playback controls
        self._pipeline_lock = asyncio.Lock()
        self._playback_lock = asyncio.Lock()
        self._playback_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._is_playing: bool = False
        self._is_synthesizing: bool = False
        self._playback_started_ts: float = 0.0
        self._barge_in_voice_frames: int = 0
        self._barge_in_duck_frames: int = 0
        self._barge_in_preroll: list[bytes] = []
        self._is_ducking: bool = False
        self._duck_gain: float = 1.0
        self._cancelled_by_barge_in: bool = False
        self._barge_in_captured: bool = False
        self._last_playback_end_ts: float = 0.0
        self._latest_pbx_transcript: str = ""
        self._order_dtmf_buffer: str = ""

        # Milestone 1: Audio Intelligence pipeline & latest diagnostics
        self.audio_pipeline = AudioIntelligencePipeline(sample_rate=PBX_SAMPLE_RATE)
        self.last_audio_quality: Optional[AudioQualityReport] = None
        self.last_enhanced_audio: Optional[EnhancedAudioResult] = None

        # Milestones 2–5: Hybrid Voice Intelligence coordinator & session state
        self.intel_pipeline = TelephonyIntelligencePipeline()
        self.conv_state = ConversationState(
            call_uuid="unknown",
            tenant_id=self.tenant_id,
            language=self.selected_language,
        )
        self._last_stt_result: Optional[dict[str, Any]] = None
        self._last_turn_pcm: bytes = b""
        self._last_turn_intel_ctx: Optional[dict[str, Any]] = None

        # RAG-First Conversational AI Engine & Continuous Knowledge Learning
        self.response_generator = RAGFirstResponseGenerator()
        self.knowledge_learner = KnowledgeLearningService()
        self._last_retrieval_tier: str = "strong"

    # ── call lifecycle handlers ───────────────────────────────────────────────

    def on_call_start(self, event: dict[str, Any]) -> None:
        """Invoked when PBX notifies of incoming call connection."""
        self.call_uuid = event.get("uuid") or event.get("call_uuid") or "unknown"
        if event.get("authorized_customer_id") or event.get("customer_reference"):
            self.authorized_customer_id = str(
                event.get("authorized_customer_id") or event.get("customer_reference")
            ).strip()
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

        self.conv_state = ConversationState(
            call_uuid=self.call_uuid,
            tenant_id=self.tenant_id,
            language=self.selected_language,
        )
        self._last_stt_result = None
        self._last_turn_pcm = b""
        self._last_turn_intel_ctx = None
        self._order_dtmf_buffer = ""

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
        self._cancel_processing_sound()
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._pcm_buffer.clear()

        # Record end-of-call resolution telemetry before clearing session state
        try:
            self.intel_pipeline.record_call_end(self.conv_state)
        except Exception as exc:
            logger.debug(f"[IVR] Non-fatal call end telemetry error: {exc}")

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
            self._cancelled_by_barge_in = reason == "voice_barge_in"
            self._playback_task.cancel()
        self._cancel_processing_sound()
        self._is_playing = False
        self._is_ducking = False
        self._duck_gain = 1.0
        self._barge_in_voice_frames = 0
        self._barge_in_duck_frames = 0
        self._barge_in_preroll.clear()

    def _start_processing_sound(self) -> None:
        """Start gentle background waiting chimes if a turn takes noticeably long."""
        if not PROCESSING_SOUND_ENABLED:
            return
        if self.menu_machine.state not in (IVRState.AI_CONVERSATION, IVRState.FEEDBACK):
            return
        if self._processing_task and not self._processing_task.done():
            return
        self._processing_task = asyncio.create_task(self._processing_sound_loop())

    def _cancel_processing_sound(self) -> None:
        """Synchronously request cancellation of any waiting/processing sound task."""
        task = self._processing_task
        self._processing_task = None
        if task and not task.done():
            task.cancel()

    async def _stop_processing_sound(self) -> None:
        """Cancel waiting/processing sound and wait until playback lock is released."""
        task = self._processing_task
        self._processing_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _processing_sound_loop(self) -> None:
        """Play a soft, non-intrusive dual-tone pulse while STT -> AI -> TTS is in flight."""
        try:
            await asyncio.sleep(max(0.0, PROCESSING_SOUND_INITIAL_DELAY_SECONDS))
            pulse_pcm = generate_processing_pulse_pcm()
            first_pulse = True
            while (
                self.menu_machine.state in (IVRState.AI_CONVERSATION, IVRState.FEEDBACK)
                and not self._is_playing
            ):
                if first_pulse:
                    logger.info(f"[IVR] 🎵 Playing waiting/processing sound (uuid={self.call_uuid})...")
                    first_pulse = False
                await self._stream_pcm_frames(pulse_pcm, verbose_logs=False)
                await asyncio.sleep(max(0.1, PROCESSING_SOUND_INTERVAL_SECONDS))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"[IVR] Processing sound loop ended: {exc}")

    def _min_speech_bytes(self) -> int:
        """Allow shorter utterances (e.g. '5', 'yes', 'no') during confirmation, feedback, or digit collection."""
        if (
            self.menu_machine.state == IVRState.FEEDBACK
            or self.conv_state.awaiting_feedback
            or self.conv_state.awaiting_resolution_confirmation
            or self.conv_state.awaiting_order_id_digits
        ):
            return MIN_SHORT_REPLY_BYTES
        return MIN_SPEECH_BYTES

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

        # Collect 1-5 satisfaction rating via keypad when awaiting feedback
        if (
            (self.menu_machine.state == IVRState.FEEDBACK or self.conv_state.awaiting_feedback)
            and digit in {"1", "2", "3", "4", "5"}
        ):
            if self._silence_task and not self._silence_task.done():
                self._silence_task.cancel()
            self._pcm_buffer.clear()
            self._preroll_buffer.clear()
            logger.info(f"[IVR] ⭐ Received DTMF satisfaction rating '{digit}/5'")
            asyncio.create_task(self._process_dtmf_order_digits(digit))
            return

        # Collect 4-digit Order ID via keypad when awaiting order digits in AI conversation
        # (checked before menu_machine so digits containing '9' do not trigger human transfer mid-entry)
        if (
            self.menu_machine.state == IVRState.AI_CONVERSATION
            and self.conv_state.awaiting_order_id_digits
            and len(digit) == 1
            and digit.isdigit()
        ):
            # Cancel any buffered voice flush so spoken noise doesn't race with keypad digits
            if self._silence_task and not self._silence_task.done():
                self._silence_task.cancel()
            self._pcm_buffer.clear()
            self._preroll_buffer.clear()

            self._order_dtmf_buffer += digit
            logger.info(
                f"[IVR] 🔢 Collected Order ID DTMF digit '{digit}' "
                f"(buffer='{self._order_dtmf_buffer}')"
            )
            if len(self._order_dtmf_buffer) >= 4:
                digits4 = self._order_dtmf_buffer[-4:]
                self._order_dtmf_buffer = ""
                asyncio.create_task(self._process_dtmf_order_digits(digits4))
            return

        new_state, action = self.menu_machine.process_dtmf(digit)
        logger.info(f"[IVR] Menu Transition -> State={new_state}, Action={action}")

        if action == "LANGUAGE_SELECTED":
            self.selected_language = self.menu_machine.language
            self._language = self.selected_language
            self.conv_state.language = self.selected_language
            logger.info(f"[IVR] Language set to: '{self.selected_language}'")
            asyncio.create_task(self._play_service_menu())

        elif action == "SERVICE_SELECTED":
            self.selected_service = self.menu_machine.service_key
            self.conv_state.service_category = self.selected_service
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
            self._barge_in_captured = False
            self._last_voice_ts = time.monotonic()
            self._last_playback_end_ts = self._last_voice_ts

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
        self._playback_started_ts = time.monotonic()
        self._barge_in_voice_frames = 0
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
                pcm_reply = await asyncio.to_thread(_tts_output_to_pcm8k, tts_bytes)
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
        if self.menu_machine.state not in (IVRState.AI_CONVERSATION, IVRState.FEEDBACK):
            return

        # Ignore microphone frames while synthesizing TTS or while an active STT/LLM/TTS turn is running
        # (unless barge-in playback is active OR we just captured a barge-in and are actively listening)
        if (self._is_synthesizing or (self._pipeline_lock.locked() and not self._barge_in_captured)) and not self._is_playing:
            return

        # When bot is actively speaking back to user:
        # 1) Softly duck & slow down AI voice on first signs of caller speech (RMS > BARGE_IN_DUCK_RMS_THRESHOLD)
        # 2) Cancel AI playback and start listening immediately on sustained speech
        if self._is_playing:
            if (
                len(data) >= 4
                and (time.monotonic() - self._playback_started_ts) >= BARGE_IN_COOLDOWN_SECONDS
            ):
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32)
                rms = float(np.sqrt(np.mean(samples ** 2)))
                if rms > BARGE_IN_DUCK_RMS_THRESHOLD:
                    self._barge_in_duck_frames += 1
                    self._barge_in_preroll.append(data)
                    if not self._is_ducking:
                        self._is_ducking = True
                        logger.debug(f"[IVR] 🔉 Caller speech onset (RMS={rms:.1f}): ducking/slowing AI voice.")

                    if rms > BARGE_IN_RMS_THRESHOLD:
                        self._barge_in_voice_frames += 1
                    else:
                        self._barge_in_voice_frames = 0

                    soft_trigger_frames = max(4, BARGE_IN_MIN_CONSECUTIVE_FRAMES * 2)
                    if (
                        self._barge_in_voice_frames >= BARGE_IN_MIN_CONSECUTIVE_FRAMES
                        or self._barge_in_duck_frames >= soft_trigger_frames
                    ):
                        logger.info(f"[IVR] ⚡ Voice Barge-in detected (RMS={rms:.1f}): interrupting playback & listening!")
                        barge_frames = list(self._barge_in_preroll)
                        self._stop_playback(reason="voice_barge_in")
                        self._barge_in_captured = True
                        self._pcm_buffer.clear()
                        for frame in self._preroll_buffer:
                            self._pcm_buffer.extend(frame)
                        self._preroll_buffer.clear()
                        for frame in barge_frames:
                            self._pcm_buffer.extend(frame)
                        self._last_voice_ts = time.monotonic()
                else:
                    self._barge_in_voice_frames = 0
                    self._barge_in_duck_frames = 0
                    self._is_ducking = False
                    self._barge_in_preroll.clear()
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

            # Hard cap — spawn flush without blocking the WebSocket receive loop
            if len(self._pcm_buffer) >= MAX_SPEECH_BYTES:
                self._spawn_flush_utterance()
        else:
            # Maintain rolling 80ms pre-roll buffer when caller is paused/idle
            if len(self._pcm_buffer) == 0:
                self._preroll_buffer.append(data)

            # Schedule a flush after SILENCE_GAP_SECONDS of quiet
            if (
                len(self._pcm_buffer) >= self._min_speech_bytes()
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
        if self.menu_machine.state not in (IVRState.AI_CONVERSATION, IVRState.FEEDBACK):
            return

        # If PBX transcripts are configured to be ignored, drop immediately to protect local Whisper STT
        if self.ignore_pbx_transcript:
            logger.debug(f"[IVR] 🔕 Ignoring PBX transcript event (PBX_IGNORE_TRANSCRIPT=True): {event.get('text')}")
            return

        text = (event.get("text") or event.get("transcript") or "").strip()
        if not text or not re.search(r'\w', text):
            logger.debug(f"[IVR] Ignoring empty or noise/punctuation PBX transcript: '{text}'")
            return

        # For non-English calls (Hindi, Odia), Asterisk PBX transcripts are English-only and must be ignored
        if self.selected_language in ("hi", "or"):
            logger.info(f"[IVR] 🛑 Discarding external English PBX transcript for {self.selected_language} session: '{text}' (relying on local Whisper)")
            return

        # Filter out stray stopwords/filler noise from PBX (e.g. 'the', 'a', 'a ago', 'we're not not')
        clean_words = [
            w for w in re.findall(r"\b[a-zA-Z]+\b", text.lower())
            if w not in {"the", "a", "an", "and", "or", "to", "of", "in", "it", "is", "by", "he", "we", "not", "i", "you", "ago"}
        ]
        if not clean_words or (len(clean_words) == 1 and len(clean_words[0]) < 3):
            logger.info(f"[IVR] 🛑 Filtering out stray PBX stopword/noise transcript: '{text}'")
            return

        # If local PCM buffer has speech accumulated, preserve caller's audio for local Whisper STT
        if len(self._pcm_buffer) > 0:
            logger.info(f"[IVR] 🎙️ Preserving local PCM speech ({len(self._pcm_buffer)} bytes) over PBX transcript '{text}'")
            self._latest_pbx_transcript = text
            return

        if self._is_playing:
            logger.info("[IVR] ⚡ Barge-in by PBX transcript while playing: interrupting audio!")
            self._stop_playback(reason="transcript_barge_in")

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
        # Detach timer handle before running STT/AI/TTS so subsequent speech cannot cancel this turn
        self._silence_task = None
        self._spawn_flush_utterance()

    async def _process_dtmf_order_digits(self, digits4: str) -> None:
        """Serialized handler for 4-digit DTMF Order ID entry to prevent concurrent STT/TTS collisions."""
        async with self._pipeline_lock:
            await self._run_ai_and_reply(digits4)

    def _spawn_flush_utterance(self) -> Optional[asyncio.Task]:
        """Snapshot accumulated speech synchronously and schedule serialized STT -> AI -> TTS task."""
        if len(self._pcm_buffer) < self._min_speech_bytes():
            self._pcm_buffer.clear()
            self._barge_in_captured = False
            return None

        pcm_snapshot = bytes(self._pcm_buffer)
        captured_at = time.monotonic()
        from_barge_in = self._barge_in_captured
        self._pcm_buffer.clear()
        self._preroll_buffer.clear()
        self._barge_in_captured = False
        self._flush_task = asyncio.create_task(
            self._run_flush_with_lock(pcm_snapshot, captured_at=captured_at, from_barge_in=from_barge_in)
        )
        return self._flush_task

    async def _run_flush_with_lock(
        self,
        pcm_snapshot: bytes,
        *,
        captured_at: float = 0.0,
        from_barge_in: bool = False,
    ) -> None:
        async with self._pipeline_lock:
            if not from_barge_in and captured_at > 0 and captured_at < self._last_playback_end_ts:
                logger.info("[IVR] 🔇 Discarding stale pre-playback audio snapshot queued behind active turn.")
                return
            await self._run_stt_ai_tts(pcm_snapshot)

    async def _flush_utterance(self) -> None:
        """Run STT → AI → TTS for whatever is in the buffer."""
        if len(self._pcm_buffer) < self._min_speech_bytes():
            self._pcm_buffer.clear()
            self._barge_in_captured = False
            return

        pcm_snapshot = bytes(self._pcm_buffer)
        captured_at = time.monotonic()
        from_barge_in = self._barge_in_captured
        self._pcm_buffer.clear()
        self._preroll_buffer.clear()
        self._barge_in_captured = False
        await self._run_flush_with_lock(pcm_snapshot, captured_at=captured_at, from_barge_in=from_barge_in)

    async def _run_stt_ai_tts(self, pcm: bytes) -> None:
        # 1. Audio Intelligence (Milestone 1): analyze & enhance while preserving original PCM
        quality = await self.audio_pipeline.analyze(pcm)
        self.last_audio_quality = quality
        logger.info(
            f"[IVR] 🎧 Audio Quality: label={quality.quality_label} score={quality.audio_quality_score:.2f} "
            f"snr={quality.snr_db:.1f}dB rms={quality.rms:.1f} vad={quality.vad_probability:.2f} "
            f"speech={quality.speech_duration_ms:.0f}ms clip={quality.clipping_pct:.2f}%"
        )

        # Reject weak or sub-syllabic non-speech noise bursts before running Whisper
        expecting_short_reply = bool(
            self.menu_machine.state == IVRState.FEEDBACK
            or self.conv_state.awaiting_feedback
            or self.conv_state.awaiting_resolution_confirmation
            or self.conv_state.awaiting_order_id_digits
        )
        rms_floor = 60.0 if expecting_short_reply else 90.0
        min_speech_ms = 80.0 if expecting_short_reply else 160.0
        if (not quality.is_speech_present and quality.rms < rms_floor) or (
            quality.speech_duration_ms < min_speech_ms and quality.rms < 250.0
        ):
            logger.info("[IVR] 🔇 Audio Intelligence rejected non-speech noise burst; skipping STT.")
            return

        self._start_processing_sound()
        try:
            enhanced_result = await self.audio_pipeline.enhance(pcm, quality=quality)
            self.last_enhanced_audio = enhanced_result
            stt_pcm = enhanced_result.segmented_pcm or enhanced_result.enhanced_pcm or pcm

            # 2. STT — wrap PCM as minimal WAV so AudioService can read it
            self._last_stt_result = None
            self._last_turn_pcm = stt_pcm
            wav_bytes = _raw_pcm_to_wav(stt_pcm, sample_rate=PBX_SAMPLE_RATE)
            try:
                stt_result = await self._audio_service.transcribe(
                    audio_bytes=wav_bytes,
                    filename="call_audio.wav",
                    language=self._language,
                )
                self._last_stt_result = stt_result if isinstance(stt_result, dict) else {"text": str(stt_result or "")}
                user_text = (self._last_stt_result.get("text") or "").strip()
            except Exception as exc:
                logger.error(f"[IVR] STT error: {exc}")
                return

            if user_text and not expecting_short_reply:
                norm_utt = re.sub(r"[^a-z0-9\s]", "", user_text.lower()).strip()
                if norm_utt in {"hmm", "hm", "mm", "mhm", "uh", "um", "ah", "oh", "huh", "er"}:
                    logger.info(f"[IVR] 🔇 Ignoring non-lexical filler utterance '{user_text}'.")
                    return
                if (
                    norm_utt in {"thank you", "thanks", "thanks for watching", "thank you very much", "bye", "you", "okay"}
                    and (quality.speech_duration_ms < 500.0 or quality.rms < 150.0)
                ):
                    logger.info(
                        f"[IVR] 🔇 Ignoring low-energy/short Whisper hallucination '{user_text}' "
                        f"(speech={quality.speech_duration_ms:.0f}ms, rms={quality.rms:.1f})."
                    )
                    return

            if not user_text or not re.search(r'\w', user_text):
                if (
                    not self.ignore_pbx_transcript
                    and self.selected_language not in ("hi", "or")
                    and self._latest_pbx_transcript
                    and re.search(r'\w', self._latest_pbx_transcript)
                ):
                    user_text = self._latest_pbx_transcript
                    self._latest_pbx_transcript = ""
                    self._last_stt_result = {"text": user_text, "language": self._language}
                    logger.info(f"[IVR] 🎙️ Local STT returned empty/noise; fell back to PBX transcript: '{user_text}'")
                elif quality.is_speech_present and quality.speech_duration_ms >= 350:
                    clarify_prompts = {
                        "hi": "माफ़ कीजिए, आपकी आवाज़ साफ़ समझ नहीं आई। कृपया दोबारा बोलें।",
                        "or": "କ୍ଷମା କରିବେ, ଆପଣଙ୍କ କଥା ସ୍ପଷ୍ଟ ଭାବରେ ଶୁଣାଗଲା ନାହିଁ। ଦୟାକରି ପୁଣି କୁହନ୍ତୁ।",
                        "en": "Sorry, I could not hear that clearly. Could you please repeat?",
                    }
                    clarify_reply = clarify_prompts.get(
                        self.selected_language or self._language,
                        clarify_prompts["en"],
                    )
                    logger.info(
                        f"[IVR] STT returned empty despite speech ({quality.speech_duration_ms:.0f}ms); "
                        f"prompting caller to repeat ({self.selected_language or self._language})."
                    )
                    await self._finalize_and_speak(user_text="", reply_text=clarify_reply)
                    return
                else:
                    logger.debug(f"[IVR] STT returned noise/silence ('{user_text}'), skipping turn.")
                    return

            logger.info(
                f"[IVR] 🎙️ Caller Voice Transcribed (Whisper lang={self._language}): "
                f"'{mask_sensitive_text(user_text)}'"
            )
            await self._run_ai_and_reply(user_text)
        finally:
            await self._stop_processing_sound()

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
        if not clean_user or not re.search(r'\w', clean_user):
            logger.debug(f"[IVR] Ignoring empty or noise/punctuation transcript: '{clean_user}'")
            return

        started_here = not (self._processing_task and not self._processing_task.done())
        if started_here:
            self._start_processing_sound()
        try:
            await self._run_ai_and_reply_inner(clean_user)
        finally:
            if started_here:
                await self._stop_processing_sound()

    async def _run_ai_and_reply_inner(self, clean_user: str) -> None:

        lower_user = clean_user.lower()
        self._history.append(ChatMessage(role="user", content=clean_user))

        # ── 0. Hybrid Voice Intelligence Pipeline (Milestones 2–5) ────────────
        stt_input = self._last_stt_result or {"text": clean_user, "language": self.selected_language}
        if not stt_input.get("text"):
            stt_input = dict(stt_input, text=clean_user)
        self._last_stt_result = None
        turn_pcm = self._last_turn_pcm
        self._last_turn_pcm = b""

        self.conv_state.language = self.selected_language
        self.conv_state.service_category = self.selected_service
        if self.call_uuid and self.conv_state.call_uuid in ("unknown", "", None):
            self.conv_state.call_uuid = self.call_uuid

        pkg, plan = await self.intel_pipeline.process_and_plan(
            stt_result=stt_input,
            state=self.conv_state,
            audio=turn_pcm,
            audio_quality=self.last_audio_quality,
            acoustic_features=(
                self.last_enhanced_audio.acoustic_features
                if self.last_enhanced_audio is not None
                else None
            ),
        )
        self._last_turn_intel_ctx = plan.intelligence_context
        effective_user = pkg.transcript.normalized_text or clean_user
        lower_user = effective_user.lower()

        if plan.trigger_transfer_event:
            self.menu_machine.state = IVRState.ESCALATED
            await self._finalize_and_speak(effective_user, plan.text)
            await self._emit_transfer_event(plan.trigger_transfer_event)
            return

        if not plan.use_llm and plan.text:
            if plan.action == Action.FEEDBACK and self.conv_state.awaiting_feedback:
                self.menu_machine.state = IVRState.FEEDBACK
            await self._finalize_and_speak(effective_user, plan.text)
            return

        from providers.chat.ivr_mini_model import (
            AUTONOMOUS_KNOWLEDGE,
            extract_order_last4_digits,
            ivr_mini_brain,
        )

        # ── 1. Fast Telephony Protocol Signals (instant response) ─────────────
        # A. Appreciation / Closing turn (auto-hangup trigger)
        if ivr_mini_brain._is_closing_intent(lower_user):
            self.conv_state.awaiting_order_id_digits = False
            self._order_dtmf_buffer = ""
            reply_text = AUTONOMOUS_KNOWLEDGE["appreciation"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["appreciation"]["en"])
            logger.info(f"[IVR] 📞 Closing farewell turn detected: '{mask_sensitive_text(clean_user)}' -> '{reply_text}'")
            return await self._finalize_and_speak(effective_user, reply_text)

        # B. Voice clarity / Hearing check
        if any(w in lower_user for w in ["hear me", "loud and clear", "awaaz", "awaz", "sound", "आवाज़", "slow down", "dheere", "धीरे", "ଶୁଣା"]):
            reply_text = AUTONOMOUS_KNOWLEDGE["voice_clarification"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["voice_clarification"]["en"])
            logger.info(f"[IVR] 🔊 Voice clarity check: '{mask_sensitive_text(clean_user)}' -> '{reply_text}'")
            return await self._finalize_and_speak(effective_user, reply_text)

        # C. Representative / Live human agent transfer
        if any(w in lower_user for w in ["representative", "agent", "executive", "human", "person", "operator", "talk to", "speak to", "एजेंट", "प्रतिनिधि", "ପ୍ରତିନିଧି", "ଏଜେଣ୍ଟ"]):
            self.conv_state.awaiting_order_id_digits = False
            self._order_dtmf_buffer = ""
            reply_text = AUTONOMOUS_KNOWLEDGE["human_agent"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["human_agent"]["en"])
            logger.info(f"[IVR] 👤 Live human agent requested: '{mask_sensitive_text(clean_user)}' -> '{reply_text}'")
            return await self._finalize_and_speak(effective_user, reply_text)

        # D. Opening greeting on very first turn
        user_turn_count = len([m for m in self._history if m.role == "user"])
        if ivr_mini_brain._is_greeting_intent(lower_user) and user_turn_count <= 1:
            reply_text = AUTONOMOUS_KNOWLEDGE["greeting"].get(self.selected_language, AUTONOMOUS_KNOWLEDGE["greeting"]["en"])
            logger.info(f"[IVR] 👋 Opening greeting turn: '{mask_sensitive_text(clean_user)}' -> '{reply_text}'")
            return await self._finalize_and_speak(effective_user, reply_text)

        # E. Order ID Status Verification (ask last 4 digits -> reply complete status)
        non_order_intents = {
            "video_kyc", "upi_refund", "upi_payment_failed", "card_block", "card_pin",
            "account_balance", "loans_emi", "human_agent", "fraud_escalation", "unrelated_general",
            "complaint", "complaint_status", "api_action",
        }
        inferred_order_check = ivr_mini_brain._infer_intent_with_history(lower_user, self._history)
        if pkg.conversation.intent.intent in non_order_intents:
            self.conv_state.awaiting_order_id_digits = False
            self._order_dtmf_buffer = ""

        matched_order_reply = ""
        matched_order_digits4: Optional[str] = None

        from tools.registry import tool_registry as _tool_registry

        remote_tools_active = any(
            src == "remote_django" for src in _tool_registry.list_tools_with_sources().values()
        )
        ivr_tools_enabled = os.getenv("IVR_ENABLE_TOOLS", "true").lower() in ("1", "true", "yes")
        all_tool_schemas = _tool_registry.get_all_tool_schemas()
        session_tools = (
            _select_ivr_session_tools(
                all_tool_schemas=all_tool_schemas,
                user_text=effective_user,
                history=self._history,
                service_category=self.selected_service,
                remote_tools_active=remote_tools_active,
            )
            if (ivr_tools_enabled and all_tool_schemas)
            else None
        )

        is_complaint_turn = (
            self.selected_service == "complaint"
            or pkg.conversation.intent.intent == "complaint"
            or any(k in lower_user for k in ("complaint", "complain", "ticket", "शिकायत", "ଅଭିଯୋଗ"))
        )

        if (
            not remote_tools_active
            and not is_complaint_turn
            and (
                self.conv_state.awaiting_order_id_digits
                or self.conv_state.intent == "order_status"
                or inferred_order_check == "order_status"
            )
        ):
            digits4 = extract_order_last4_digits(
                effective_user,
                allow_bare_digits=(self.conv_state.awaiting_order_id_digits or self.conv_state.intent == "order_status"),
            )
            if digits4:
                self.conv_state.awaiting_order_id_digits = False
                self._order_dtmf_buffer = ""
                self.conv_state.order_id_last4 = digits4
                matched_order_digits4 = digits4
                matched_order_reply = ivr_mini_brain.lookup_order_status(digits4, lang=self.selected_language)
                logger.info(f"[IVR] 📦 Resolved Order ID ending '{digits4}' -> providing complete status.")
            else:
                self.conv_state.awaiting_order_id_digits = True
                self._order_dtmf_buffer = ""
                ask_digits_reply = AUTONOMOUS_KNOWLEDGE["order_status_ask_digits"].get(
                    self.selected_language,
                    AUTONOMOUS_KNOWLEDGE["order_status_ask_digits"]["en"],
                )
                active_emotion = str(self.conv_state.emotion.get("fused") or "neutral").lower()
                ask_digits_reply = ivr_mini_brain.apply_sentiment_tone(
                    ask_digits_reply,
                    active_emotion,
                    lang=self.selected_language,
                    previous_attempts=self.conv_state.failed_attempts,
                )
                logger.info(f"[IVR] 📦 Prompting caller for last 4 digits of Order ID ({self.selected_language}).")
                return await self._finalize_and_speak(effective_user, ask_digits_reply)

        # ── 2. RAG-First Conversational AI (Hybrid Retrieval + Gemma + Validator) ───
        ivr_tenant = self.tenant_id or os.getenv("IVR_TENANT_ID", "879e9bda-8049-4824-9aa0-fc5e9f0012d9")
        service_info = SERVICE_BY_KEY.get(self.selected_service or "customer_support", SERVICE_CATEGORIES.get("1", {}))
        service_name = service_info.get("name_en", "Customer Support")
        active_emotion = str(self.conv_state.emotion.get("fused") or "neutral").lower()

        enable_gemma = os.getenv("ENABLE_GEMMA_OPEN_DOMAIN", "true").lower() in ("1", "true", "yes")
        if enable_gemma:
            from platform_core.config.settings import settings as _ivr_settings

            ivr_model_alias = _resolve_ivr_tool_model_alias(
                preferred_alias=GEMMA_MODEL_ALIAS,
                has_tools=bool(session_tools),
                settings_alias=_ivr_settings.GEMMA_MODEL_ALIAS,
            )

            ivr_context = TenantContext(
                tenant_id=ivr_tenant,
                tenant_name=service_name or "IVR Voice Tenant",
                api_key_id="ivr_session",
                scopes={"*"},
            )
            gen_res = await self.response_generator.generate_response(
                user_message=effective_user,
                conversation_history=self._history,
                tenant_id=ivr_tenant,
                language=self.selected_language,
                orchestrator=self._orchestrator,
                model_alias=ivr_model_alias,
                service_category=self.selected_service,
                service_name=service_name,
                intelligence_context=plan.intelligence_context,
                tool_output=matched_order_reply,
                required_token=matched_order_digits4,
                active_emotion=active_emotion,
                previous_attempts=self.conv_state.failed_attempts,
                timeout_seconds=(
                    float(os.getenv("IVR_TOOL_TURN_TIMEOUT_SECONDS", str(getattr(_ivr_settings, "IVR_TOOL_TURN_TIMEOUT_SECONDS", 95.0))))
                    if session_tools
                    else 3.5
                ),
                session_factory=async_session,
                context=ivr_context if session_tools else None,
                tools=session_tools,
                authorized_customer_id=getattr(self, "authorized_customer_id", None),
                call_uuid=self.call_uuid,
                caller_number=self.caller_number,
                tools_require_remote=remote_tools_active,
            )
            reply_text = gen_res.text
            self._last_retrieval_tier = gen_res.retrieval_eval.tier.value
            if gen_res.resolved_customer_id and not getattr(self, "authorized_customer_id", None):
                self.authorized_customer_id = gen_res.resolved_customer_id
        else:
            retrieved = await self.response_generator.retrieval_service.search(
                query=effective_user,
                tenant_id=ivr_tenant,
                top_k=4,
                preferred_category=self.selected_service,
                session_factory=async_session,
            )
            self._last_retrieval_tier = retrieved.evaluation.tier.value
            if matched_order_reply:
                reply_text = ivr_mini_brain.apply_sentiment_tone(
                    matched_order_reply,
                    active_emotion,
                    lang=self.selected_language,
                    previous_attempts=self.conv_state.failed_attempts,
                )
            else:
                reply_text = ivr_mini_brain.synthesize(
                    user_query=effective_user,
                    messages=self._history,
                    rag_context="\n\n".join(retrieved.chunk_texts),
                    language=self.selected_language,
                    service_category=self.selected_service,
                    intelligence_context=plan.intelligence_context,
                )

        await self._finalize_and_speak(effective_user, reply_text)

    async def _emit_transfer_event(self, payload: dict[str, Any]) -> None:
        """Emit a structured JSON transfer control event to Asterisk safely."""
        try:
            await self._ws.send_text(json.dumps(payload))
            logger.info(
                f"[IVR] 🔀 Emitted transfer event to PBX (uuid={payload.get('uuid')}, "
                f"reason={payload.get('reason')})."
            )
        except Exception as exc:
            logger.warning(f"[IVR] Failed to emit Asterisk transfer event (non-fatal): {exc}")

    async def _finalize_and_speak(self, user_text: str, reply_text: str) -> None:
        """Sanitizes text, updates history, triggers learning, and streams PCM audio."""
        reply_text = (reply_text or "").strip()
        # Strict Language Guard: If session is Hindi, force Devanagari Hindi text
        if self.selected_language == "hi" and not any('\u0900' <= ch <= '\u097f' for ch in reply_text):
            logger.warning(f"[IVR] Enforcing Hindi language on reply text: '{reply_text}'")
            reply_text = ivr_mini_brain.synthesize(
                user_query=user_text,
                messages=self._history,
                language="hi",
                service_category=self.selected_service,
            )

        if not reply_text or "mock response" in reply_text.lower():
            if self.selected_language == "hi":
                reply_text = "माफ़ कीजिए, आपकी आवाज़ साफ़ नहीं आई। कृपया अपनी बात दोबारा कहें।"
            elif self.selected_language == "or":
                reply_text = "କ୍ଷମା କରିବେ, ଆପଣଙ୍କ କଥା ସ୍ପଷ୍ଟ ଶୁଣାଗଲା ନାହିଁ। ଦୟାକରି ଆପଣଙ୍କ ଅନୁରୋଧ ପୁଣିଥରେ କୁହନ୍ତୁ।"
            else:
                reply_text = "I'm sorry, I couldn't clearly hear that. Could you please repeat your request?"

        from providers.text_to_speech.base import MockTextToSpeechAdapter

        active_emotion = str(self.conv_state.emotion.get("fused") or "neutral").lower()
        active_trend = str(self.conv_state.emotion.get("trend") or "stable").lower()
        sat_val = self.conv_state.satisfaction_score
        sat_str = f"{sat_val:.2f}" if isinstance(sat_val, (int, float)) else "n/a"
        tts_rate, tts_pitch = MockTextToSpeechAdapter.prosody_for_emotion(active_emotion)
        reply_text = reply_text.strip()
        logger.info(
            f"[IVR] 💬 Reply Sentiment: sentiment={active_emotion} trend={active_trend} "
            f"satisfaction={sat_str} tts_rate={tts_rate} tts_pitch={tts_pitch}"
        )
        logger.info(f"[IVR] AI reply (emotion={active_emotion}, trend={active_trend}): '{reply_text}'")

        # Keep rolling history
        self._history.append(ChatMessage(role="assistant", content=reply_text))

        # Trim history to avoid context explosion
        if len(self._history) > 21:
            self._history = self._history[:1] + self._history[-20:]

        # 3. TTS → raw PCM 8 kHz (clean spoken text of markdown/bullets/emojis)
        spoken_reply = _clean_spoken_text(reply_text)
        self._is_synthesizing = True
        try:
            logger.info(
                f"[IVR] Synthesizing speech for reply ({len(spoken_reply)} chars, emotion={active_emotion})..."
            )
            tts_bytes = await self._audio_service.synthesize(
                text=spoken_reply,
                model="tts-1",
                voice="alloy",
                response_format="wav",
                emotion=active_emotion,
            )
            pcm_reply = await asyncio.to_thread(_tts_output_to_pcm8k, tts_bytes)
            logger.info(f"[IVR] TTS ready: {len(pcm_reply)} bytes PCM ({len(pcm_reply)/16000:.1f}s)")
        except Exception as exc:
            logger.error(f"[IVR] TTS error: {exc}")
            return
        finally:
            self._is_synthesizing = False

        # Trigger Level 1 / Level 2 knowledge learning after TTS synthesis so embedding never delays audio
        quality_ok = (self.conv_state.transcript_confidence >= 0.55) if self.conv_state.turn_number > 0 else True
        resolved_flag = (self.conv_state.resolution_status == "resolved") if self.conv_state.turn_number > 0 else None
        if self._last_retrieval_tier == "none" and quality_ok:
            asyncio.create_task(
                self.knowledge_learner.record_unanswered_query(
                    tenant_id=self.tenant_id,
                    call_uuid=self.call_uuid,
                    user_text=user_text,
                    provisional_reply=reply_text,
                )
            )
        else:
            asyncio.create_task(
                learn_from_voice_turn(
                    tenant_id=self.tenant_id,
                    call_uuid=self.call_uuid,
                    user_text=user_text,
                    reply_text=reply_text,
                    quality_ok=quality_ok,
                    resolved=resolved_flag,
                )
            )

        # Stop any active waiting/processing chime before streaming the final spoken reply
        await self._stop_processing_sound()

        # Cancel any prior playback before streaming the new reply so frames never interleave
        if self._playback_task and not self._playback_task.done():
            self._stop_playback(reason="superseded_by_new_reply")
            await asyncio.sleep(0)

        # 4. Send reply over WebSocket as cancellable playback task (supports voice barge-in)
        self._is_playing = True
        self._cancelled_by_barge_in = False
        self._playback_started_ts = time.monotonic()
        self._barge_in_voice_frames = 0
        self._playback_task = asyncio.create_task(self._send_pcm_reply(pcm_reply))
        was_barge_in_interrupted = False
        try:
            await self._playback_task
        except asyncio.CancelledError:
            was_barge_in_interrupted = self._cancelled_by_barge_in
            logger.info(f"[IVR] ⚡ AI reply playback interrupted by caller barge-in (uuid={self.call_uuid}).")
        finally:
            self._is_playing = False
            self._playback_task = None
            # Preserve buffered caller speech frames when cancelled by voice barge-in
            if not self._cancelled_by_barge_in:
                self._pcm_buffer.clear()
                self._barge_in_captured = False
            self._cancelled_by_barge_in = False
            self._last_voice_ts = time.monotonic()
            self._last_playback_end_ts = self._last_voice_ts

        # 5. Auto Cut after Thank You / Closing turn (do NOT auto-cut when awaiting caller reply)
        is_pending_reply = any(
            getattr(self.conv_state, flag, False)
            for flag in (
                "awaiting_resolution_confirmation",
                "awaiting_transfer_confirmation",
                "awaiting_feedback",
                "awaiting_order_id_digits",
            )
        )
        is_clarify_reply = any(
            marker in reply_text.lower()
            for marker in ("trouble hearing", "please repeat", "repeat your question", "दोबारा", "दोहराएं", "ସ୍ପଷ୍ଟ ଶୁଣାଗଲା ନାହିଁ")
        )
        if (
            not was_barge_in_interrupted
            and not is_pending_reply
            and not is_clarify_reply
            and not (self.menu_machine.state == IVRState.FEEDBACK and self.conv_state.awaiting_feedback)
            and is_closing_turn(user_text, reply_text)
        ):
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
        """Stream a spoken AI reply or menu prompt to PBX over WebSocket."""
        await self._stream_pcm_frames(pcm, verbose_logs=True)

    async def _stream_pcm_frames(self, pcm: bytes, verbose_logs: bool = True) -> None:
        """
        Stream raw PCM frames to PBX over WebSocket with jitter-buffer pre-filling
        and high-precision drift compensation to eliminate glitches and network stutter.
        """
        async with self._playback_lock:
            try:
                if verbose_logs:
                    logger.info(f"[IVR] Sending audio reply to Asterisk (uuid={self.call_uuid})...")
                if self.call_uuid:
                    header = json.dumps({"event": "play", "uuid": self.call_uuid})
                    await self._ws.send_text(header)

                chunks = list(chunk_pcm(pcm, PBX_CHUNK_BYTES))
                total_chunks = len(chunks)
                if total_chunks == 0:
                    return

                # PREBUFFERING: Send initial burst of 3 frames (60 ms) immediately to fill
                # Asterisk/MizoDroid jitter buffer smoothly without clipping or long startup bursts.
                PREBUFFER_FRAMES = min(3, total_chunks)
                for i in range(PREBUFFER_FRAMES):
                    await self._ws.send_bytes(chunks[i])

                chunks_sent = PREBUFFER_FRAMES
                start_time = time.monotonic()
                target_time = start_time

                for i in range(PREBUFFER_FRAMES, total_chunks):
                    frame = chunks[i]
                    if self._is_ducking:
                        self._duck_gain = max(BARGE_IN_DUCK_GAIN, self._duck_gain * 0.65)
                        frame_interval = FRAME_INTERVAL_SECONDS * BARGE_IN_DUCK_SLOWDOWN_FACTOR
                    else:
                        if self._duck_gain < 0.999:
                            self._duck_gain = min(1.0, self._duck_gain + 0.15)
                        frame_interval = FRAME_INTERVAL_SECONDS

                    if self._duck_gain < 0.999:
                        frame = _scale_pcm_frame(frame, self._duck_gain)

                    await self._ws.send_bytes(frame)
                    chunks_sent += 1

                    # Calculate target wall-clock time for subsequent frames
                    target_time += frame_interval
                    now = time.monotonic()
                    sleep_duration = target_time - now

                    # If behind by more than 100ms due to a system freeze, resync to avoid a huge burst
                    if sleep_duration < -0.100:
                        target_time = now
                        sleep_duration = 0

                    if sleep_duration > 0.002:
                        await asyncio.sleep(sleep_duration)

                # Comfort trailing silence padding (40ms) to prevent abrupt tail click/cutoff
                silence_frame = b"\x00" * PBX_CHUNK_BYTES
                for _ in range(2):
                    await asyncio.sleep(FRAME_INTERVAL_SECONDS)
                    await self._ws.send_bytes(silence_frame)
                    chunks_sent += 1

                if verbose_logs:
                    logger.info(f"[IVR] Finished sending audio: {chunks_sent} frames sent.")

            except asyncio.CancelledError:
                if verbose_logs:
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

def _scale_pcm_frame(frame: bytes, gain: float) -> bytes:
    """Scale 16-bit signed little-endian PCM frame amplitude cleanly without overflow."""
    if not frame or len(frame) < 2 or gain >= 0.999:
        return frame
    samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) * max(0.0, gain)
    return np.clip(samples, -32768.0, 32767.0).astype("<i2").tobytes()


def is_closing_turn(user_text: str, reply_text: str) -> bool:
    """
    Detects if this conversational turn is a closing/farewell turn
    (e.g. caller saying 'thank you', 'thanks', 'dhanyawad', 'bye', or AI delivering closing greeting),
    unless the AI reply asks a question or requests confirmation/feedback/digits.
    """
    u = (user_text or "").lower().strip()
    r = (reply_text or "").lower().strip()

    # Never end the call when the assistant asks a question or waits for caller confirmation/rating.
    question_or_prompt_markers = (
        "?",
        "confirm whether your issue",
        "anything else i can assist",
        "scale of 1 to 5",
        "how satisfied are you",
        "last 4 digits",
        "trouble hearing",
        "please repeat",
        "क्या आपकी समस्या",
        "पुष्टि कर सकते",
        "ଆପଣଙ୍କ ସମସ୍ୟାର ସମାଧାନ",
    )
    if any(marker in r for marker in question_or_prompt_markers):
        return False

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
    # Remove bullet markers, numbered paren lists, and header hashes without deleting spoken numbers
    text = re.sub(r"^\s*(?:[-*#•]+|\d+\))\s+", "", text, flags=re.MULTILINE)
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

    # Strip quotes while preserving spoken numbers and decimals
    text = text.replace('"', '').replace("'", '').strip()

    # Deduplicate consecutive identical sentences (e.g. "Sentence A. Sentence A." -> "Sentence A.")
    pattern = r'([^.!?।\n]+[.!?।])\s*(?:\1\s*)+'
    text = re.sub(pattern, r'\1 ', text).strip()

    # Phonetic normalization: detect Odia first so shared Devanagari danda punctuation ('।', U+0964)
    # in Odia sentences never triggers Hindi Devanagari replacements.
    has_odia = any('\u0b00' <= ch <= '\u0b7f' for ch in text)
    has_hindi = (not has_odia) and any(
        '\u0900' <= ch <= '\u0963' or '\u0966' <= ch <= '\u097f' for ch in text
    )
    if has_hindi:
        hindi_loanwords = [
            (r"\bvideo\s*kyc\b", "वीडियो केवाईसी"),
            (r"\bkyc\b", "केवाईसी"),
            (r"\bupi\b", "यूपीआई"),
            (r"\botp\b", "ओटीपी"),
            (r"\bpan\s*card\b", "पैन कार्ड"),
            (r"\bpan\b", "पैन"),
            (r"\baadha+r\b", "आधार"),
            (r"\batm\b", "एटीएम"),
            (r"\bpin\b", "पिन"),
            (r"\bsms\b", "एसएमएस"),
            (r"\butr\b", "यूटीआर"),
            (r"\bemi\b", "ईएमआई"),
            (r"\bnet\s*banking\b", "नेटबैंकिंग"),
            (r"\bdebit\s*card\b", "डेबिट कार्ड"),
            (r"\bcredit\s*card\b", "क्रेडिट कार्ड"),
            (r"\bcustomer\s*support\b", "ग्राहक सेवा"),
            (r"\bdocuments?\b", "डॉक्यूमेंट्स"),
            (r"\boriginal\b", "ओरिजिनल"),
            (r"\bphysical\b", "फिजिकल"),
            (r"\bverification\b", "वेरिफिकेशन"),
            (r"\bverify\b", "वेरिफाई"),
            (r"\bmobile\b", "मोबाइल"),
            (r"\bnumber\b", "नंबर"),
            (r"\baccount\b", "अकाउंट"),
            (r"\bbalance\b", "बैलेंस"),
            (r"\bstatus\b", "स्टेटस"),
            (r"\bbank\b", "बैंक"),
            (r"\bfailed\b", "फेल्ड"),
            (r"\btransfer\b", "ट्रांसफर"),
            (r"\b(?:agent|representative)\b", "प्रतिनिधि"),
            (r"\bapp\b", "ऐप"),
            (r"\bpassword\b", "पासवर्ड"),
            (r"\brefund\b", "रिफंड"),
            (r"\bpayment\b", "पेमेंट"),
            (r"\border\s*id\b", "ऑर्डर आईडी"),
            (r"\border\b", "ऑर्डर"),
            (r"\bdelivery\b", "डिलीवरी"),
            (r"\bcourier\b", "कूरियर"),
            (r"\btracking\b", "ट्रैकिंग"),
        ]
        for pat, rep in hindi_loanwords:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # Phonetic normalization for Odia speech so native Odia VITS receives pure Odia script
    if has_odia:
        odia_loanwords = [
            (r"\bvideo\s*kyc\b", "ଭିଡିଓ କେୱାଇସି"),
            (r"\bkyc\b", "କେୱାଇସି"),
            (r"\bupi\b", "ୟୁପିଆଇ"),
            (r"\botp\b", "ଓଟିପି"),
            (r"\bpan\s*card\b", "ପ୍ୟାନ୍ କାର୍ଡ"),
            (r"\bpan\b", "ପ୍ୟାନ୍"),
            (r"\baadha+r\b", "ଆଧାର"),
            (r"\batm\b", "ଏଟିଏମ୍"),
            (r"\bpin\b", "ପିନ୍"),
            (r"\bsms\b", "ଏସ୍ଏମ୍ଏସ୍"),
            (r"\butr\b", "ୟୁଟିଆର୍"),
            (r"\bemi\b", "ଇଏମ୍ଆଇ"),
            (r"\bnet\s*banking\b", "ନେଟ୍ ବ୍ୟାଙ୍କିଂ"),
            (r"\bdebit\s*card\b", "ଡେବିଟ୍ କାର୍ଡ"),
            (r"\bcredit\s*card\b", "କ୍ରେଡିଟ୍ କାର୍ଡ"),
            (r"\bcustomer\s*support\b", "ଗ୍ରାହକ ସେବା"),
            (r"\bdocuments?\b", "ଡକ୍ୟୁମେଣ୍ଟ୍"),
            (r"\boriginal\b", "ଅରିଜିନାଲ୍"),
            (r"\bphysical\b", "ଫିଜିକାଲ୍"),
            (r"\bverification\b", "ଭେରିଫିକେସନ୍"),
            (r"\bverify\b", "ଭେରିଫାଇ"),
            (r"\bmobile\b", "ମୋବାଇଲ୍"),
            (r"\bnumber\b", "ନମ୍ବର"),
            (r"\baccount\b", "ଆକାଉଣ୍ଟ"),
            (r"\bbalance\b", "ବାଲାନ୍ସ"),
            (r"\bstatus\b", "ଷ୍ଟାଟସ୍"),
            (r"\bbank\b", "ବ୍ୟାଙ୍କ"),
            (r"\bfailed\b", "ଫେଲ୍ଡ"),
            (r"\btransfer\b", "ଟ୍ରାନ୍ସଫର୍"),
            (r"\b(?:agent|representative)\b", "ପ୍ରତିନିଧି"),
            (r"\bapp\b", "ଆପ୍"),
            (r"\bpassword\b", "ପାସୱାର୍ଡ"),
            (r"\brefund\b", "ରିଫଣ୍ଡ"),
            (r"\bpayment\b", "ପେମେଣ୍ଟ"),
            (r"\border\s*id\b", "ଅର୍ଡର ଆଇଡି"),
            (r"\border\b", "ଅର୍ଡର"),
            (r"\bdelivery\b", "ଡେଲିଭରି"),
            (r"\bcourier\b", "କୁରିଅର"),
            (r"\btracking\b", "ଟ୍ରାକିଂ"),
        ]
        for pat, rep in odia_loanwords:
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
