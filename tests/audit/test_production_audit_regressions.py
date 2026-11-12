"""
Automated Regression & Audit Reproducer Suite for Hybrid Voice Intelligence.

Covers:
  1. Verified production guarantees (Conversation Intelligence separation of
     emotion vs certainty vs transcript confidence, explicit agent transfer priority,
     tenant isolation, DB failure resilience, STT/LLM timeout recovery).
  2. Strict `xfail` audit reproducers for unpatched defects discovered during the
     Production Readiness & Reliability Audit (held unpatched pending user review).
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from decision_engine.decision_manager import Action, DecisionManager
from platform_core.contracts.requests import (
    ChatMessage,
    NormalizedChatResponse,
    NormalizedChoice,
    UsageInfo,
)
from providers.chat.ivr_mini_model import extract_order_last4_digits, ivr_mini_brain
from providers.speech_to_text.whisper_local import WhisperLocalSpeechToTextAdapter
from providers.text_to_speech.base import MockTextToSpeechAdapter
from telephony.intelligence.pipeline import TelephonyIntelligencePipeline
from telephony.intelligence.session_state import ConversationState
from telephony.ivr_audio_cache import get_cached_prompt_pcm
from telephony.ivr_menu import IVRState
from telephony.ivr_prompts import HUMAN_TRANSFER
from telephony.ivr_session import IVRCallSession, _clean_spoken_text


def _make_session(reply_text: str = "Your UPI refund is in progress."):
    ws = MagicMock()
    ws.send_text = AsyncMock()
    ws.send_bytes = AsyncMock()

    orchestrator = MagicMock()
    orchestrator.execute = AsyncMock(
        return_value=NormalizedChatResponse(
            id="audit-resp",
            model="ivr-rag-mini",
            created=0,
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
    )

    audio_svc = MagicMock()
    audio_svc.synthesize = AsyncMock(return_value=b"")
    session = IVRCallSession(ws=ws, orchestrator=orchestrator, audio_service=audio_svc)
    session.intel_pipeline.persistence_enabled = False
    session.on_call_start({"uuid": "audit-call-001", "caller_id": "9999999999"})
    session.menu_machine.state = IVRState.AI_CONVERSATION
    session.selected_language = "en"
    session._language = "en"
    return session, ws, orchestrator, audio_svc


# ── Verified Production Guarantees (Passing) ───────────────────────────────────


def test_emotion_alone_does_not_escalate_clear_billing_turn():
    """Negative emotion must act as an uncertain indicator, not auto-escalate alone."""
    dm = DecisionManager()
    state = ConversationState(call_uuid="c1", tenant_id="t1", language="en")
    state.turn_number = 1
    state.intent = "upi_refund"
    state.emotion = {"text": "frustrated", "voice": "neutral", "fused": "frustrated", "trend": "stable", "confidence": 0.6}
    state.audio_quality_score = 0.88
    state.transcript_confidence = 0.85
    state.caller_certainty = 0.80
    state.satisfaction_score = 0.45

    decision = dm.decide(state)
    assert decision.action == Action.CONTINUE


def test_low_transcript_confidence_requests_clarification():
    """Unclear / low-confidence audio or transcript must clarify rather than guess."""
    dm = DecisionManager()
    state = ConversationState(call_uuid="c2", tenant_id="t1", language="en")
    state.turn_number = 1
    state.intent = "unknown"
    state.audio_quality_score = 0.25
    state.transcript_confidence = 0.40

    decision = dm.decide(state, needs_clarification=True)
    assert decision.action == Action.CLARIFY


def test_explicit_agent_request_overrides_low_confidence():
    """Explicit request for human representative must immediately trigger TRANSFER."""
    dm = DecisionManager()
    state = ConversationState(call_uuid="c3", tenant_id="t1", language="en")
    state.turn_number = 1
    state.intent = "human_agent_request"
    state.explicit_agent_request = True
    state.audio_quality_score = 0.25
    state.transcript_confidence = 0.30

    decision = dm.decide(state, needs_clarification=True)
    assert decision.action == Action.TRANSFER
    assert decision.metadata and decision.metadata["transfer_payload"]["event"] == "transfer"


def test_conversation_state_isolates_tenant_contexts():
    """Concurrent calls sharing a call_uuid across tenants keep distinct tenant_ids."""
    s_alpha = ConversationState(call_uuid="shared-uuid-1", tenant_id="tenant-alpha", language="en")
    s_beta = ConversationState(call_uuid="shared-uuid-1", tenant_id="tenant-beta", language="hi")
    s_alpha.intent = "upi_payment"
    s_beta.intent = "card_block"

    assert s_alpha.tenant_id == "tenant-alpha"
    assert s_beta.tenant_id == "tenant-beta"
    assert s_alpha.to_intelligence_context()["intent"] != s_beta.to_intelligence_context()["intent"]


@pytest.mark.asyncio
async def test_ivr_continues_operating_when_postgres_raises_operational_error():
    """DB outages must not abort active telephone voice replies."""
    session, ws, _, audio_svc = _make_session()
    audio_svc.transcribe = AsyncMock(
        return_value={"text": "Where is my UPI refund", "diagnostics": {"inference_ms": 5.0}}
    )
    speech_pcm = get_cached_prompt_pcm(HUMAN_TRANSFER["en"])

    def _broken_db(*a, **kw):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    with patch("telephony.ivr_session.async_session", side_effect=_broken_db), patch(
        "telephony.intelligence.event_handler.async_session", side_effect=_broken_db
    ), patch("telephony.ivr_session.learn_from_voice_turn", AsyncMock()):
        await session._run_stt_ai_tts(speech_pcm)

    assert session.conv_state.turn_number == 1
    assert audio_svc.synthesize.await_count == 1
    sent_events = [json.loads(c.args[0]) for c in ws.send_text.await_args_list]
    assert not any(e.get("event") == "hangup" or e.get("command") == "hangup" for e in sent_events)


@pytest.mark.asyncio
async def test_ivr_recovers_from_stt_and_orchestrator_timeouts():
    """STT timeout skips cleanly; Gemma timeout falls back to IVR mini-brain."""
    session, _, orchestrator, audio_svc = _make_session()
    speech_pcm = get_cached_prompt_pcm(HUMAN_TRANSFER["en"])

    # STT timeout -> non-fatal
    audio_svc.transcribe = AsyncMock(side_effect=asyncio.TimeoutError("STT timeout"))
    await session._run_stt_ai_tts(speech_pcm)
    assert session.conv_state.turn_number == 0

    # Orchestrator timeout -> falls back cleanly to ivr_mini_brain
    audio_svc.transcribe = AsyncMock(
        return_value={"text": "Tell me about Video KYC documents", "diagnostics": {"inference_ms": 5.0}}
    )
    orchestrator.execute = AsyncMock(side_effect=asyncio.TimeoutError("Gemma timeout"))
    with patch("telephony.ivr_session.async_session", side_effect=RuntimeError("skip pg")), patch(
        "telephony.ivr_session.learn_from_voice_turn", AsyncMock()
    ):
        await session._run_stt_ai_tts(speech_pcm)
    assert session.conv_state.turn_number == 1
    assert audio_svc.synthesize.await_count == 1


def test_clean_spoken_text_preserves_numbers_and_normalizes_banking_loanwords():
    """Spoken digits/amounts must stay intact while Hindi/Odia banking terms are localized."""
    assert _clean_spoken_text("Please press 1. Refund takes 24 hours.") == "Please press 1. Refund takes 24 hours."
    assert _clean_spoken_text("5000 rupees debited.") == "5000 rupees debited."
    assert _clean_spoken_text("1) पहले विकल्प चुनें।") == "पहले विकल्प चुनें।"

    hi_out = _clean_spoken_text("वीडियो केवाईसी के लिए PAN card और Aadhaar documents तैयार रखें।")
    assert "पैन कार्ड" in hi_out
    assert "आधार" in hi_out
    assert "डॉक्यूमेंट्स" in hi_out
    assert "PAN" not in hi_out and "documents" not in hi_out

    or_out = _clean_spoken_text("ଭିଡିଓ କେୱାଇସି ପାଇଁ original PAN card documents ଆବଶ୍ୟକ।")
    assert "ଅରିଜିନାଲ୍" in or_out
    assert "ପ୍ୟାନ୍ କାର୍ଡ" in or_out
    assert "ଡକ୍ୟୁମେଣ୍ଟ୍" in or_out
    assert "original" not in or_out and "PAN" not in or_out and "documents" not in or_out


@pytest.mark.asyncio
async def test_hindi_empty_stt_with_real_speech_prompts_clarification_instead_of_silence():
    """When caller spoke clearly (>350ms) and STT returned blank, IVR must reply politely."""
    session, _, _, audio_svc = _make_session()
    session.selected_language = "hi"
    session._language = "hi"
    speech_pcm = get_cached_prompt_pcm(HUMAN_TRANSFER["en"])

    audio_svc.transcribe = AsyncMock(return_value={"text": "", "diagnostics": {"inference_ms": 5.0}})
    await session._run_stt_ai_tts(speech_pcm)

    assert audio_svc.synthesize.await_count == 1
    spoken = audio_svc.synthesize.await_args.kwargs["text"]
    assert "माफ़ कीजिए" in spoken


@pytest.mark.asyncio
async def test_barge_in_ignores_cooldown_clicks_and_requires_sustained_speech():
    """Barge-in must ignore playback-start echo cooldown and short clicks (<6 consecutive frames)."""
    session, _, _, _ = _make_session()
    loud_frame = struct.pack("<160h", *([3000] * 160))
    quiet_frame = b"\x00" * 320

    session._is_playing = True
    session._playback_started_ts = time.monotonic()
    session._barge_in_voice_frames = 0
    session._playback_task = asyncio.create_task(asyncio.sleep(5))

    try:
        # 1. Loud frames during initial 0.35s cooldown must NOT trigger barge-in
        for _ in range(8):
            await session.on_pcm_frame(loud_frame)
        assert session._is_playing is True

        # 2. Past cooldown: 3 loud frames + 1 quiet frame resets streak -> no barge-in
        session._playback_started_ts = time.monotonic() - 1.0
        for _ in range(3):
            await session.on_pcm_frame(loud_frame)
        await session.on_pcm_frame(quiet_frame)
        assert session._is_playing is True
        assert session._barge_in_voice_frames == 0

        # 3. Past cooldown: 6 consecutive loud frames must cancel playback
        for _ in range(6):
            await session.on_pcm_frame(loud_frame)
        assert session._is_playing is False
    finally:
        if session._playback_task and not session._playback_task.done():
            session._playback_task.cancel()


def test_extract_order_last4_digits_handles_voice_and_indic_formats():
    assert extract_order_last4_digits("My order ORD-994821 status") == "4821"
    assert extract_order_last4_digits("four eight two one", allow_bare_digits=True) == "4821"
    assert extract_order_last4_digits("चार आठ दो एक", allow_bare_digits=True) == "4821"
    assert extract_order_last4_digits("४८२१", allow_bare_digits=True) == "4821"
    assert extract_order_last4_digits("5000 rupees", allow_bare_digits=False) is None


@pytest.mark.asyncio
async def test_ivr_order_id_two_turn_status_flow_hindi_and_english():
    """Asking about order status asks for last 4 digits, then replies back complete status."""
    # English voice two-turn flow
    session_en, _, orchestrator_en, audio_en = _make_session()
    orchestrator_en.execute = AsyncMock(side_effect=RuntimeError("offline fallback"))
    await session_en._run_ai_and_reply("What is my order id status?")
    first_reply = audio_en.synthesize.await_args.kwargs["text"]
    assert session_en.conv_state.awaiting_order_id_digits is True
    assert "last 4 digits" in first_reply.lower()

    await session_en._run_ai_and_reply("4821")
    second_reply = audio_en.synthesize.await_args.kwargs["text"]
    assert session_en.conv_state.awaiting_order_id_digits is False
    assert session_en.conv_state.order_id_last4 == "4821"
    assert "4821" in second_reply and "Out for Delivery" in second_reply and "BlueDart" in second_reply

    # Hindi voice two-turn flow with spoken Hindi digit words
    session_hi, _, orchestrator_hi, audio_hi = _make_session()
    session_hi.selected_language = "hi"
    session_hi._language = "hi"
    orchestrator_hi.execute = AsyncMock(side_effect=RuntimeError("offline fallback"))
    await session_hi._run_ai_and_reply("मेरा ऑर्डर आईडी स्टेटस बताओ")
    hi_first = audio_hi.synthesize.await_args.kwargs["text"]
    assert session_hi.conv_state.awaiting_order_id_digits is True
    assert "अंतिम 4 अंक" in hi_first

    await session_hi._run_ai_and_reply("चार आठ दो एक")
    hi_second = audio_hi.synthesize.await_args.kwargs["text"]
    assert session_hi.conv_state.awaiting_order_id_digits is False
    assert session_hi.conv_state.order_id_last4 == "4821"
    assert "4821" in hi_second and "ब्लूडार्ट" in hi_second


@pytest.mark.asyncio
async def test_ivr_order_id_dtmf_4_digits_and_single_turn_direct_lookup():
    """Supports both 4-digit DTMF entry while awaiting digits and direct 4-digit voice query."""
    session, _, orchestrator, audio_svc = _make_session()
    orchestrator.execute = AsyncMock(side_effect=RuntimeError("offline fallback"))

    await session._run_ai_and_reply("Track my order")
    assert session.conv_state.awaiting_order_id_digits is True

    for d in ("7", "3", "9", "2"):
        await session.on_dtmf(d)
    for _ in range(40):
        if audio_svc.synthesize.await_count >= 2:
            break
        await asyncio.sleep(0.025)

    assert session.conv_state.order_id_last4 == "7392"
    dtmf_reply = audio_svc.synthesize.await_args.kwargs["text"]
    assert "7392" in dtmf_reply and "In Transit" in dtmf_reply and "Delhivery" in dtmf_reply

    # Direct single-turn query with digits already provided
    await session._run_ai_and_reply("Check order status ending 1024")
    direct_reply = audio_svc.synthesize.await_args.kwargs["text"]
    assert session.conv_state.order_id_last4 == "1024"
    assert "1024" in direct_reply and "Packed" in direct_reply


def test_confident_recognition_with_soft_telephony_audio_does_not_false_clarify():
    """Soft telephony RMS (~0.34 quality) must not force Action.CLARIFY when STT decoded cleanly."""
    dm = DecisionManager()
    state = ConversationState(
        call_uuid="c-soft",
        tenant_id="t1",
        language="en",
        turn_number=4,
        intent="appreciation",
        audio_quality_score=0.34,
        transcript_confidence=0.81,
        system_confidence=0.81,
    )
    decision = dm.decide(state, needs_clarification=False)
    assert decision.action != Action.CLARIFY


@pytest.mark.asyncio
async def test_order_slot_fill_then_thank_you_reports_satisfied_not_frustrated():
    """2-turn Order ID slot fill + 'Thank you very much.' must NOT count as failure or frustration."""
    pipeline = TelephonyIntelligencePipeline(persistence_enabled=False)
    state = ConversationState(call_uuid="c-sent", tenant_id="t1", language="en")

    # Turn 1: ask order status -> IVR enters awaiting_order_id_digits
    await pipeline.process_and_plan(stt_result={"text": "What is my order status?"}, state=state)
    state.awaiting_order_id_digits = True

    # Turn 2: supply 4 digits (slot fill for order_status) -> must not increment failed_attempts
    await pipeline.process_and_plan(stt_result={"text": "4821"}, state=state)
    state.awaiting_order_id_digits = False
    state.resolution_status = "resolved"
    assert state.failed_attempts == 0

    # Turn 3: appreciation
    pkg3, _ = await pipeline.process_and_plan(stt_result={"text": "Thank you very much."}, state=state)
    assert pkg3.conversation.text_emotion.primary_emotion == "satisfied"
    assert pkg3.conversation.fused_emotion == "satisfied"
    assert state.emotion["fused"] == "satisfied"


@pytest.mark.asyncio
async def test_sentiment_expressed_in_reply_text_and_tts_prosody(caplog):
    """Replies include empathetic/reassuring/warm wording, forward caller emotion to TTS, and log sentiment."""
    import logging
    caplog.set_level(logging.INFO)
    # 1. Text sentiment tone helper
    frust_reply = ivr_mini_brain.apply_sentiment_tone("Order ID 4821 is Out for Delivery.", "frustrated", lang="en")
    concern_reply = ivr_mini_brain.apply_sentiment_tone("Order ID 4821 is Out for Delivery.", "concerned", lang="en")
    sat_reply = ivr_mini_brain.apply_sentiment_tone("Order ID 4821 is Out for Delivery.", "satisfied", lang="en")
    assert "understand your frustration" in frust_reply.lower()
    assert "don't worry" in concern_reply.lower()
    assert "glad to help" in sat_reply.lower()

    # No double-prefixing if reply already opens warmly
    assert ivr_mini_brain.apply_sentiment_tone("You're most welcome! Have a great day.", "satisfied", lang="en") == "You're most welcome! Have a great day."

    # 2. IVRCallSession forwards emotion to audio_svc.synthesize and shapes offline order reply
    session, _, orchestrator, audio_svc = _make_session()
    orchestrator.execute = AsyncMock(side_effect=RuntimeError("offline fallback"))
    await session._run_ai_and_reply("I am so frustrated! Where is my order 4821?")
    call_kwargs = audio_svc.synthesize.await_args.kwargs
    assert call_kwargs["emotion"] in ("frustrated", "angry")
    assert "frustration" in call_kwargs["text"].lower() or "understand" in call_kwargs["text"].lower()
    assert "4821" in call_kwargs["text"]
    assert "Caller Sentiment:" in caplog.text
    assert "Reply Sentiment:" in caplog.text


@pytest.mark.asyncio
async def test_tts_adapter_modulates_edge_tts_rate_and_pitch_by_emotion():
    """MockTextToSpeechAdapter maps caller emotions to distinct Edge-TTS rate and pitch settings."""
    adapter = MockTextToSpeechAdapter("mock-tts", {})
    assert adapter.prosody_for_emotion("frustrated") != adapter.prosody_for_emotion("happy")
    assert adapter.prosody_for_emotion("concerned") != adapter.prosody_for_emotion("neutral")

    with patch.object(adapter, "_synthesize_edge_tts", new=AsyncMock(return_value=b"RIFF" + b"\x00" * 200)) as mock_edge:
        await adapter.synthesize("Checking sentiment calibration text alpha", emotion="frustrated")
        assert mock_edge.await_args.kwargs["rate"] == "-9%"
        assert mock_edge.await_args.kwargs["pitch"] == "-2Hz"

        await adapter.synthesize("Checking sentiment calibration text beta", emotion="happy")
        assert mock_edge.await_args.kwargs["rate"] == "-1%"
        assert mock_edge.await_args.kwargs["pitch"] == "+3Hz"


@pytest.mark.asyncio
async def test_new_speech_during_stt_does_not_cancel_inflight_turn():
    """Incoming PCM speech while Whisper is transcribing must not cancel _silence_task mid-turn."""
    session, _, _, audio_svc = _make_session("For Video KYC, keep PAN card ready.")
    session.menu_machine.state = IVRState.AI_CONVERSATION

    # Simulate slow Whisper STT while caller breathes/speaks again
    stt_started = asyncio.Event()

    async def slow_stt(*args, **kwargs):
        stt_started.set()
        await asyncio.sleep(0.08)
        return {"text": "Video KYC documents.", "language": "en"}

    audio_svc.transcribe = AsyncMock(side_effect=slow_stt)

    # Feed >= 0.4s 400Hz speech-band PCM waveform into buffer and trigger _delayed_flush
    import math
    voice_frame = struct.pack(
        "<" + "h" * 160,
        *[int(4000 * math.sin(2.0 * math.pi * 400.0 * i / 8000.0)) for i in range(160)],
    )
    for _ in range(20):
        await session.on_pcm_frame(voice_frame)

    flush_trigger = asyncio.create_task(session._delayed_flush(0.01))
    session._silence_task = flush_trigger
    await asyncio.wait_for(stt_started.wait(), timeout=2.0)

    # While Whisper is running, send new active speech frame (previously cancelled _silence_task mid-STT!)
    await session.on_pcm_frame(voice_frame)
    await asyncio.gather(flush_trigger, return_exceptions=True)
    if session._flush_task:
        await asyncio.wait_for(session._flush_task, timeout=5.0)

    assert audio_svc.synthesize.await_count == 1
    assert "PAN card" in audio_svc.synthesize.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_voice_learning_updates_chunks_and_accelerates_faq_rag():
    """Repeated/popular call queries update chunk ask_count + variants and hit FAQ cache in <1ms."""
    from types import SimpleNamespace
    from storage.repositories.document_repo import DocumentRepository
    from telephony.voice_learning import (
        cache_rag_chunks,
        get_fast_rag_chunks,
        rank_chunks_with_faq_boost,
    )

    # 1. Tenant-isolated fast FAQ cache returns chunks instantly on repeat ask
    cache_rag_chunks("tenant-fast-1", "What documents for Video KYC?", ["Bring PAN card and Aadhaar phone."])
    assert get_fast_rag_chunks("tenant-fast-1", "what documents for video kyc") == [
        "Bring PAN card and Aadhaar phone."
    ]
    assert get_fast_rag_chunks("tenant-fast-2", "What documents for Video KYC?") is None

    # 2. Frequently asked chunks receive priority ranking boost over near-identical cold chunks
    cold_chunk = SimpleNamespace(content="Cold KYC chunk", meta_json=json.dumps({"ask_count": 1}))
    popular_chunk = SimpleNamespace(content="Popular Video KYC FAQ chunk", meta_json=json.dumps({"ask_count": 20}))
    ranked = rank_chunks_with_faq_boost([(cold_chunk, 0.72), (popular_chunk, 0.70)], threshold=0.55, top_k=2)
    assert ranked[0] == "Popular Video KYC FAQ chunk"

    # 3. DocumentRepository.record_chunk_hit increments ask_count, merges variants, and refreshes content
    mock_session = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    repo = DocumentRepository(mock_session)
    chunk = SimpleNamespace(
        content="Old content",
        meta_json=json.dumps({"type": "voice_interaction_turn", "ask_count": 2, "variants": ["video kyc docs"]}),
        embedding=None,
        embedding_768=None,
        embedding_json=None,
    )
    updated = await repo.record_chunk_hit(
        chunk,
        variant_query="documents needed for video kyc",
        updated_content="Updated FAQ content",
        updated_embedding=[0.1] * 384,
    )
    meta = json.loads(updated.meta_json)
    assert meta["ask_count"] == 3
    assert "documents needed for video kyc" in meta["variants"]
    assert updated.content == "Updated FAQ content"
    assert len(updated.embedding) == 384


# ── Audit Defect Reproducers (Strict XFAIL Until User Approves Fixes) ──────────


@pytest.mark.asyncio
async def test_thank_you_confirmation_prompt_must_not_prematurely_hangup():
    """When DecisionManager asks Action.CONFIRM after 'thank you', IVR must wait rather than hanging up."""
    session, ws, _, _ = _make_session()
    session.conv_state.resolution_status = "unresolved"
    await session._run_ai_and_reply("thank you")

    sent_events = [json.loads(call.args[0]) for call in ws.send_text.await_args_list]
    hangup_events = [e for e in sent_events if e.get("event") == "hangup" or e.get("command") == "hangup"]
    assert session.conv_state.awaiting_resolution_confirmation is True
    assert hangup_events == [], f"Unexpected hangup emitted before caller confirmed resolution: {sent_events}"


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="Audit CRITICAL-1b: Keypad 9 HUMAN_TRANSFER only plays notice and omits WebSocket transfer event",
)
async def test_dtmf_human_transfer_must_emit_asterisk_transfer_event():
    """Pressing 9 during service selection must emit structured {'event': 'transfer'} to Asterisk."""
    session, ws, _, _ = _make_session()
    session.menu_machine.state = IVRState.SERVICE_SELECTION
    await session.on_dtmf("9")
    await asyncio.sleep(0.05)

    sent_events = [json.loads(call.args[0]) for call in ws.send_text.await_args_list]
    transfer_events = [e for e in sent_events if e.get("event") == "transfer"]
    assert len(transfer_events) >= 1


@pytest.mark.xfail(
    strict=True,
    reason="Audit CRITICAL-3: WhisperLocalSpeechToTextAdapter forces language='hi' on Odia ('or') audio",
)
def test_whisper_odia_does_not_force_hindi_decoder_language():
    """Odia must not be decoded using Whisper's Hindi decoder without an Odia-capable ASR backend."""
    adapter = WhisperLocalSpeechToTextAdapter("audit-whisper", {"model": "openai/whisper-base"})
    with patch.object(adapter, "_get_pipeline", return_value=lambda arr, generate_kwargs=None: {"text": "test", "_kw": generate_kwargs}):
        import inspect
        src = inspect.getsource(WhisperLocalSpeechToTextAdapter._transcribe_sync)
    assert 'gen_kwargs["language"] = "hi"' not in src


@pytest.mark.xfail(
    strict=True,
    reason="Audit MEDIUM-1: whisper_local diagnostics omit avg_logprob/no_speech_prob, forcing fallback prior",
)
def test_whisper_local_diagnostics_include_logprob_and_no_speech_prob():
    """Whisper diagnostics should include avg_logprob and no_speech_prob for calibrated confidence."""
    import inspect
    src = inspect.getsource(WhisperLocalSpeechToTextAdapter._transcribe_sync)
    assert "avg_logprob" in src and "no_speech_prob" in src


@pytest.mark.asyncio
async def test_voice_barge_in_during_feedback_preserves_short_rating_speech():
    """
    When caller barges in with a short reply (e.g. '5' / 'five') while the 1-5 feedback prompt plays,
    _finalize_and_speak must NOT clear _pcm_buffer on CancelledError, and all 6 onset frames must remain buffered.
    """
    session, _, _, audio_svc = _make_session()
    session.menu_machine.state = IVRState.FEEDBACK
    session.conv_state.awaiting_feedback = True
    dummy_pcm = get_cached_prompt_pcm(HUMAN_TRANSFER["en"])

    async def _slow_send(_pcm):
        await asyncio.sleep(5)

    session._send_pcm_reply = _slow_send
    with patch("telephony.ivr_session._tts_output_to_pcm8k", return_value=dummy_pcm):
        audio_svc.synthesize = AsyncMock(return_value=b"wav-bytes")
        speak_task = asyncio.create_task(
            session._finalize_and_speak("yes", "On a scale of 1 to 5, how satisfied are you with today's support?")
        )
        await asyncio.sleep(0.05)
        session._playback_started_ts = time.monotonic() - 1.0  # past barge-in cooldown

        loud_frame = struct.pack("<" + "h" * 160, *([3000, -3000] * 80))
        for _ in range(6):
            await session.on_pcm_frame(loud_frame)

        await speak_task

    assert len(session._pcm_buffer) == len(loud_frame) * 6
    assert session._min_speech_bytes() <= len(session._pcm_buffer)


@pytest.mark.asyncio
async def test_feedback_accepts_spoken_number_words_and_dtmf_digits():
    """Both spoken rating words ('five'/'पांच') and DTMF digits ('1'-'5') record feedback during IVRState.FEEDBACK."""
    from conversation_intelligence.intent_analyzer import IntentAnalyzer

    analyzer = IntentAnalyzer()
    assert analyzer.analyze("five", awaiting_feedback=True).rating_value == 5
    assert analyzer.analyze("पांच", awaiting_feedback=True).rating_value == 5
    assert analyzer.analyze("four out of 5", awaiting_feedback=True).rating_value == 4

    session, _, _, _ = _make_session()
    session.menu_machine.state = IVRState.FEEDBACK
    session.conv_state.resolution_status = "resolved"
    session.conv_state.customer_confirmed = True
    session.conv_state.feedback_requested = True
    session.conv_state.awaiting_feedback = True

    with patch("telephony.ivr_session.learn_from_voice_turn", AsyncMock()):
        await session.on_dtmf("5")
        await asyncio.sleep(0.1)

    assert session.conv_state.feedback_received is True
    assert session.conv_state.explicit_rating == 5


@pytest.mark.asyncio
async def test_hindi_whisper_hallucination_rejected_and_cross_lingual_rag_works():
    """
    Verify three Hindi telephony safeguards:
      1. Sub-160ms noise/syllable bursts are skipped before Whisper STT.
      2. Improbable high-WPS Devanagari hallucinations from short clips are filtered in whisper_local.
      3. Hindi Devanagari queries augment English RAG anchors and resolve cleanly in ivr_mini_brain.
    """
    from io import BytesIO
    import wave
    from services.knowledge.retrieval_service import HybridRetrievalService

    # 1. Whisper rejects impossible 14-word Devanagari hallucination from a 0.3s clip
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(struct.pack("<" + "h" * 2400, *([1800, -1800] * 1200)))  # 0.3s loud tone
    adapter = WhisperLocalSpeechToTextAdapter("whisper-test", {"model": "openai/whisper-base"})
    gibberish = "अपन कुल पन पर पहन ज़ाँ वान दुझे लिवेदा लाए लगार लेगे नहीं लीवी नपराचाव लोगविदे"
    with patch.object(adapter, "_get_pipeline", return_value=lambda raw, generate_kwargs=None, **kw: {"text": gibberish}):
        res = adapter._transcribe_sync(buf.getvalue(), language="hi")
    assert res["diagnostics"]["hallucination_filtered"] is True

    # 2. Cross-lingual augmentation bridges Devanagari Hindi into English pgvector knowledge
    augmented = HybridRetrievalService._augment_cross_lingual_query("मेरा यूपीआई पेमेंट अटक गया है")
    assert "upi payment failed" in augmented.lower()

    # 3. ivr_mini_brain handles everyday Hindi banking queries accurately in Devanagari
    reply = ivr_mini_brain.synthesize("मेरा यूपीआई पेमेंट अटक गया है", language="hi")
    assert "यूपीआई" in reply
    assert any("\u0900" <= ch <= "\u097f" for ch in reply)


@pytest.mark.asyncio
async def test_waiting_processing_sound_plays_on_slow_turn_and_stops_cleanly():
    """
    Slow STT/AI/TTS turns emit the gentle waiting/processing pulse chime, stop cleanly
    before streaming the spoken reply, and fast turns skip extra waiting chimes.
    """
    import math
    import numpy as np
    from telephony import ivr_session as ivr_mod
    from telephony.pcm_utils import generate_processing_pulse_pcm

    # 1. Verify gentle dual-tone waiting PCM amplitude and format
    pulse = generate_processing_pulse_pcm()
    assert len(pulse) >= 2400
    samples = np.frombuffer(pulse, dtype="<i2")
    assert 500 < int(np.max(np.abs(samples))) <= 2600

    # 2. Slow turn -> emits waiting chime play event before final reply play event
    session, ws, _, audio_svc = _make_session()
    await asyncio.sleep(0.02)
    session._stop_playback(reason="test_setup")
    await asyncio.sleep(0.02)
    ws.send_text.reset_mock()
    ws.send_bytes.reset_mock()

    orig_synth = audio_svc.synthesize

    async def _slow_synth(*args, **kwargs):
        await asyncio.sleep(0.30)
        return await orig_synth(*args, **kwargs)

    audio_svc.synthesize = AsyncMock(side_effect=_slow_synth)

    pcm_voice = struct.pack(
        "<" + "h" * 2400,
        *[int(4000 * math.sin(2.0 * math.pi * 300.0 * i / 8000.0)) for i in range(2400)],
    )
    audio_svc.transcribe = AsyncMock(return_value={"text": "What documents do I need for Video KYC?", "language": "en"})

    with (
        patch.object(ivr_mod, "PROCESSING_SOUND_ENABLED", True),
        patch.object(ivr_mod, "PROCESSING_SOUND_INITIAL_DELAY_SECONDS", 0.04),
        patch.object(ivr_mod, "PROCESSING_SOUND_INTERVAL_SECONDS", 1.0),
        patch("telephony.ivr_session.learn_from_voice_turn", AsyncMock()),
    ):
        await session._run_stt_ai_tts(pcm_voice)

    play_events = [json.loads(c.args[0]) for c in ws.send_text.await_args_list if '"play"' in c.args[0]]
    assert len(play_events) >= 2, f"Expected waiting chime + final reply play events, got: {play_events}"
    assert session._processing_task is None

    # 3. Fast turn (< initial delay) -> emits only the final reply play event (zero waiting chime)
    session_fast, ws_fast, _, _ = _make_session()
    await asyncio.sleep(0.02)
    session_fast._stop_playback(reason="test_setup")
    await asyncio.sleep(0.02)
    ws_fast.send_text.reset_mock()
    with (
        patch.object(ivr_mod, "PROCESSING_SOUND_ENABLED", True),
        patch.object(ivr_mod, "PROCESSING_SOUND_INITIAL_DELAY_SECONDS", 2.0),
        patch("telephony.ivr_session.learn_from_voice_turn", AsyncMock()),
    ):
        await session_fast._run_ai_and_reply("What documents do I need for Video KYC?")

    fast_play_events = [json.loads(c.args[0]) for c in ws_fast.send_text.await_args_list if '"play"' in c.args[0]]
    assert len(fast_play_events) == 1
    assert session_fast._processing_task is None

    # 4. Call hangup cancels an active waiting task cleanly
    session_hangup, _, _, _ = _make_session()
    session_hangup._start_processing_sound()
    assert session_hangup._processing_task is not None
    session_hangup.on_call_end({"uuid": session_hangup.call_uuid, "reason": "caller_hangup"})
    await asyncio.sleep(0.01)
    assert session_hangup._processing_task is None or session_hangup._processing_task.done()
