"""
End-to-End Integration & Resilience Tests for Hybrid Voice Intelligence (Milestones 1–6).
Covers:
  • Async SQLite DB persistence across all 5 intelligence tables
  • IVRCallSession Gemma-offline resilient fallback across English, Hindi, and Odia
  • Asterisk WebSocket structured JSON transfer event emission & non-fatal recovery
  • Post-resolution feedback gating (AI_CONVERSATION -> FEEDBACK -> rated)
  • Voice learning quality gate (skips uncertain transcripts, masks PII)
  • Turn intelligence latency budget verification
"""
from __future__ import annotations

import asyncio
import json
import time
import wave
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from decision_engine.decision_manager import Action, DecisionManager
from decision_engine.transfer_policy import TransferPolicy
from storage.database import Base
from storage.models.call_events import CallEvent, CallTransfer, ConversationResolution
from storage.models.call_feedback import CallFeedback
from storage.models.call_intelligence import CallIntelligence
from telephony.intelligence.event_handler import IntelligenceEventHandler
from telephony.intelligence.pipeline import TelephonyIntelligencePipeline
from telephony.intelligence.session_state import ConversationState
from telephony.ivr_menu import IVRState
from telephony.ivr_session import IVRCallSession
from telephony.voice_learning import learn_from_voice_turn


def _make_dummy_wav() -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(np.zeros(800, dtype="<i2").tobytes())
    return buf.getvalue()


def _make_session(orchestrator=None, transfer_agent_available: bool = True):
    ws = MagicMock()
    ws.send_bytes = AsyncMock()
    ws.send_text = AsyncMock()

    if orchestrator is None:
        orchestrator = MagicMock()
        orchestrator.execute = AsyncMock(side_effect=RuntimeError("Gemma gateway offline"))

    audio_svc = MagicMock()
    audio_svc.synthesize = AsyncMock(return_value=_make_dummy_wav())

    session = IVRCallSession(ws=ws, orchestrator=orchestrator, audio_service=audio_svc)
    session.call_uuid = "pbx-call-9001"
    session.menu_machine.state = IVRState.AI_CONVERSATION
    session.selected_language = "en"
    session._language = "en"
    session.selected_service = "upi_payments"
    session.conv_state.call_uuid = "pbx-call-9001"

    if not transfer_agent_available:
        session.intel_pipeline = TelephonyIntelligencePipeline(
            persistence_enabled=False,
        )
        session.intel_pipeline.orchestrator.decision_manager = DecisionManager(
            transfer_policy=TransferPolicy(agent_available=False)
        )
    else:
        session.intel_pipeline.persistence_enabled = False

    return session, ws, orchestrator, audio_svc


class TestHybridIntelligenceDatabasePersistence:
    @pytest.mark.asyncio
    async def test_full_call_lifecycle_persists_all_five_tables(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        intel_tables = [
            CallIntelligence.__table__,
            CallEvent.__table__,
            CallTransfer.__table__,
            ConversationResolution.__table__,
            CallFeedback.__table__,
        ]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, tables=intel_tables)

        session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        handler = IntelligenceEventHandler()
        pipeline = TelephonyIntelligencePipeline(persistence_enabled=True, event_handler=handler)
        state = ConversationState(call_uuid="uuid-db-01", tenant_id="tenant-xyz", language="en")

        with patch("telephony.intelligence.event_handler.async_session", session_factory):
            # Turn 1: Concerned UPI failure -> emits CallIntelligence + emotion_changed
            await pipeline.process_and_plan(
                {"text": "My UPI payment of rs 1500 failed and 987654321012 was debited"},
                state=state,
            )
            # Turn 2: Resolved confirmation -> asks feedback
            await pipeline.process_and_plan(
                {"text": "Got it, issue solved, thank you"},
                state=state,
            )
            # Turn 3: Feedback rating 5 -> emits CallFeedback
            await pipeline.process_and_plan(
                {"text": "5 out of 5, very helpful"},
                state=state,
            )
            # Turn 4: Explicit human transfer -> emits CallTransfer
            await pipeline.process_and_plan(
                {"text": "Actually please connect me to a human representative"},
                state=state,
            )
            # Call end -> emits ConversationResolution
            pipeline.record_call_end(state)

            await handler._queue.join()
            if handler._worker_task and not handler._worker_task.done():
                handler._worker_task.cancel()

        async with session_factory() as db:
            turns = (await db.execute(select(CallIntelligence))).scalars().all()
            events = (await db.execute(select(CallEvent))).scalars().all()
            feedbacks = (await db.execute(select(CallFeedback))).scalars().all()
            transfers = (await db.execute(select(CallTransfer))).scalars().all()
            resolutions = (await db.execute(select(ConversationResolution))).scalars().all()

        await engine.dispose()

        assert len(turns) == 4
        assert "[REDACTED_12DIGITS]" in turns[0].analysis["masked_transcript"]
        assert "987654321012" not in turns[0].analysis["masked_transcript"]
        assert len(events) >= 1
        assert len(feedbacks) == 1
        assert feedbacks[0].rating == 5
        assert len(transfers) == 1
        assert transfers[0].transfer_status == "initiated"
        assert len(resolutions) == 1
        assert resolutions[0].resolved_by == "human_agent"


class TestIVRSessionHybridIntegration:
    @pytest.mark.asyncio
    async def test_gemma_offline_fallback_and_context_injection_multilingual(self):
        for lang, query, check_fn in [
            ("en", "My UPI refund has been pending since yesterday", lambda s: len(s) > 10),
            ("hi", "मेरा यूपीआई पेमेंट अटक गया है", lambda s: any("\u0900" <= c <= "\u097f" for c in s)),
            ("or", "ମୋର ୟୁପିଆଇ ପେମେଣ୍ଟ ଅଟକି ଯାଇଛି", lambda s: any("\u0b00" <= c <= "\u0b7f" for c in s)),
        ]:
            session, ws, orch, _ = _make_session()
            session.selected_language = lang
            session._language = lang
            session.conv_state.language = lang

            with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)):
                await session._run_ai_and_reply(query)

            orch.execute.assert_awaited()
            call_kwargs = orch.execute.call_args.kwargs
            assert "intelligence_context" in call_kwargs
            assert call_kwargs["intelligence_context"]["intent"] in ("upi_refund", "upi_payment_failed")

            assistant_replies = [m.content for m in session._history if m.role == "assistant"]
            assert len(assistant_replies) == 1
            assert check_fn(assistant_replies[0])
            ws.send_bytes.assert_awaited()

    @pytest.mark.asyncio
    async def test_asterisk_transfer_event_emitted_and_ws_failure_non_fatal(self):
        session, ws, _, _ = _make_session()
        with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)):
            await session._run_ai_and_reply("Connect me to an agent now, unauthorized fraud on my account 123456789012")

        assert session.menu_machine.state == IVRState.ESCALATED
        sent_events = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        transfer_events = [ev for ev in sent_events if ev.get("event") == "transfer"]
        assert len(transfer_events) == 1
        event_payload = transfer_events[0]
        assert event_payload["uuid"] == "pbx-call-9001"
        assert "123456789012" not in event_payload["summary"]

        # Simulate WebSocket text send failure -> must recover safely without raising
        session2, ws2, _, _ = _make_session()
        ws2.send_text = AsyncMock(side_effect=ConnectionError("PBX socket closed"))
        with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)):
            await session2._run_ai_and_reply("Transfer me to human agent")
        assert session2.menu_machine.state == IVRState.ESCALATED

    @pytest.mark.asyncio
    async def test_agent_unavailable_recovers_with_polite_fallback(self):
        session, ws, _, _ = _make_session(transfer_agent_available=False)
        with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)):
            await session._run_ai_and_reply("Speak to representative please")
        sent_events = [json.loads(c.args[0]) for c in ws.send_text.call_args_list]
        transfer_events = [ev for ev in sent_events if ev.get("event") == "transfer"]
        assert len(transfer_events) == 0
        assistant_replies = [m.content for m in session._history if m.role == "assistant"]
        assert "agents are currently busy" in assistant_replies[0].lower() or "sorry" in assistant_replies[0].lower()

    @pytest.mark.asyncio
    async def test_post_resolution_feedback_flow_updates_state(self):
        session, _, _, _ = _make_session()
        with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)):
            await session._run_ai_and_reply("Yes, that solved my problem, thank you!")
            assert session.conv_state.feedback_requested is True
            assert session.menu_machine.state == IVRState.FEEDBACK

            await session._run_ai_and_reply("I rate this 5 out of 5")
            assert session.conv_state.feedback_received is True
            assert session.conv_state.explicit_rating == 5

    @pytest.mark.asyncio
    async def test_voice_learning_skips_low_quality_transcripts(self):
        skipped = await learn_from_voice_turn(
            tenant_id="t1",
            call_uuid="c1",
            user_text="How do I complete my video KYC with PAN?",
            reply_text="Please keep your PAN card ready and follow the mobile app link.",
            quality_ok=False,
        )
        assert skipped is False

    @pytest.mark.asyncio
    async def test_full_stt_to_ai_pipeline_with_real_audio_and_diagnostics(self):
        session, ws, _, audio_svc = _make_session()
        audio_svc.transcribe = AsyncMock(
            return_value={
                "text": "My UPI refund of rs 500 is pending",
                "language": "en",
                "diagnostics": {"avg_logprob": -0.22, "no_speech_prob": 0.04, "compression_ratio": 1.1},
            }
        )
        sr = 8000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        pcm = (np.sin(2 * np.pi * 240 * t) * 7000).astype("<i2").tobytes()

        with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)):
            await session._run_stt_ai_tts(pcm)

        assert session.last_audio_quality is not None
        assert session.last_enhanced_audio is not None
        assert session.conv_state.intent == "upi_refund"
        assert session.conv_state.detected_amounts == ["500"]
        ws.send_bytes.assert_awaited()

    @pytest.mark.asyncio
    async def test_turn_intelligence_latency_within_real_time_budget(self):
        pipeline = TelephonyIntelligencePipeline(persistence_enabled=False)
        state = ConversationState(call_uuid="perf-1", tenant_id="t1", language="en")
        sr = 8000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        pcm = (np.sin(2 * np.pi * 220 * t) * 5000).astype("<i2").tobytes()

        start = time.perf_counter()
        for idx in range(5):
            pkg, plan = await pipeline.process_and_plan(
                {"text": "My UPI refund of rs 500 is still pending", "diagnostics": {"avg_logprob": -0.2}},
                state=state,
                audio=pcm,
            )
            assert pkg.decision.action in tuple(Action)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        avg_turn_ms = elapsed_ms / 5.0
        # Target real-time non-LLM intelligence overhead well under 150ms per turn
        assert avg_turn_ms < 150.0, f"Average intelligence latency too high: {avg_turn_ms:.1f}ms"
