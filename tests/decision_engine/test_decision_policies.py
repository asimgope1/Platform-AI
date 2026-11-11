"""
Unit tests for Milestone 4 (Decision Engine Policies & ResponseOrchestrator).
"""
from __future__ import annotations

from decision_engine.decision_manager import Action, DecisionManager
from decision_engine.transfer_policy import TransferPolicy
from orchestration.response_orchestrator import ResponseOrchestrator
from telephony.intelligence.session_state import ConversationState


def _make_state(**overrides) -> ConversationState:
    defaults = dict(
        call_uuid="call-1234",
        tenant_id="tenant-abc",
        language="en",
        service_category="upi_payments",
        turn_number=1,
        intent="upi_refund",
        emotion={"text": "neutral", "voice": "neutral", "fused": "neutral", "trend": "stable", "confidence": 0.7},
        caller_certainty=0.80,
        caller_certainty_label="certain",
        system_confidence=0.85,
        satisfaction_score=0.65,
        satisfaction_label="moderate",
        transcript_confidence=0.88,
        audio_quality_score=0.85,
    )
    defaults.update(overrides)
    return ConversationState(**defaults)


class TestDecisionEnginePriorities:
    def test_explicit_agent_request_triggers_immediate_transfer(self):
        dm = DecisionManager()
        state = _make_state(explicit_agent_request=True, last_masked_transcript="connect me to agent")
        decision = dm.decide(state)
        assert decision.action == Action.TRANSFER
        assert decision.priority == 1
        payload = decision.metadata["transfer_payload"]
        assert payload["event"] == "transfer"
        assert payload["uuid"] == "call-1234"
        assert "summary" in payload

    def test_mandatory_fraud_escalation_triggers_transfer_in_hindi(self):
        dm = DecisionManager()
        state = _make_state(
            language="hi",
            requires_mandatory_escalation=True,
            last_masked_transcript="मेरे खाते से फ्रॉड हुआ है",
        )
        decision = dm.decide(state)
        assert decision.action == Action.TRANSFER
        assert any("\u0900" <= c <= "\u097f" for c in (decision.policy_prompt or ""))

    def test_agent_unavailable_falls_back_gracefully_without_crashing(self):
        dm = DecisionManager(transfer_policy=TransferPolicy(agent_available=False))
        state = _make_state(explicit_agent_request=True, language="or")
        decision = dm.decide(state)
        assert decision.action == Action.CONTINUE
        assert decision.reason == "transfer_agent_unavailable"
        assert decision.policy_prompt is not None
        assert any("\u0b00" <= c <= "\u0b7f" for c in decision.policy_prompt)

    def test_low_audio_or_uncertain_transcript_triggers_clarify(self):
        dm = DecisionManager()
        state = _make_state(audio_quality_score=0.20, transcript_confidence=0.35)
        decision = dm.decide(state, needs_clarification=True)
        assert decision.action == Action.CLARIFY

    def test_consecutive_bad_audio_or_frustration_offers_transfer(self):
        dm = DecisionManager()
        state = _make_state(clarification_count=2, audio_quality_score=0.20)
        decision = dm.decide(state, needs_clarification=True)
        assert decision.action == Action.OFFER_TRANSFER

        state_frust = _make_state(
            failed_attempts=2,
            emotion={"text": "frustrated", "voice": "neutral", "fused": "frustrated", "trend": "worsening", "confidence": 0.8},
        )
        decision2 = dm.decide(state_frust)
        assert decision2.action == Action.OFFER_TRANSFER

    def test_low_caller_certainty_or_amounts_triggers_confirmation(self):
        dm = DecisionManager()
        state = _make_state(
            caller_certainty=0.35,
            caller_certainty_label="uncertain",
            detected_amounts=["2000"],
        )
        decision = dm.decide(state, needs_confirmation=True)
        assert decision.action == Action.CONFIRM
        assert "2000" in (decision.policy_prompt or "")

    def test_feedback_gated_strictly_behind_resolution_confirmation(self):
        dm = DecisionManager()
        # Potentially resolved -> must confirm resolution first
        s1 = _make_state(resolution_status="potentially_resolved", customer_confirmed=False)
        d1 = dm.decide(s1)
        assert d1.action == Action.CONFIRM
        assert d1.reason == "confirm_potential_resolution"

        # Confirmed resolved -> asks for 1–5 feedback
        s2 = _make_state(resolution_status="resolved", customer_confirmed=True)
        d2 = dm.decide(s2)
        assert d2.action == Action.FEEDBACK
        assert d2.reason == "request_post_resolution_feedback"

        # Rating received -> thanks caller
        s3 = _make_state(
            resolution_status="resolved",
            customer_confirmed=True,
            feedback_received=True,
            explicit_rating=5,
            awaiting_feedback=False,
        )
        d3 = dm.decide(s3)
        assert d3.action == Action.FEEDBACK
        assert "5" in (d3.policy_prompt or "")

    def test_response_orchestrator_updates_lifecycle_flags(self):
        dm = DecisionManager()
        ro = ResponseOrchestrator()
        state = _make_state(resolution_status="resolved", customer_confirmed=True)
        dec = dm.decide(state)
        plan = ro.prepare(dec, state)
        assert plan.use_llm is False
        assert state.feedback_requested is True
        assert state.awaiting_feedback is True
