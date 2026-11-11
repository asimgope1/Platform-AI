"""
Deterministic Decision Engine (Milestone 4).

Controls what the voice system does next strictly via auditable policy rules:
  Action.TRANSFER         — immediate PBX handoff (explicit request, mandatory escalation, or accepted)
  Action.CLARIFY          — ask caller to repeat cleanly when audio/transcript failed
  Action.OFFER_TRANSFER   — offer human agent after repeated failure or worsening frustration
  Action.CONFIRM          — read-back confirmation for uncertainty, amounts, or potential resolution
  Action.FEEDBACK         — request 1–5 rating after confirmed resolution
  Action.CONTINUE         — proceed with RAG + Gemma / Mini-Brain conversational reply
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from decision_engine.confirmation_policy import ConfirmationPolicy
from decision_engine.escalation_rules import EscalationRules
from decision_engine.feedback_policy import FeedbackPolicy
from decision_engine.transfer_policy import TransferPolicy

if TYPE_CHECKING:
    from telephony.intelligence.session_state import ConversationState


class Action(str, Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    OFFER_TRANSFER = "offer_transfer"
    TRANSFER = "transfer"
    FEEDBACK = "feedback"


@dataclass
class DecisionResult:
    action: Action
    reason: str
    policy_prompt: Optional[str] = None
    priority: int = 99
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "policy_prompt": self.policy_prompt,
            "priority": self.priority,
            "metadata": self.metadata or {},
        }


class DecisionManager:
    """Deterministic policy controller governing telephony conversation actions."""

    def __init__(
        self,
        audio_clarify_threshold: float = 0.35,
        transcript_confirm_threshold: float = 0.60,
        escalation_rules: Optional[EscalationRules] = None,
        transfer_policy: Optional[TransferPolicy] = None,
        confirmation_policy: Optional[ConfirmationPolicy] = None,
        feedback_policy: Optional[FeedbackPolicy] = None,
    ) -> None:
        self.audio_clarify_threshold = audio_clarify_threshold
        self.transcript_confirm_threshold = transcript_confirm_threshold
        self.escalation_rules = escalation_rules or EscalationRules()
        self.transfer_policy = transfer_policy or TransferPolicy()
        self.confirmation_policy = confirmation_policy or ConfirmationPolicy()
        self.feedback_policy = feedback_policy or FeedbackPolicy()

    def decide(
        self,
        state: ConversationState,
        needs_clarification: bool = False,
        needs_confirmation: bool = False,
    ) -> DecisionResult:
        # 1. Explicit agent request / Mandatory escalation / Accepted transfer (Priority 1–2)
        esc = self.escalation_rules.evaluate(state)
        if esc.should_transfer_immediately:
            reason = esc.reason or "escalation_triggered"
            if self.transfer_policy.agent_available:
                handoff = self.transfer_policy.build_handoff_payload(state, reason=reason)
                return DecisionResult(
                    action=Action.TRANSFER,
                    reason=reason,
                    policy_prompt=self.transfer_policy.get_initiate_prompt(state.language),
                    priority=1,
                    metadata={"transfer_payload": handoff.ws_event, "summary": handoff.summary},
                )
            return DecisionResult(
                action=Action.CONTINUE,
                reason="transfer_agent_unavailable",
                policy_prompt=self.transfer_policy.get_unavailable_prompt(state.language),
                priority=1,
            )

        # 2. If feedback rating was just received, acknowledge and conclude feedback
        if state.feedback_received and state.explicit_rating is not None and not state.awaiting_feedback:
            return DecisionResult(
                action=Action.FEEDBACK,
                reason="feedback_received",
                policy_prompt=self.feedback_policy.get_feedback_thanks(state.language, state.explicit_rating),
                priority=2,
                metadata={"rating": state.explicit_rating, "completed": True},
            )

        # 3. Audio / Transcription Failure (Priority 3)
        if needs_clarification or (
            state.audio_quality_score > 0.0 and state.audio_quality_score < self.audio_clarify_threshold
        ):
            # After 2 consecutive clarifications, offer human transfer if available
            if state.clarification_count >= 2 and not state.transfer_offered:
                return DecisionResult(
                    action=Action.OFFER_TRANSFER,
                    reason="repeated_audio_clarification",
                    policy_prompt=self.transfer_policy.get_offer_prompt(state.language),
                    priority=3,
                )
            return DecisionResult(
                action=Action.CLARIFY,
                reason="low_audio_or_transcript_quality",
                policy_prompt=self.confirmation_policy.get_clarify_audio_prompt(state.language),
                priority=3,
            )

        # 4. Repeated Failure or Worsening Dissatisfaction -> Offer Human Agent (Priority 4)
        if esc.should_offer_transfer:
            return DecisionResult(
                action=Action.OFFER_TRANSFER,
                reason=esc.reason or "offer_transfer",
                policy_prompt=self.transfer_policy.get_offer_prompt(state.language),
                priority=4,
            )

        # 5. Resolution confirmed -> Collect Satisfaction Feedback (Priority 5)
        if state.resolution_status == "resolved":
            if state.customer_confirmed and not state.feedback_requested and not state.feedback_received:
                return DecisionResult(
                    action=Action.FEEDBACK,
                    reason="request_post_resolution_feedback",
                    policy_prompt=self.feedback_policy.get_feedback_prompt(state.language),
                    priority=5,
                    metadata={"completed": False},
                )
            if not state.customer_confirmed and not state.awaiting_resolution_confirmation:
                return DecisionResult(
                    action=Action.CONFIRM,
                    reason="confirm_issue_resolution",
                    policy_prompt=self.confirmation_policy.get_resolution_confirm_prompt(state.language),
                    priority=5,
                )

        # 6. Potentially resolved -> ask confirmation before feedback
        if state.resolution_status == "potentially_resolved" and not state.customer_confirmed and not state.awaiting_resolution_confirmation:
            return DecisionResult(
                action=Action.CONFIRM,
                reason="confirm_potential_resolution",
                policy_prompt=self.confirmation_policy.get_resolution_confirm_prompt(state.language),
                priority=6,
            )

        # 7. Low Transcript Confidence or Monetary Read-back Confirmation (Priority 7)
        if needs_confirmation or (
            0.0 < state.transcript_confidence < self.transcript_confirm_threshold
            and state.intent not in ("greeting", "confirmation_yes", "confirmation_no")
        ):
            return DecisionResult(
                action=Action.CONFIRM,
                reason="low_transcript_confidence_or_entity",
                policy_prompt=self.confirmation_policy.get_confirm_prompt(state),
                priority=7,
            )

        # 8. Default: Continue conversational RAG + Gemma/Mini-Brain handling
        return DecisionResult(
            action=Action.CONTINUE,
            reason="normal_conversation",
            policy_prompt=None,
            priority=8,
        )
