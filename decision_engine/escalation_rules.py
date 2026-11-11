"""
Deterministic Escalation Rules for Banking IVR Safety & Reliability.

Gemma never decides transfer or banking escalation policies independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from telephony.intelligence.session_state import ConversationState


@dataclass
class EscalationEvaluation:
    should_transfer_immediately: bool
    should_offer_transfer: bool
    reason: Optional[str] = None


class EscalationRules:
    """Evaluates security and dissatisfaction escalation triggers."""

    def __init__(
        self,
        transfer_after_failed_attempts: int = 2,
        low_satisfaction_threshold: float = 0.35,
    ) -> None:
        self.transfer_after_failed_attempts = transfer_after_failed_attempts
        self.low_satisfaction_threshold = low_satisfaction_threshold

    def evaluate(self, state: ConversationState) -> EscalationEvaluation:
        if state.explicit_agent_request:
            return EscalationEvaluation(
                should_transfer_immediately=True,
                should_offer_transfer=False,
                reason="explicit_agent_request",
            )

        if state.requires_mandatory_escalation:
            return EscalationEvaluation(
                should_transfer_immediately=True,
                should_offer_transfer=False,
                reason="mandatory_security_escalation",
            )

        if state.transfer_accepted:
            return EscalationEvaluation(
                should_transfer_immediately=True,
                should_offer_transfer=False,
                reason="caller_accepted_transfer",
            )

        if (
            not state.transfer_offered
            and state.satisfaction_score is not None
            and state.satisfaction_score < self.low_satisfaction_threshold
            and state.failed_attempts >= self.transfer_after_failed_attempts
        ):
            return EscalationEvaluation(
                should_transfer_immediately=False,
                should_offer_transfer=True,
                reason="repeated_failure_low_satisfaction",
            )

        if (
            not state.transfer_offered
            and state.emotion.get("trend") == "worsening"
            and state.failed_attempts >= self.transfer_after_failed_attempts
        ):
            return EscalationEvaluation(
                should_transfer_immediately=False,
                should_offer_transfer=True,
                reason="worsening_emotion_repeated_failure",
            )

        return EscalationEvaluation(
            should_transfer_immediately=False,
            should_offer_transfer=False,
            reason=None,
        )
