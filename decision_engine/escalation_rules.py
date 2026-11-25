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
        # 1. Explicit Agent Transfer Requests
        if state.explicit_agent_request:
            return EscalationEvaluation(
                should_transfer_immediately=True,
                should_offer_transfer=False,
                reason="explicit_agent_request",
            )

        # 2. High Frustration / Distress / Anger Sentiment Detection
        fused_emo = str(state.emotion.get("fused") or "").lower()
        emo_trend = str(state.emotion.get("trend") or "").lower()
        last_text = str(state.last_masked_transcript or "").lower()

        frustration_keywords = (
            "fed up", "useless", "waste of time", "ridiculous", "frustrated", "stop repeating",
            "terrible", "horrible", "screw", "angry", "hate this", "damn", "manager", "connect human",
            "talk to human", "speak to someone", "representative", "agent",
            "बकवास", "बेकार", "परेशान हो गया", "दिमाग खराब", "पागल बना रहे हो", "गुस्सा", "फालतू",
            "टाइम खराब", "मज़ाक बना रखा है", "चुप करो", "एजेंट से बात", "कस्टमर केयर",
            "ବେକାର", "ବକବାସ", "ରାଗ ଲାଗୁଛି", "ସମୟ ନଷ୍ଟ", "କିଛି ବୁଝୁନ", "ଅସୁବିଧା ହେଉଛି", "କଣ ମଜାକ କରୁଛ",
            "ପ୍ରତିନିଧି", "ଏଜେଣ୍ଟ"
        )
        has_explicit_frustration_kw = any(kw in last_text for kw in frustration_keywords)
        is_appreciation = str(getattr(state, "intent", "") or "").lower() == "appreciation"
        is_high_frustration = (
            not (is_appreciation and not has_explicit_frustration_kw)
            and (
                fused_emo in ("frustration", "anger", "angry", "frustrated", "distressed")
                or has_explicit_frustration_kw
                or (emo_trend == "worsening" and state.failed_attempts >= 1)
            )
        )

        if is_high_frustration:
            return EscalationEvaluation(
                should_transfer_immediately=True,
                should_offer_transfer=False,
                reason="sentiment_high_frustration",
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

        # Do not interrupt active multi-turn data collection (complaint slot filling or order ID entry)
        if getattr(state, "awaiting_complaint_slots", False) or getattr(state, "awaiting_order_id_digits", False):
            return EscalationEvaluation(
                should_transfer_immediately=False,
                should_offer_transfer=False,
                reason=None,
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
