"""
Conversation State Manager — Central state container per active call_uuid.

Maintains strict separation between:
  • Emotion (how the caller may be feeling: text, voice, fused, trend)
  • Caller certainty (how sure the caller is about their statement)
  • System confidence (calibrated recognition/interpretation confidence)
  • Satisfaction (explicit rating vs conversational estimate)
  • Resolution & transfer lifecycle flags
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from audio_intelligence.audio_quality import AudioQualityReport
from conversation_intelligence.conversation_analyzer import ConversationIntelligenceResult
from conversation_intelligence.conversation_memory import ConversationMemory
from conversation_intelligence.transcript_confidence import TranscriptAnalysis


@dataclass
class ConversationState:
    call_uuid: str
    tenant_id: str
    language: str = "en"
    service_category: Optional[str] = None
    turn_number: int = 0
    initial_intent: Optional[str] = None
    intent: Optional[str] = None

    emotion: Dict[str, Any] = field(
        default_factory=lambda: {
            "text": "unknown",
            "voice": "unknown",
            "fused": "neutral",
            "trend": "stable",
            "confidence": 0.0,
        }
    )
    caller_certainty: Optional[float] = None
    caller_certainty_label: str = "moderate"
    system_confidence: float = 0.0
    satisfaction_score: Optional[float] = None
    satisfaction_label: str = "unknown"
    explicit_rating: Optional[int] = None

    resolution_status: str = "unresolved"
    customer_confirmed: bool = False
    transcript_confidence: float = 0.0
    audio_quality_score: float = 0.0

    clarification_count: int = 0
    failed_attempts: int = 0
    consecutive_low_audio_turns: int = 0

    explicit_agent_request: bool = False
    requires_mandatory_escalation: bool = False
    transfer_offered: bool = False
    transfer_accepted: bool = False
    transfer_completed: bool = False
    transfer_reason: Optional[str] = None

    feedback_requested: bool = False
    feedback_received: bool = False

    # Pending conversational state guards
    awaiting_transfer_confirmation: bool = False
    awaiting_resolution_confirmation: bool = False
    awaiting_feedback: bool = False
    awaiting_order_id_digits: bool = False
    order_id_last4: Optional[str] = None

    last_transcript: str = ""
    last_masked_transcript: str = ""
    detected_amounts: List[str] = field(default_factory=list)
    memory: ConversationMemory = field(default_factory=lambda: ConversationMemory(maxlen=10))

    def update_from_turn(
        self,
        audio_quality: Optional[AudioQualityReport],
        transcript: TranscriptAnalysis,
        intelligence: ConversationIntelligenceResult,
    ) -> None:
        """Update all dimensions from a newly analyzed caller turn."""
        self.turn_number += 1
        self.last_transcript = transcript.normalized_text
        self.last_masked_transcript = transcript.masked_text
        self.detected_amounts = list(transcript.detected_amounts)

        if audio_quality is not None:
            self.audio_quality_score = float(audio_quality.audio_quality_score)
            if audio_quality.audio_quality_score < 0.35:
                self.consecutive_low_audio_turns += 1
            else:
                self.consecutive_low_audio_turns = 0

        self.transcript_confidence = float(transcript.transcript_confidence)

        # Track confirmation acceptance of prior transfer offer
        if self.awaiting_transfer_confirmation:
            if intelligence.intent.intent == "confirmation_yes":
                self.transfer_accepted = True
                self.awaiting_transfer_confirmation = False
            elif intelligence.intent.intent == "confirmation_no":
                self.transfer_accepted = False
                self.awaiting_transfer_confirmation = False

        # Update intent (preserve primary domain intent across brief confirmations)
        if intelligence.intent.intent not in ("unknown", "confirmation_yes", "confirmation_no"):
            self.intent = intelligence.intent.intent
            if self.initial_intent is None and self.intent not in ("greeting",):
                self.initial_intent = self.intent

        self.explicit_agent_request = bool(intelligence.intent.explicit_agent_request)
        self.requires_mandatory_escalation = bool(
            self.requires_mandatory_escalation or intelligence.intent.requires_mandatory_escalation
        )
        if self.requires_mandatory_escalation and not self.transfer_reason:
            self.transfer_reason = "mandatory_security_escalation"
        elif self.explicit_agent_request and not self.transfer_reason:
            self.transfer_reason = "caller_requested_agent"

        self.caller_certainty = float(intelligence.caller_confidence.caller_certainty)
        self.caller_certainty_label = intelligence.caller_confidence.certainty_label
        self.system_confidence = round(
            0.70 * self.transcript_confidence + 0.30 * float(intelligence.intent.confidence),
            3,
        )

        # Feedback rating tracking
        if intelligence.intent.rating_value is not None:
            self.explicit_rating = intelligence.intent.rating_value
            self.feedback_received = True
            self.awaiting_feedback = False

        # Resolution & satisfaction
        self.resolution_status = intelligence.resolution.resolution_status
        self.customer_confirmed = intelligence.resolution.customer_confirmed
        self.satisfaction_score = intelligence.satisfaction.satisfaction_score
        self.satisfaction_label = intelligence.satisfaction.satisfaction_label

        # Record in sliding window and detect repeated failures/trend
        trend = self.memory.record_turn(
            turn_number=self.turn_number,
            intent=self.intent or "unknown",
            primary_emotion=intelligence.fused_emotion,
            satisfaction=self.satisfaction_score,
            resolution_status=self.resolution_status,
        )

        # Increment failed attempts if caller repeats unresolved intent or explicitly rejects resolution.
        # Do not penalize legitimate multi-turn slot filling (e.g. supplying order ID digits) or positive/closing turns.
        non_failure_intents = {"appreciation", "greeting", "feedback_rating", "confirmation_yes"}
        if (
            (
                self.memory.consecutive_same_intent >= 1
                and not self.awaiting_order_id_digits
                and intelligence.intent.intent not in non_failure_intents
            )
            or (self.awaiting_resolution_confirmation and intelligence.intent.intent == "confirmation_no")
        ):
            self.failed_attempts += 1

        if intelligence.intent.intent in ("confirmation_yes", "confirmation_no"):
            self.awaiting_resolution_confirmation = False

        self.emotion = {
            "text": intelligence.text_emotion.primary_emotion,
            "voice": intelligence.voice_emotion.primary_emotion,
            "fused": intelligence.fused_emotion,
            "trend": trend,
            "confidence": intelligence.fused_emotion_confidence,
        }

    def to_intelligence_context(self, recommended_action: str = "continue") -> Dict[str, Any]:
        """Structured summary injected into Gemma and Mini-Brain prompts."""
        return {
            "intent": self.intent or self.service_category or "general_inquiry",
            "emotion": dict(self.emotion),
            "caller_certainty": self.caller_certainty_label,
            "system_confidence": self.system_confidence,
            "transcript_quality": (
                "high" if self.transcript_confidence >= 0.75
                else ("medium" if self.transcript_confidence >= 0.50 else "low")
            ),
            "satisfaction": self.satisfaction_label,
            "resolution": self.resolution_status,
            "previous_attempts": self.failed_attempts,
            "recommended_action": recommended_action,
            "language": self.language,
            "order_id_last4": self.order_id_last4,
            "awaiting_order_id_digits": self.awaiting_order_id_digits,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("memory", None)
        return data

