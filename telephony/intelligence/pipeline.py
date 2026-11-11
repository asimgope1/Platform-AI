"""
Telephony Intelligence Pipeline — End-to-End Coordinator for IVRCallSession.

Connects:
  • IntelligenceOrchestrator (Milestones 2–4)
  • ResponseOrchestrator (policy & LLM prompt summary routing)
  • IntelligenceEventHandler (non-blocking bounded DB telemetry & feedback logging)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from audio_intelligence.audio_emotion import AcousticFeatures
from audio_intelligence.audio_quality import AudioQualityReport
from decision_engine.decision_manager import Action
from orchestration.intelligence_orchestrator import IntelligenceOrchestrator, TurnIntelligencePackage
from orchestration.response_orchestrator import PlannedResponse, ResponseOrchestrator
from telephony.intelligence.event_handler import IntelligenceEventHandler
from telephony.intelligence.session_state import ConversationState

logger = logging.getLogger(__name__)


class TelephonyIntelligencePipeline:
    """Per-session Hybrid Voice Intelligence coordinator."""

    def __init__(
        self,
        persistence_enabled: bool = True,
        orchestrator: Optional[IntelligenceOrchestrator] = None,
        response_orchestrator: Optional[ResponseOrchestrator] = None,
        event_handler: Optional[IntelligenceEventHandler] = None,
    ) -> None:
        self.persistence_enabled = persistence_enabled
        self.orchestrator = orchestrator or IntelligenceOrchestrator()
        self.response_orchestrator = response_orchestrator or ResponseOrchestrator()
        self.event_handler = event_handler or IntelligenceEventHandler()

    async def process_and_plan(
        self,
        stt_result: Dict[str, Any] | str,
        state: ConversationState,
        audio: bytes = b"",
        audio_quality: Optional[AudioQualityReport] = None,
        acoustic_features: Optional[AcousticFeatures] = None,
    ) -> tuple[TurnIntelligencePackage, PlannedResponse]:
        prev_emotion = state.emotion.get("fused", "neutral")

        pkg = await self.orchestrator.process_turn(
            stt_result=stt_result,
            state=state,
            audio=audio,
            audio_quality=audio_quality,
            acoustic_features=acoustic_features,
        )
        plan = self.response_orchestrator.prepare(decision=pkg.decision, state=state)

        logger.info(
            f"[VoiceIntel] Turn={state.turn_number} Intent={state.intent} "
            f"Emotion={state.emotion.get('fused')}({state.emotion.get('trend')}) "
            f"Certainty={state.caller_certainty} SysConf={state.system_confidence} "
            f"Action={plan.action.value} Reason={pkg.decision.reason}"
        )

        if self.persistence_enabled and state.call_uuid and state.call_uuid != "unknown":
            self._emit_turn_telemetry(state, pkg, prev_emotion)

        return pkg, plan

    def _emit_turn_telemetry(
        self,
        state: ConversationState,
        pkg: TurnIntelligencePackage,
        prev_emotion: str,
    ) -> None:
        conv = pkg.conversation
        self.event_handler.emit_turn(
            tenant_id=state.tenant_id,
            call_uuid=state.call_uuid,
            turn_number=state.turn_number,
            intent=state.intent,
            text_emotion=conv.text_emotion.primary_emotion,
            voice_emotion=conv.voice_emotion.primary_emotion,
            emotion_confidence=conv.fused_emotion_confidence,
            emotion_trend=state.emotion.get("trend"),
            caller_certainty=state.caller_certainty,
            system_confidence=state.system_confidence,
            satisfaction_score=state.satisfaction_score,
            transcript_confidence=state.transcript_confidence,
            audio_quality_score=state.audio_quality_score,
            resolution_status=state.resolution_status,
            recommended_action=pkg.decision.action.value,
            analysis={
                "masked_transcript": pkg.transcript.masked_text,
                "issues": pkg.transcript.issues,
                "emotion_signals": conv.text_emotion.signals,
                "experimental_voice_emotion": conv.voice_emotion.experimental_signal,
            },
        )

        if state.emotion.get("fused") != prev_emotion:
            self.event_handler.emit_event(
                tenant_id=state.tenant_id,
                call_uuid=state.call_uuid,
                event_type="emotion_changed",
                event_data={"from": prev_emotion, "to": state.emotion.get("fused")},
            )

        if pkg.decision.action == Action.CLARIFY:
            self.event_handler.emit_event(
                tenant_id=state.tenant_id,
                call_uuid=state.call_uuid,
                event_type="clarification_requested",
                event_data={"reason": pkg.decision.reason},
            )
        elif pkg.decision.action == Action.OFFER_TRANSFER:
            self.event_handler.emit_event(
                tenant_id=state.tenant_id,
                call_uuid=state.call_uuid,
                event_type="agent_transfer_offered",
                event_data={"reason": pkg.decision.reason},
            )
        elif pkg.decision.action == Action.TRANSFER:
            self.event_handler.emit_transfer(
                tenant_id=state.tenant_id,
                call_uuid=state.call_uuid,
                transfer_reason=pkg.decision.reason,
                emotion_at_transfer=state.emotion.get("fused", "neutral"),
                satisfaction_at_transfer=state.satisfaction_score,
                caller_accepted=bool(state.transfer_accepted or state.explicit_agent_request),
                transfer_status="initiated",
                agent_identifier="pbx_queue_default",
            )
        elif state.feedback_received and state.explicit_rating is not None:
            self.event_handler.emit_feedback(
                tenant_id=state.tenant_id,
                call_uuid=state.call_uuid,
                rating=state.explicit_rating,
                feedback_text=pkg.transcript.masked_text,
                satisfaction_label=state.satisfaction_label,
                issue_resolved=(state.resolution_status == "resolved"),
                language=state.language,
            )

    def record_call_end(self, state: ConversationState) -> None:
        if not self.persistence_enabled or not state.call_uuid or state.call_uuid == "unknown":
            return
        self.event_handler.emit_resolution(
            tenant_id=state.tenant_id,
            call_uuid=state.call_uuid,
            initial_intent=state.initial_intent or state.service_category,
            final_resolution=state.resolution_status,
            issue_resolved=(state.resolution_status == "resolved"),
            resolved_by="human_agent" if state.transfer_completed else "ai_agent",
            customer_confirmed=state.customer_confirmed,
            escalation_required=bool(state.transfer_completed or state.requires_mandatory_escalation),
        )
