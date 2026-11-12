"""
Intelligence Orchestrator — Coordinates Transcript Intelligence, Parallel
Conversation Analysis, ConversationState Update, and Deterministic Decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from audio_intelligence.audio_emotion import AcousticFeatures
from audio_intelligence.audio_quality import AudioQualityReport
from conversation_intelligence.conversation_analyzer import (
    ConversationAnalyzer,
    ConversationIntelligenceResult,
)
from conversation_intelligence.transcript_confidence import TranscriptAnalysis, TranscriptAnalyzer
from decision_engine.decision_manager import DecisionManager, DecisionResult
from telephony.intelligence.session_state import ConversationState


@dataclass
class TurnIntelligencePackage:
    transcript: TranscriptAnalysis
    conversation: ConversationIntelligenceResult
    decision: DecisionResult


class IntelligenceOrchestrator:
    """Runs Milestone 2–4 analysis and deterministic policy selection per caller turn."""

    def __init__(
        self,
        transcript_analyzer: Optional[TranscriptAnalyzer] = None,
        conversation_analyzer: Optional[ConversationAnalyzer] = None,
        decision_manager: Optional[DecisionManager] = None,
    ) -> None:
        self.transcript_analyzer = transcript_analyzer or TranscriptAnalyzer()
        self.conversation_analyzer = conversation_analyzer or ConversationAnalyzer()
        self.decision_manager = decision_manager or DecisionManager()

    async def process_turn(
        self,
        stt_result: Dict[str, Any] | str,
        state: ConversationState,
        audio: bytes = b"",
        audio_quality: Optional[AudioQualityReport] = None,
        acoustic_features: Optional[AcousticFeatures] = None,
    ) -> TurnIntelligencePackage:
        # 1. Validate & normalize STT transcript
        tx_analysis = self.transcript_analyzer.analyze(
            stt_result=stt_result,
            audio_quality=audio_quality,
            selected_language=state.language,
        )

        # 2. Parallel emotional, intent, certainty, satisfaction & resolution assessment
        conv_result = await self.conversation_analyzer.analyze(
            transcript=tx_analysis.normalized_text or tx_analysis.raw_text,
            audio=audio,
            language=state.language,
            service_category=state.service_category,
            acoustic_features=acoustic_features,
            failed_attempts=state.failed_attempts,
            prior_resolution_status=state.resolution_status,
            prior_confirmed=state.customer_confirmed,
            prior_satisfaction=state.satisfaction_score,
            awaiting_resolution_confirmation=state.awaiting_resolution_confirmation,
            awaiting_feedback=state.awaiting_feedback,
            awaiting_order_id_digits=state.awaiting_order_id_digits,
        )

        # 3. Update per-call ConversationState
        state.update_from_turn(
            audio_quality=audio_quality,
            transcript=tx_analysis,
            intelligence=conv_result,
        )

        # 4. Deterministic Decision Engine selection
        decision = self.decision_manager.decide(
            state=state,
            needs_clarification=tx_analysis.needs_clarification,
            needs_confirmation=tx_analysis.needs_confirmation,
        )

        return TurnIntelligencePackage(
            transcript=tx_analysis,
            conversation=conv_result,
            decision=decision,
        )

