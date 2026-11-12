"""
Conversation Analyzer (Milestone 3 Orchestrator).

Runs parallel async intelligence assessments via asyncio.gather():
  • SpeechBrain / acoustic voice emotion
  • Multilingual text emotion
  • Intent & escalation rule detection
  • Caller certainty evaluation
  • Satisfaction & resolution tracking
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from audio_intelligence.audio_emotion import AcousticFeatures
from conversation_intelligence.caller_confidence import CallerConfidenceAnalyzer, CallerConfidenceResult
from conversation_intelligence.intent_analyzer import IntentAnalyzer, IntentResult
from conversation_intelligence.resolution_tracker import ResolutionResult, ResolutionTracker
from conversation_intelligence.satisfaction_analyzer import SatisfactionAnalyzer, SatisfactionResult
from conversation_intelligence.text_emotion import TextEmotionAnalyzer, TextEmotionResult
from providers.speech_emotion.speechbrain_emotion import SpeechBrainEmotionProvider, VoiceEmotionResult


@dataclass
class ConversationIntelligenceResult:
    intent: IntentResult
    text_emotion: TextEmotionResult
    voice_emotion: VoiceEmotionResult
    caller_confidence: CallerConfidenceResult
    satisfaction: SatisfactionResult
    resolution: ResolutionResult
    fused_emotion: str
    fused_emotion_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "text_emotion": self.text_emotion.to_dict(),
            "voice_emotion": self.voice_emotion.to_dict(),
            "caller_confidence": self.caller_confidence.to_dict(),
            "satisfaction": self.satisfaction.to_dict(),
            "resolution": self.resolution.to_dict(),
            "fused_emotion": self.fused_emotion,
            "fused_emotion_confidence": self.fused_emotion_confidence,
        }


class ConversationAnalyzer:
    """Coordinates parallel acoustic and conversational intelligence analysis."""

    def __init__(
        self,
        voice_emotion_provider: Optional[SpeechBrainEmotionProvider] = None,
        text_emotion_analyzer: Optional[TextEmotionAnalyzer] = None,
        intent_analyzer: Optional[IntentAnalyzer] = None,
        caller_confidence_analyzer: Optional[CallerConfidenceAnalyzer] = None,
        satisfaction_analyzer: Optional[SatisfactionAnalyzer] = None,
        resolution_tracker: Optional[ResolutionTracker] = None,
    ) -> None:
        self.voice_emotion_provider = voice_emotion_provider or SpeechBrainEmotionProvider()
        self.text_emotion_analyzer = text_emotion_analyzer or TextEmotionAnalyzer()
        self.intent_analyzer = intent_analyzer or IntentAnalyzer()
        self.caller_confidence_analyzer = caller_confidence_analyzer or CallerConfidenceAnalyzer()
        self.satisfaction_analyzer = satisfaction_analyzer or SatisfactionAnalyzer()
        self.resolution_tracker = resolution_tracker or ResolutionTracker()

    async def analyze(
        self,
        transcript: str,
        audio: bytes = b"",
        language: str = "en",
        service_category: Optional[str] = None,
        acoustic_features: Optional[AcousticFeatures] = None,
        failed_attempts: int = 0,
        prior_resolution_status: str = "unresolved",
        prior_confirmed: bool = False,
        prior_satisfaction: Optional[float] = None,
        awaiting_resolution_confirmation: bool = False,
        awaiting_feedback: bool = False,
        awaiting_order_id_digits: bool = False,
    ) -> ConversationIntelligenceResult:
        voice_task = asyncio.to_thread(
            self.voice_emotion_provider.analyze,
            audio,
            acoustic_features,
            language,
        )
        intent_task = asyncio.to_thread(
            self.intent_analyzer.analyze,
            transcript,
            service_category,
            awaiting_feedback,
            awaiting_order_id_digits,
        )
        certainty_task = asyncio.to_thread(
            self.caller_confidence_analyzer.analyze,
            transcript,
        )

        voice_res, intent_res, certainty_res = await asyncio.gather(
            voice_task,
            intent_task,
            certainty_task,
        )

        # Text emotion uses voice hint to guard against sarcastic positive phrasing
        text_res = await asyncio.to_thread(
            self.text_emotion_analyzer.analyze,
            transcript,
            failed_attempts,
            voice_res.primary_emotion,
        )

        resolution_res = self.resolution_tracker.evaluate(
            text=transcript,
            intent=intent_res.intent,
            prior_status=prior_resolution_status,
            awaiting_resolution_confirmation=awaiting_resolution_confirmation,
            prior_confirmed=prior_confirmed,
        )

        satisfaction_res = self.satisfaction_analyzer.analyze(
            text_emotion=text_res.primary_emotion,
            resolution_status=resolution_res.resolution_status,
            failed_attempts=failed_attempts,
            explicit_rating=intent_res.rating_value,
            customer_confirmed=resolution_res.customer_confirmed,
            prior_score=prior_satisfaction,
        )

        # Weighted fusion: trust text emotion higher on non-English calls where IEMOCAP/prosody is experimental
        voice_weight = 0.22 if voice_res.experimental_signal else 0.35
        text_weight = 1.0 - voice_weight
        if text_res.primary_emotion == voice_res.primary_emotion:
            fused_emotion = text_res.primary_emotion
            fused_conf = min(1.0, round(text_res.confidence * text_weight + voice_res.confidence * voice_weight + 0.10, 3))
        elif text_res.primary_emotion != "neutral" or text_res.confidence >= voice_res.confidence:
            fused_emotion = text_res.primary_emotion
            fused_conf = round(text_res.confidence * text_weight, 3)
        else:
            fused_emotion = voice_res.primary_emotion
            fused_conf = round(voice_res.confidence * voice_weight, 3)

        return ConversationIntelligenceResult(
            intent=intent_res,
            text_emotion=text_res,
            voice_emotion=voice_res,
            caller_confidence=certainty_res,
            satisfaction=satisfaction_res,
            resolution=resolution_res,
            fused_emotion=fused_emotion,
            fused_emotion_confidence=fused_conf,
        )

