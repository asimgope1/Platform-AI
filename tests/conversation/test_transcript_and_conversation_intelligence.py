"""
Unit tests for Milestone 2 (Transcript Intelligence) & Milestone 3 (Conversation Intelligence).
"""
from __future__ import annotations

import numpy as np
import pytest

from audio_intelligence.audio_quality import AudioQualityReport
from conversation_intelligence.caller_confidence import CallerConfidenceAnalyzer
from conversation_intelligence.conversation_analyzer import ConversationAnalyzer
from conversation_intelligence.conversation_memory import ConversationMemory
from conversation_intelligence.intent_analyzer import IntentAnalyzer
from conversation_intelligence.resolution_tracker import ResolutionTracker
from conversation_intelligence.satisfaction_analyzer import SatisfactionAnalyzer
from conversation_intelligence.text_emotion import TextEmotionAnalyzer
from conversation_intelligence.transcript_confidence import TranscriptAnalyzer, mask_sensitive_text
from providers.speech_emotion.speechbrain_emotion import SpeechBrainEmotionProvider


class TestTranscriptIntelligence:
    def test_banking_vocabulary_normalization_and_amounts(self):
        analyzer = TranscriptAnalyzer()
        result = analyzer.analyze(
            {
                "text": "my you be i payment of rs 2500 failed during vedio k y c",
                "diagnostics": {"avg_logprob": -0.25, "no_speech_prob": 0.05, "compression_ratio": 1.1},
            },
            audio_quality=AudioQualityReport(
                sample_rate=8000,
                duration_ms=2500.0,
                rms=1400.0,
                rms_dbfs=-27.4,
                peak=5000.0,
                snr_db=22.0,
                clipping_pct=0.0,
                silence_ratio=0.15,
                vad_probability=0.92,
                peak_vad_probability=0.98,
                speech_duration_ms=2100.0,
                audio_quality_score=0.88,
                is_speech_present=True,
                quality_label="good",
            ),
            selected_language="en",
        )
        assert "UPI" in result.normalized_text
        assert "Video KYC" in result.normalized_text
        assert result.detected_amounts == ["2500"]
        assert result.transcript_confidence >= 0.70
        assert not result.needs_clarification

    def test_pii_masking_hides_account_numbers_and_upi_ids(self):
        masked = mask_sensitive_text(
            "My account 987654321012 and UPI id ramesh.kumar@okicici failed for 1500"
        )
        assert "987654321012" not in masked
        assert "ramesh.kumar@okicici" not in masked
        assert "[REDACTED_12DIGITS]" in masked
        assert "[REDACTED_UPI_ID]" in masked

    def test_low_logprob_and_bad_audio_trigger_clarify(self):
        analyzer = TranscriptAnalyzer()
        result = analyzer.analyze(
            {
                "text": "uh hmm",
                "diagnostics": {
                    "avg_logprob": -1.45,
                    "no_speech_prob": 0.78,
                    "compression_ratio": 2.8,
                },
            },
            audio_quality=AudioQualityReport(
                sample_rate=8000,
                duration_ms=1000.0,
                rms=60.0,
                rms_dbfs=-54.7,
                peak=200.0,
                snr_db=2.0,
                clipping_pct=0.0,
                silence_ratio=0.85,
                vad_probability=0.15,
                peak_vad_probability=0.22,
                speech_duration_ms=120.0,
                audio_quality_score=0.18,
                is_speech_present=False,
                quality_label="poor",
            ),
            selected_language="en",
        )
        assert result.transcript_confidence < 0.45
        assert result.needs_clarification is True
        assert result.is_meaningful is False

    def test_hindi_language_mismatch_penalizes_english_output(self):
        analyzer = TranscriptAnalyzer()
        result = analyzer.analyze(
            {"text": "hello check my balance", "diagnostics": {"avg_logprob": -0.3}},
            selected_language="hi",
        )
        assert result.language_match is False
        assert "script_mismatch_hi" in result.issues


class TestConversationIntelligenceModules:
    def test_intent_explicit_agent_and_fraud_escalation(self):
        ia = IntentAnalyzer()
        res_agent = ia.analyze("Please connect me to a human representative now")
        assert res_agent.explicit_agent_request is True
        assert res_agent.intent == "human_agent"

        res_fraud = ia.analyze("My card was stolen and unauthorized fraud transaction happened!")
        assert res_fraud.requires_mandatory_escalation is True
        assert res_fraud.intent == "fraud_escalation"

        res_hi = ia.analyze("मुझे अपने एजेंट से बात करनी है")
        assert res_hi.explicit_agent_request is True

        res_or = ia.analyze("ମୋତେ ଜଣେ ପ୍ରତିନିଧିଙ୍କ ସହିତ କଥା ହେବାର ଅଛି")
        assert res_or.explicit_agent_request is True

    def test_text_emotion_avoids_sarcasm_false_positive(self):
        tea = TextEmotionAnalyzer()
        # Surface "I am fine thank you" but repeated failure context + angry voice hint -> frustrated
        res = tea.analyze(
            "Great, thank you for nothing, same thing not working",
            failed_attempts=2,
            voice_emotion_hint="angry",
        )
        assert res.primary_emotion in ("frustrated", "angry")

    def test_caller_certainty_strictly_separate_from_emotion(self):
        cca = CallerConfidenceAnalyzer()
        uncertain = cca.analyze("I think maybe Rs 2000 was debited, not sure")
        certain = cca.analyze("Definitely Rs 2000 was debited, 100% sure")
        assert uncertain.caller_certainty < 0.55
        assert uncertain.certainty_label == "uncertain"
        assert certain.caller_certainty > 0.75
        assert certain.certainty_label == "certain"

    def test_conversation_memory_ema_smoothing_and_trend(self):
        mem = ConversationMemory(maxlen=10)
        t1 = mem.record_turn(1, "upi_refund", "neutral", 0.55, "unresolved")
        t2 = mem.record_turn(2, "upi_refund", "frustrated", 0.40, "unresolved")
        t3 = mem.record_turn(3, "upi_refund", "angry", 0.25, "unresolved")
        assert t1 == "stable"
        assert t3 == "worsening"
        assert mem.consecutive_same_intent >= 2
        assert -1.0 <= mem.ema_valence < -0.4

    def test_resolution_requires_confirmation_not_just_politeness(self):
        rt = ResolutionTracker()
        r1 = rt.evaluate("Thank you", intent="appreciation", prior_status="unresolved")
        assert r1.resolution_status == "potentially_resolved"
        assert r1.customer_confirmed is False

        r2 = rt.evaluate(
            "Yes that resolved it",
            intent="confirmation_yes",
            prior_status="potentially_resolved",
            awaiting_resolution_confirmation=True,
        )
        assert r2.resolution_status == "resolved"
        assert r2.customer_confirmed is True

    def test_voice_emotion_provider_acoustic_fallback_is_experimental_and_non_blocking(self):
        provider = SpeechBrainEmotionProvider()
        # High energy simulated speech (amplitude ~9000 at 8 kHz)
        sr = 8000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        pcm = (np.sin(2 * np.pi * 240 * t) * 9000).astype("<i2").tobytes()
        res_en = provider.analyze(pcm, language="en")
        res_hi = provider.analyze(pcm, language="hi")
        assert res_en.provider == "acoustic_prosody"
        assert res_en.primary_emotion in ("neutral", "concerned", "frustrated", "angry")
        assert res_hi.experimental_signal is True

    @pytest.mark.asyncio
    async def test_parallel_conversation_analyzer_fusion(self):
        ca = ConversationAnalyzer()
        sat = SatisfactionAnalyzer()
        res = await ca.analyze(
            transcript="My UPI payment failed again, I am very frustrated!",
            language="en",
            failed_attempts=2,
        )
        assert res.intent.intent in ("upi_payment_failed", "upi_refund", "complaint")
        assert res.fused_emotion in ("frustrated", "angry")
        assert res.satisfaction.satisfaction_score is not None
        assert res.satisfaction.satisfaction_score < 0.5
        explicit_sat = sat.analyze("satisfied", "resolved", explicit_rating=5)
        assert explicit_sat.satisfaction_score == 1.0
        assert explicit_sat.explicit_rating == 5
