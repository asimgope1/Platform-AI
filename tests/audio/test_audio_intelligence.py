"""
Unit & Integration Tests for Milestone 1: Audio Intelligence Layer.

Tests quality analysis, VAD, noise suppression, echo cancellation,
speech segmentation, acoustic feature extraction, pipeline bypass guards,
and non-disruptive IVRCallSession integration.
"""
from __future__ import annotations

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock

from audio_intelligence import (
    AcousticFeatureExtractor,
    AdaptiveTelephonyVAD,
    AudioIntelligencePipeline,
    AudioQualityAnalyzer,
    PlaybackEchoCanceller,
    SileroVADProcessor,
    SpectralNoiseSuppressor,
    SpeechSegmenter,
    create_noise_suppressor,
    create_vad_processor,
)
from telephony.ivr_menu import IVRState
from telephony.ivr_session import IVRCallSession


SAMPLE_RATE = 8000


def _make_speech_like_pcm(
    duration_s: float = 1.2,
    amplitude: float = 7000.0,
    f0_hz: float = 165.0,
    leading_silence_s: float = 0.0,
    trailing_silence_s: float = 0.0,
) -> bytes:
    """Generate harmonic vowel-like telephony waveform with amplitude modulation."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    # Harmonic stack (F0 + 2*F0 + 3*F0) modulated at ~4 Hz syllable envelope
    envelope = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.0 * t)
    wave = (
        0.60 * np.sin(2.0 * np.pi * f0_hz * t)
        + 0.28 * np.sin(2.0 * np.pi * (2.0 * f0_hz) * t)
        + 0.12 * np.sin(2.0 * np.pi * (3.0 * f0_hz) * t)
    ) * envelope * amplitude

    lead = np.zeros(int(SAMPLE_RATE * leading_silence_s), dtype=np.float32)
    trail = np.zeros(int(SAMPLE_RATE * trailing_silence_s), dtype=np.float32)
    combined = np.concatenate([lead, wave, trail])
    return np.clip(combined, -32767.0, 32767.0).astype("<i2").tobytes()


def _make_noise_pcm(duration_s: float = 1.2, std_amp: float = 500.0, seed: int = 42) -> bytes:
    """Generate stationary broadband fan/hiss noise."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, std_amp, size=int(SAMPLE_RATE * duration_s))
    return np.clip(samples, -32767.0, 32767.0).astype("<i2").tobytes()


def _mix_pcm(pcm_a: bytes, pcm_b: bytes) -> bytes:
    arr_a = np.frombuffer(pcm_a, dtype="<i2").astype(np.float32)
    arr_b = np.frombuffer(pcm_b, dtype="<i2").astype(np.float32)
    n = min(len(arr_a), len(arr_b))
    mixed = np.clip(arr_a[:n] + arr_b[:n], -32767.0, 32767.0)
    return mixed.astype("<i2").tobytes()


class TestAudioQualityAndVAD:
    def test_clean_speech_scores_good(self):
        pcm = _make_speech_like_pcm(duration_s=1.0, amplitude=8000.0, leading_silence_s=0.3)
        report = AudioQualityAnalyzer().analyze(pcm)

        assert report.is_speech_present is True
        assert report.vad_probability >= 0.60
        assert report.snr_db >= 15.0
        assert report.clipping_pct == 0.0
        assert report.audio_quality_score >= 0.65
        assert report.quality_label == "good"

    def test_soft_voice_detected_above_silence_gate(self):
        pcm = _make_speech_like_pcm(duration_s=1.0, amplitude=350.0, leading_silence_s=0.2)
        report = AudioQualityAnalyzer().analyze(pcm)

        assert report.is_speech_present is True
        assert report.rms > 45.0
        assert report.speech_duration_ms > 200.0

    def test_digital_silence_rejected(self):
        pcm = b"\x00\x00" * (SAMPLE_RATE * 1)
        report = AudioQualityAnalyzer().analyze(pcm)

        assert report.is_speech_present is False
        assert report.quality_label == "silent"
        assert report.audio_quality_score == 0.0

    def test_stationary_fan_noise_rejected_by_vad(self):
        noise_pcm = _make_noise_pcm(duration_s=1.2, std_amp=450.0)
        report = AudioQualityAnalyzer().analyze(noise_pcm)

        assert report.is_speech_present is False
        assert report.peak_vad_probability < 0.45

    def test_clipped_audio_penalized(self):
        # Saturate heavily above int16 limits
        pcm = _make_speech_like_pcm(duration_s=1.0, amplitude=65000.0)
        report = AudioQualityAnalyzer().analyze(pcm)

        assert report.clipping_pct > 5.0

    def test_silero_factory_falls_back_cleanly(self):
        vad = create_vad_processor(provider="silero", threshold=0.45)
        assert isinstance(vad, SileroVADProcessor)
        # Should work seamlessly whether Silero weights exist or Adaptive fallback is active
        speech_f32 = np.frombuffer(_make_speech_like_pcm(duration_s=0.5), dtype="<i2").astype(np.float32) / 32768.0
        probs = vad.utterance_probabilities(speech_f32, sample_rate=SAMPLE_RATE)
        assert len(probs) > 0
        assert max(probs) > 0.5


class TestNoiseSuppressionAndAEC:
    def test_spectral_suppressor_reduces_noise_and_improves_snr(self):
        speech = _make_speech_like_pcm(duration_s=1.4, amplitude=5000.0, leading_silence_s=0.4, trailing_silence_s=0.4)
        noise = _make_noise_pcm(duration_s=2.2, std_amp=650.0)
        noisy = _mix_pcm(speech, noise)

        analyzer = AudioQualityAnalyzer()
        before = analyzer.analyze(noisy)

        suppressor = create_noise_suppressor("spectral")
        denoised = suppressor.suppress(noisy, sample_rate=SAMPLE_RATE, snr_db=before.snr_db)
        after = analyzer.analyze(denoised)

        assert len(denoised) == len(noisy)
        assert after.snr_db >= before.snr_db
        assert after.is_speech_present is True

    def test_clean_high_snr_speech_bypasses_suppression(self):
        clean = _make_speech_like_pcm(duration_s=1.0, amplitude=8000.0, leading_silence_s=0.3)
        suppressor = SpectralNoiseSuppressor(clean_bypass_snr_db=20.0)
        out = suppressor.suppress(clean, sample_rate=SAMPLE_RATE, snr_db=28.0)
        assert out == clean

    def test_echo_canceller_attenuates_correlated_playback(self):
        playback_ref = _make_speech_like_pcm(duration_s=1.0, amplitude=6000.0, f0_hz=210.0)
        # Delayed echo bleed (40ms = 320 samples) + near-end silence
        ref_arr = np.frombuffer(playback_ref, dtype="<i2").astype(np.float32)
        echo_arr = np.zeros_like(ref_arr)
        delay = 320
        echo_arr[delay:] = 0.65 * ref_arr[:-delay]
        mic_pcm = np.clip(echo_arr, -32767.0, 32767.0).astype("<i2").tobytes()

        aec = PlaybackEchoCanceller(attenuation_strength=0.80)
        cleaned = aec.process(mic_pcm, reference_pcm=playback_ref, sample_rate=SAMPLE_RATE)

        mic_rms = float(np.sqrt(np.mean(np.frombuffer(mic_pcm, dtype="<i2").astype(np.float32) ** 2)))
        clean_rms = float(np.sqrt(np.mean(np.frombuffer(cleaned, dtype="<i2").astype(np.float32) ** 2)))
        assert clean_rms < mic_rms * 0.5


class TestSegmentationAndAcousticFeatures:
    def test_speech_segmenter_trims_outer_silence_with_preroll(self):
        pcm = _make_speech_like_pcm(
            duration_s=0.8,
            amplitude=7000.0,
            leading_silence_s=0.8,
            trailing_silence_s=0.8,
        )
        segmenter = SpeechSegmenter(sample_rate=SAMPLE_RATE, preroll_ms=120, hangover_ms=180)
        seg = segmenter.segment(pcm)

        assert seg.was_trimmed is True
        assert len(seg.pcm) < len(pcm)
        # Must preserve speech plus preroll/hangover padding (> 0.8s)
        assert len(seg.pcm) >= int(SAMPLE_RATE * 0.8 * 2)

    def test_acoustic_feature_extractor_detects_pitch_and_rms(self):
        pcm = _make_speech_like_pcm(duration_s=1.0, amplitude=7500.0, f0_hz=160.0)
        feats = AcousticFeatureExtractor(sample_rate=SAMPLE_RATE).extract(pcm)

        assert feats.rms_mean > 0.05
        assert feats.voiced_frame_ratio > 0.5
        assert 120.0 <= feats.pitch_f0_mean_hz <= 200.0
        assert feats.spectral_centroid_hz > 100.0


class TestAudioPipelineAndIVRIntegration:
    def test_pipeline_preserves_original_and_supports_bypass(self):
        speech = _make_speech_like_pcm(duration_s=1.0, amplitude=5000.0, leading_silence_s=0.3)
        noise = _make_noise_pcm(duration_s=1.3, std_amp=500.0)
        noisy = _mix_pcm(speech, noise)

        pipeline_enabled = AudioIntelligencePipeline(enabled=True, noise_suppression_enabled=True)
        res_enh = pipeline_enabled.enhance_sync(noisy)
        assert res_enh.original_pcm == noisy
        assert len(res_enh.enhanced_pcm) == len(noisy)
        assert res_enh.bypassed is False

        pipeline_bypassed = AudioIntelligencePipeline(enabled=False)
        res_byp = pipeline_bypassed.enhance_sync(noisy)
        assert res_byp.bypassed is True
        assert res_byp.enhanced_pcm == noisy
        assert res_byp.original_pcm == noisy

    @pytest.mark.asyncio
    async def test_ivr_session_runs_audio_intelligence_without_breaking_pipeline(self):
        mock_ws = AsyncMock()
        mock_audio = AsyncMock()
        mock_audio.transcribe = AsyncMock(return_value={"text": "UPI refund status"})
        mock_orch = AsyncMock()

        session = IVRCallSession(
            ws=mock_ws,
            orchestrator=mock_orch,
            audio_service=mock_audio,
        )
        session.menu_machine.state = IVRState.AI_CONVERSATION
        session._run_ai_and_reply = AsyncMock()

        # 1. Weak non-speech noise must be filtered before calling Whisper STT
        weak_noise = _make_noise_pcm(duration_s=0.6, std_amp=60.0)
        await session._run_stt_ai_tts(weak_noise)
        mock_audio.transcribe.assert_not_called()
        assert session.last_audio_quality is not None
        assert session.last_audio_quality.is_speech_present is False

        # 2. Valid speech flows through Audio Intelligence -> STT -> AI reply
        valid_speech = _make_speech_like_pcm(duration_s=1.0, amplitude=7000.0, leading_silence_s=0.2)
        await session._run_stt_ai_tts(valid_speech)
        mock_audio.transcribe.assert_awaited_once()
        session._run_ai_and_reply.assert_awaited_once_with("UPI refund status")
        assert session.last_audio_quality.is_speech_present is True
        assert session.last_enhanced_audio is not None
        assert session.last_enhanced_audio.original_pcm == valid_speech

