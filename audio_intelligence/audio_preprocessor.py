"""
Audio Intelligence Preprocessor & Orchestration Pipeline (Milestone 1).

Coordinates:
  1. Audio quality & SNR diagnostics (AudioQualityAnalyzer)
  2. Optional playback echo cancellation (PlaybackEchoCanceller)
  3. Pluggable noise suppression with clean-speech bypass & safety guard
  4. Speech boundary segmentation with pre-roll & hangover protection
  5. Acoustic prosodic/spectral feature extraction

Always preserves `original_pcm` so raw and enhanced streams can be compared
or rolled back without losing caller audio.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from audio_intelligence.audio_emotion import AcousticFeatureExtractor, AcousticFeatures
from audio_intelligence.audio_quality import AudioQualityAnalyzer, AudioQualityReport
from audio_intelligence.echo_canceller import PlaybackEchoCanceller
from audio_intelligence.noise_suppressor import BaseNoiseSuppressor, create_noise_suppressor
from audio_intelligence.speech_segmenter import SpeechSegmenter
from audio_intelligence.vad_processor import BaseVADProcessor, create_vad_processor

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAudioResult:
    original_pcm: bytes
    enhanced_pcm: bytes
    segmented_pcm: bytes
    quality: AudioQualityReport
    acoustic_features: AcousticFeatures
    bypassed: bool = False
    enhancement_applied: List[str] = field(default_factory=list)
    processing_ms: float = 0.0


class AudioIntelligencePipeline:
    """Modular Milestone 1 Audio Intelligence pipeline for telephony PCM."""

    def __init__(
        self,
        sample_rate: int = 8000,
        enabled: Optional[bool] = None,
        vad_provider: Optional[str] = None,
        vad_threshold: Optional[float] = None,
        noise_suppression_enabled: Optional[bool] = None,
        noise_suppression_provider: Optional[str] = None,
        echo_cancellation_enabled: Optional[bool] = None,
        vad: Optional[BaseVADProcessor] = None,
        suppressor: Optional[BaseNoiseSuppressor] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("AUDIO_INTELLIGENCE_ENABLED", "true").lower() in ("1", "true", "yes")
        )
        self.noise_suppression_enabled = (
            noise_suppression_enabled
            if noise_suppression_enabled is not None
            else os.getenv("NOISE_SUPPRESSION_ENABLED", "true").lower() in ("1", "true", "yes")
        )
        self.echo_cancellation_enabled = (
            echo_cancellation_enabled
            if echo_cancellation_enabled is not None
            else os.getenv("ECHO_CANCELLATION_ENABLED", "false").lower() in ("1", "true", "yes")
        )

        chosen_vad_provider = vad_provider or os.getenv("VAD_PROVIDER", "adaptive")
        chosen_vad_threshold = (
            vad_threshold
            if vad_threshold is not None
            else float(os.getenv("VAD_THRESHOLD", "0.42"))
        )
        chosen_ns_provider = noise_suppression_provider or os.getenv("NOISE_SUPPRESSION_PROVIDER", "spectral")

        self.vad = vad or create_vad_processor(provider=chosen_vad_provider, threshold=chosen_vad_threshold)
        self.suppressor = suppressor or create_noise_suppressor(provider=chosen_ns_provider)
        self.echo_canceller = PlaybackEchoCanceller()
        self.quality_analyzer = AudioQualityAnalyzer(
            sample_rate=sample_rate,
            vad_processor=self.vad,
            vad_threshold=chosen_vad_threshold,
        )
        self.segmenter = SpeechSegmenter(
            sample_rate=sample_rate,
            vad_processor=self.vad,
            vad_threshold=chosen_vad_threshold,
        )
        self.feature_extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

    def analyze_sync(self, pcm: bytes) -> AudioQualityReport:
        """Synchronously inspect audio quality, SNR, clipping, and VAD speech duration."""
        return self.quality_analyzer.analyze(pcm)

    async def analyze(self, pcm: bytes) -> AudioQualityReport:
        """Async wrapper executing audio quality inspection off the event loop."""
        return await asyncio.to_thread(self.analyze_sync, pcm)

    def enhance_sync(
        self,
        pcm: bytes,
        quality: Optional[AudioQualityReport] = None,
        reference_pcm: Optional[bytes] = None,
    ) -> EnhancedAudioResult:
        """Run optional AEC, noise suppression, segmentation, and acoustic extraction."""
        t0 = time.perf_counter()
        report = quality or self.analyze_sync(pcm)
        features = self.feature_extractor.extract(pcm)

        if not self.enabled or not pcm or len(pcm) < 64:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return EnhancedAudioResult(
                original_pcm=pcm,
                enhanced_pcm=pcm,
                segmented_pcm=pcm,
                quality=report,
                acoustic_features=features,
                bypassed=True,
                enhancement_applied=[],
                processing_ms=round(elapsed, 2),
            )

        current_pcm = pcm
        applied: List[str] = []

        # 1. Optional acoustic echo cancellation
        if self.echo_cancellation_enabled and reference_pcm:
            aec_pcm = self.echo_canceller.process(
                current_pcm,
                reference_pcm=reference_pcm,
                sample_rate=self.sample_rate,
            )
            if aec_pcm != current_pcm:
                current_pcm = aec_pcm
                applied.append("aec")

        # 2. Noise suppression (skipped automatically when clean or disabled)
        bypassed_ns = False
        if self.noise_suppression_enabled and report.clipping_pct < 8.0:
            denoised = self.suppressor.suppress(
                current_pcm,
                sample_rate=self.sample_rate,
                snr_db=report.snr_db,
            )
            if denoised != current_pcm and not _did_over_attenuate(current_pcm, denoised):
                current_pcm = denoised
                applied.append(f"noise_suppression:{self.suppressor.name}")
            elif denoised != current_pcm:
                logger.warning("[AudioIntelligence] Noise suppressor over-attenuated speech; reverting to original PCM.")
                bypassed_ns = True
        else:
            bypassed_ns = not self.noise_suppression_enabled

        # 3. Speech boundary segmentation (preserve pre-roll & hangover)
        segmented = self.segmenter.segment(current_pcm)
        if segmented.was_trimmed:
            applied.append("speech_segmentation")

        elapsed = (time.perf_counter() - t0) * 1000.0
        return EnhancedAudioResult(
            original_pcm=pcm,
            enhanced_pcm=current_pcm,
            segmented_pcm=segmented.pcm,
            quality=report,
            acoustic_features=features,
            bypassed=bypassed_ns and len(applied) == 0,
            enhancement_applied=applied,
            processing_ms=round(elapsed, 2),
        )

    async def enhance(
        self,
        pcm: bytes,
        quality: Optional[AudioQualityReport] = None,
        reference_pcm: Optional[bytes] = None,
    ) -> EnhancedAudioResult:
        """Async wrapper executing audio enhancement off the event loop."""
        return await asyncio.to_thread(self.enhance_sync, pcm, quality, reference_pcm)


def _did_over_attenuate(original_pcm: bytes, enhanced_pcm: bytes) -> bool:
    """Safety guard: ensure denoising never wipes out >75% of active peak/RMS speech."""
    if not original_pcm or not enhanced_pcm or len(enhanced_pcm) < 4:
        return True
    orig = np.frombuffer(original_pcm[: len(original_pcm) - (len(original_pcm) % 2)], dtype="<i2").astype(np.float32)
    enh = np.frombuffer(enhanced_pcm[: len(enhanced_pcm) - (len(enhanced_pcm) % 2)], dtype="<i2").astype(np.float32)
    orig_p90 = float(np.percentile(np.abs(orig), 90))
    enh_p90 = float(np.percentile(np.abs(enh), 90))
    if orig_p90 > 60.0 and enh_p90 < orig_p90 * 0.25:
        return True
    return False

