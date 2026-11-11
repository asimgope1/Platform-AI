"""
Audio Quality Analyzer for 8 kHz s16le telephony PCM streams.

Computes frame-level and utterance-level diagnostics:
  • SNR (approximate frame-percentile speech-to-noise ratio in dB)
  • RMS (int16 scale & dBFS)
  • Clipping percentage (saturation near ±32767)
  • Silence ratio
  • VAD probability (mean & peak)
  • Active speech duration (ms)
  • Overall normalized audio_quality_score in [0.0, 1.0]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from audio_intelligence.vad_processor import AdaptiveTelephonyVAD, BaseVADProcessor


@dataclass
class AudioQualityReport:
    sample_rate: int
    duration_ms: float
    rms: float
    rms_dbfs: float
    peak: float
    snr_db: float
    clipping_pct: float
    silence_ratio: float
    vad_probability: float
    peak_vad_probability: float
    speech_duration_ms: float
    audio_quality_score: float
    is_speech_present: bool
    quality_label: str  # "good" | "fair" | "poor" | "silent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioQualityAnalyzer:
    """Computes telephony audio quality diagnostics on raw s16le PCM."""

    def __init__(
        self,
        sample_rate: int = 8000,
        vad_processor: Optional[BaseVADProcessor] = None,
        silence_rms_threshold: float = 45.0,
        vad_threshold: float = 0.42,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad = vad_processor or AdaptiveTelephonyVAD(threshold=vad_threshold)
        self.silence_rms_threshold = silence_rms_threshold
        self.vad_threshold = vad_threshold

    def analyze(self, pcm: bytes) -> AudioQualityReport:
        """Analyze raw s16le mono PCM buffer and return structured quality metrics."""
        if not pcm or len(pcm) < 4:
            return AudioQualityReport(
                sample_rate=self.sample_rate,
                duration_ms=0.0,
                rms=0.0,
                rms_dbfs=-96.0,
                peak=0.0,
                snr_db=0.0,
                clipping_pct=0.0,
                silence_ratio=1.0,
                vad_probability=0.0,
                peak_vad_probability=0.0,
                speech_duration_ms=0.0,
                audio_quality_score=0.0,
                is_speech_present=False,
                quality_label="silent",
            )

        # Ensure even length for int16 parsing
        usable_len = len(pcm) - (len(pcm) % 2)
        samples_i16 = np.frombuffer(pcm[:usable_len], dtype="<i2").astype(np.float32)
        n_samples = len(samples_i16)
        duration_ms = (n_samples / self.sample_rate) * 1000.0

        # DC removal for energy calculations (keep raw peaks for clipping detection)
        peak = float(np.max(np.abs(samples_i16)))
        clipping_count = int(np.sum(np.abs(samples_i16) >= 32700.0))
        clipping_pct = (clipping_count / n_samples) * 100.0

        centered_i16 = samples_i16 - float(np.mean(samples_i16))
        rms = float(np.sqrt(np.mean(centered_i16 ** 2)))
        rms_dbfs = float(20.0 * np.log10(max(rms, 1.0) / 32768.0))

        # Split into 20 ms frames for silence ratio, SNR, and VAD
        frame_len = max(16, int(self.sample_rate * 0.020))
        frames_i16 = [
            centered_i16[i : i + frame_len]
            for i in range(0, n_samples, frame_len)
            if len(centered_i16[i : i + frame_len]) >= max(16, frame_len // 2)
        ]

        if not frames_i16:
            frames_i16 = [centered_i16]

        frame_rms = np.array(
            [float(np.sqrt(np.mean(f ** 2))) for f in frames_i16],
            dtype=np.float32,
        )
        silent_frames = int(np.sum(frame_rms < self.silence_rms_threshold))
        silence_ratio = float(silent_frames / len(frame_rms))

        # Approximate SNR using active vs background frame energy percentiles
        active_level = float(np.percentile(frame_rms, 90)) if len(frame_rms) >= 2 else rms
        noise_level = float(np.percentile(frame_rms, 15)) if len(frame_rms) >= 4 else float(np.min(frame_rms))
        # Apply minimum digital quantization floor (~12 int16 units) to avoid infinite SNR
        effective_noise = max(noise_level, 12.0)
        if active_level < self.silence_rms_threshold:
            snr_db = 0.0
        else:
            snr_db = float(np.clip(20.0 * np.log10(active_level / effective_noise), 0.0, 45.0))

        # VAD probabilities on float32 [-1, 1]
        samples_f32 = centered_i16 / 32768.0
        vad_probs = self.vad.utterance_probabilities(samples_f32, sample_rate=self.sample_rate, frame_ms=20)
        if vad_probs:
            vad_arr = np.array(vad_probs, dtype=np.float32)
            # Use top-quartile mean as utterance speech likelihood so silent edges don't dilute short words
            top_k = max(1, len(vad_arr) // 3)
            sorted_vad = np.sort(vad_arr)
            vad_probability = float(np.mean(sorted_vad[-top_k:]))
            peak_vad_probability = float(np.max(vad_arr))
            active_vad_frames = int(np.sum(vad_arr >= self.vad_threshold))
            frame_step_ms = duration_ms / len(vad_arr)
            speech_duration_ms = active_vad_frames * frame_step_ms
        else:
            vad_probability = 0.0
            peak_vad_probability = 0.0
            speech_duration_ms = 0.0

        is_speech_present = (
            rms >= self.silence_rms_threshold
            and peak_vad_probability >= self.vad_threshold
            and speech_duration_ms >= 60.0
        )

        # Composite quality score [0.0, 1.0]
        if not is_speech_present and rms < self.silence_rms_threshold:
            score = 0.0
            label = "silent"
        else:
            snr_norm = float(np.clip(snr_db / 25.0, 0.0, 1.0))
            # Soft voice penalty if active speech RMS < 180 int16; healthy telephony range 350-12000
            vol_norm = float(np.clip((active_level - 45.0) / 600.0, 0.0, 1.0))
            clip_penalty = float(np.clip(clipping_pct / 5.0, 0.0, 0.6))
            speech_presence = float(np.clip(vad_probability, 0.0, 1.0))

            score = float(
                np.clip(
                    (0.40 * snr_norm + 0.30 * speech_presence + 0.30 * vol_norm) - clip_penalty,
                    0.0,
                    1.0,
                )
            )
            if score >= 0.65:
                label = "good"
            elif score >= 0.35:
                label = "fair"
            else:
                label = "poor"

        return AudioQualityReport(
            sample_rate=self.sample_rate,
            duration_ms=round(duration_ms, 2),
            rms=round(rms, 2),
            rms_dbfs=round(rms_dbfs, 2),
            peak=round(peak, 2),
            snr_db=round(snr_db, 2),
            clipping_pct=round(clipping_pct, 3),
            silence_ratio=round(silence_ratio, 3),
            vad_probability=round(vad_probability, 3),
            peak_vad_probability=round(peak_vad_probability, 3),
            speech_duration_ms=round(speech_duration_ms, 2),
            audio_quality_score=round(score, 3),
            is_speech_present=bool(is_speech_present),
            quality_label=label,
        )

