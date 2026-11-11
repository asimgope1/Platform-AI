"""
Acoustic Feature Extractor for Telephony Audio (Zero external librosa/openSMILE requirement).

Extracts prosodic and spectral features from 8 kHz s16le PCM:
  • RMS energy (mean / std)
  • Zero-crossing rate (ZCR mean / std)
  • Pitch F0 estimate via autocorrelation (mean / std / voiced ratio)
  • Spectral centroid & spectral flatness
  • Envelope-peak speech rate proxy & pause ratio

These acoustic cues supplement Milestone 3 emotion analysis and audio diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class AcousticFeatures:
    rms_mean: float
    rms_std: float
    zcr_mean: float
    zcr_std: float
    pitch_f0_mean_hz: float
    pitch_f0_std_hz: float
    voiced_frame_ratio: float
    spectral_centroid_hz: float
    spectral_flatness: float
    speech_rate_proxy: float
    pause_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AcousticFeatureExtractor:
    """Fast NumPy-based acoustic feature extractor for telephony PCM."""

    def __init__(self, sample_rate: int = 8000, frame_ms: int = 25) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

    def extract(self, pcm: bytes) -> AcousticFeatures:
        """Compute prosodic and spectral features from raw s16le mono PCM."""
        if not pcm or len(pcm) < 64:
            return AcousticFeatures(
                rms_mean=0.0,
                rms_std=0.0,
                zcr_mean=0.0,
                zcr_std=0.0,
                pitch_f0_mean_hz=0.0,
                pitch_f0_std_hz=0.0,
                voiced_frame_ratio=0.0,
                spectral_centroid_hz=0.0,
                spectral_flatness=0.0,
                speech_rate_proxy=0.0,
                pause_ratio=1.0,
            )

        usable_len = len(pcm) - (len(pcm) % 2)
        samples = np.frombuffer(pcm[:usable_len], dtype="<i2").astype(np.float32) / 32768.0
        samples = samples - float(np.mean(samples))

        frame_len = max(64, int(self.sample_rate * self.frame_ms / 1000))
        hop_len = max(32, frame_len // 2)

        rms_vals: List[float] = []
        zcr_vals: List[float] = []
        f0_vals: List[float] = []
        centroids: List[float] = []
        flatness_vals: List[float] = []

        # Autocorrelation pitch search bounds (75 Hz to 380 Hz human speech)
        min_lag = max(2, int(self.sample_rate / 380.0))
        max_lag = min(frame_len - 1, int(self.sample_rate / 75.0))

        for start in range(0, len(samples) - frame_len + 1, hop_len):
            frame = samples[start : start + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_vals.append(rms)

            signs = np.signbit(frame)
            zcr = float(np.mean(signs[:-1] != signs[1:]))
            zcr_vals.append(zcr)

            if rms > 0.005:
                # Spectral centroid & flatness
                win = frame * np.hanning(len(frame))
                spec = np.abs(np.fft.rfft(win)) + 1e-12
                freqs = np.fft.rfftfreq(len(win), d=1.0 / self.sample_rate)
                mag_sum = float(np.sum(spec))
                centroids.append(float(np.sum(freqs * spec) / mag_sum))

                power = spec ** 2
                geo_mean = float(np.exp(np.mean(np.log(power))))
                arith_mean = float(np.mean(power))
                flatness_vals.append(float(np.clip(geo_mean / (arith_mean + 1e-12), 0.0, 1.0)))

                # Autocorrelation F0 estimation
                if max_lag > min_lag:
                    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
                    ac0 = float(ac[0]) + 1e-12
                    segment = ac[min_lag:max_lag]
                    if len(segment) > 0:
                        best_idx = int(np.argmax(segment))
                        peak_norm = float(segment[best_idx]) / ac0
                        if peak_norm >= 0.40:
                            lag = min_lag + best_idx
                            f0_vals.append(float(self.sample_rate / lag))

        if not rms_vals:
            rms_vals = [float(np.sqrt(np.mean(samples ** 2)))]
            zcr_vals = [0.0]

        rms_arr = np.array(rms_vals, dtype=np.float32)
        pause_threshold = max(0.002, float(np.max(rms_arr)) * 0.12)
        pause_ratio = float(np.mean(rms_arr < pause_threshold))

        # Envelope peak count as lightweight syllable / speech-rate proxy (peaks/sec)
        duration_sec = len(samples) / float(self.sample_rate)
        speech_rate_proxy = 0.0
        if len(rms_arr) >= 5 and duration_sec > 0.05:
            above = rms_arr > max(0.004, float(np.mean(rms_arr)) * 0.8)
            peaks = sum(
                1
                for i in range(1, len(rms_arr) - 1)
                if above[i] and rms_arr[i] > rms_arr[i - 1] and rms_arr[i] >= rms_arr[i + 1]
            )
            speech_rate_proxy = round(peaks / duration_sec, 2)

        return AcousticFeatures(
            rms_mean=round(float(np.mean(rms_arr)), 5),
            rms_std=round(float(np.std(rms_arr)), 5),
            zcr_mean=round(float(np.mean(zcr_vals)), 4),
            zcr_std=round(float(np.std(zcr_vals)), 4),
            pitch_f0_mean_hz=round(float(np.mean(f0_vals)), 2) if f0_vals else 0.0,
            pitch_f0_std_hz=round(float(np.std(f0_vals)), 2) if len(f0_vals) > 1 else 0.0,
            voiced_frame_ratio=round(len(f0_vals) / max(1, len(rms_vals)), 3),
            spectral_centroid_hz=round(float(np.mean(centroids)), 2) if centroids else 0.0,
            spectral_flatness=round(float(np.mean(flatness_vals)), 4) if flatness_vals else 0.0,
            speech_rate_proxy=speech_rate_proxy,
            pause_ratio=round(pause_ratio, 3),
        )
