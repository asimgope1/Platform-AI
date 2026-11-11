"""
Acoustic Echo Cancellation (AEC) / Playback Bleed Suppression for telephony audio.

When caller microphone input overlaps with recent AI playback frames,
estimates normalized cross-correlation lag and attenuates echo bleed
without distorting near-end caller voice. Passes through when no
reference playback audio is supplied.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.signal

from providers.audio.webrtc_provider import WebRTCAudioProvider

logger = logging.getLogger(__name__)


class PlaybackEchoCanceller:
    """Lightweight cross-correlated echo suppressor with WebRTC APM probe."""

    def __init__(self, max_lag_ms: int = 120, attenuation_strength: float = 0.65) -> None:
        self.max_lag_ms = max_lag_ms
        self.attenuation_strength = float(np.clip(attenuation_strength, 0.0, 0.95))
        self.webrtc = WebRTCAudioProvider()

    def process(
        self,
        mic_pcm: bytes,
        reference_pcm: Optional[bytes] = None,
        sample_rate: int = 8000,
    ) -> bytes:
        """Subtract correlated far-end playback echo from near-end PCM if detected."""
        if not mic_pcm or not reference_pcm or len(mic_pcm) < 640 or len(reference_pcm) < 640:
            return mic_pcm

        usable_mic = len(mic_pcm) - (len(mic_pcm) % 2)
        usable_ref = len(reference_pcm) - (len(reference_pcm) % 2)
        mic = np.frombuffer(mic_pcm[:usable_mic], dtype="<i2").astype(np.float32)
        ref = np.frombuffer(reference_pcm[:usable_ref], dtype="<i2").astype(np.float32)

        # Align length for correlation check
        n = min(len(mic), len(ref))
        mic_seg = mic[:n]
        ref_seg = ref[:n]

        mic_norm = float(np.linalg.norm(mic_seg))
        ref_norm = float(np.linalg.norm(ref_seg))
        if mic_norm < 1e-3 or ref_norm < 1e-3:
            return mic_pcm

        max_lag_samples = min(n // 2, int(sample_rate * self.max_lag_ms / 1000))
        if max_lag_samples <= 0:
            return mic_pcm

        corr = scipy.signal.correlate(mic_seg, ref_seg, mode="full")
        mid = len(corr) // 2
        window = corr[mid : mid + max_lag_samples]
        best_offset = int(np.argmax(np.abs(window)))
        peak_corr = float(window[best_offset]) / (mic_norm * ref_norm + 1e-9)

        # Only cancel if clear acoustic echo correlation exists (|r| >= 0.35)
        if abs(peak_corr) < 0.35:
            return mic_pcm

        shifted_ref = np.zeros_like(mic)
        if best_offset < n:
            copy_len = min(len(ref), len(mic) - best_offset)
            shifted_ref[best_offset : best_offset + copy_len] = ref[:copy_len]

        # Projection scale alpha = (mic · shifted_ref) / (shifted_ref · shifted_ref)
        ref_energy = float(np.dot(shifted_ref, shifted_ref)) + 1e-9
        alpha = float(np.dot(mic, shifted_ref) / ref_energy)
        alpha = float(np.clip(alpha, -1.2, 1.2)) * self.attenuation_strength

        cleaned = mic - alpha * shifted_ref
        cleaned = np.clip(cleaned, -32767.0, 32767.0)
        return cleaned.astype("<i2").tobytes()

