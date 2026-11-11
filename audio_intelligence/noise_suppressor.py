"""
Noise suppression implementations for 8 kHz / 16 kHz s16le telephony PCM.

Provides:
  • SpectralNoiseSuppressor — lightweight NumPy/SciPy STFT Wiener-style
    noise suppressor tuned to preserve telephony consonants and avoid musical noise.
  • Pluggable RNNoise wrapper via providers.audio.rnnoise_provider with automatic
    fallback to SpectralNoiseSuppressor.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import scipy.signal

from providers.audio.rnnoise_provider import RNNoiseProvider

logger = logging.getLogger(__name__)


class BaseNoiseSuppressor(ABC):
    """Abstract interface for telephony noise suppression providers."""

    name: str = "base"

    @abstractmethod
    def suppress(self, pcm: bytes, sample_rate: int = 8000, snr_db: Optional[float] = None) -> bytes:
        """Return denoised s16le mono PCM bytes (or original PCM on bypass)."""


class SpectralNoiseSuppressor(BaseNoiseSuppressor):
    """
    Conservative STFT Wiener noise suppressor for telephony speech.

    Key protections:
      1. High-pass filter at 85 Hz eliminates AC hum and mic wind thumps.
      2. Clean-speech bypass when SNR >= 24 dB so crisp calls aren't degraded.
      3. Spectral gain floor (`gain_floor=0.22`) prevents musical-noise artifacts
         and preserves unvoiced consonants (/s/, /t/, /k/, /p/).
    """

    name: str = "spectral"

    def __init__(
        self,
        gain_floor: float = 0.22,
        oversubtraction: float = 1.15,
        clean_bypass_snr_db: float = 24.0,
    ) -> None:
        self.gain_floor = gain_floor
        self.oversubtraction = oversubtraction
        self.clean_bypass_snr_db = clean_bypass_snr_db

    def suppress(self, pcm: bytes, sample_rate: int = 8000, snr_db: Optional[float] = None) -> bytes:
        if not pcm or len(pcm) < 640:  # < 40 ms at 8 kHz
            return pcm

        if snr_db is not None and snr_db >= self.clean_bypass_snr_db:
            return pcm

        usable_len = len(pcm) - (len(pcm) % 2)
        samples = np.frombuffer(pcm[:usable_len], dtype="<i2").astype(np.float32)

        # Remove DC offset
        samples = samples - float(np.mean(samples))

        # 1. Gentle 85 Hz Butterworth high-pass filter (removes line hum / rumble)
        if len(samples) > 32 and sample_rate >= 4000:
            try:
                sos = scipy.signal.butter(2, 85.0, btype="highpass", fs=sample_rate, output="sos")
                samples = scipy.signal.sosfiltfilt(sos, samples).astype(np.float32)
            except Exception as exc:
                logger.debug(f"[NoiseSuppressor] High-pass skipped: {exc}")

        # 2. STFT Wiener attenuation
        nperseg = max(64, int(sample_rate * 0.020))  # 160 samples at 8 kHz
        noverlap = nperseg // 2
        if len(samples) < nperseg * 2:
            return np.clip(samples, -32767.0, 32767.0).astype("<i2").tobytes()

        freqs, _, zxx = scipy.signal.stft(
            samples,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )
        mag = np.abs(zxx)
        phase = np.angle(zxx)
        power = mag ** 2

        # Estimate stationary noise PSD using lower 20th percentile per frequency bin
        noise_psd = np.percentile(power, 20, axis=1, keepdims=True)
        # Temporal smoothing across frames to prevent choppy gain jumps
        smooth_power = scipy.signal.lfilter([0.35, 0.65], [1.0], power, axis=1)

        # Protect high-frequency consonant band (>1.8 kHz) from aggressive over-subtraction
        freq_alpha = np.where(freqs[:, None] > 1800.0, self.oversubtraction * 0.72, self.oversubtraction)

        # Wiener-like gain: G = max(floor, (P - alpha * N) / P)
        snr_post = np.maximum(smooth_power - freq_alpha * noise_psd, 0.0)
        gain = np.where(smooth_power > 1e-9, snr_post / (smooth_power + 1e-9), self.gain_floor)
        gain = np.sqrt(np.clip(gain, self.gain_floor ** 2, 1.0))

        # Apply gain and reconstruct time-domain signal
        cleaned_zxx = (mag * gain) * np.exp(1j * phase)
        _, reconstructed = scipy.signal.istft(
            cleaned_zxx,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # Match exact input length and clip safely to int16 range
        if len(reconstructed) > len(samples):
            reconstructed = reconstructed[: len(samples)]
        elif len(reconstructed) < len(samples):
            reconstructed = np.pad(reconstructed, (0, len(samples) - len(reconstructed)))

        reconstructed = np.clip(reconstructed, -32767.0, 32767.0)
        return reconstructed.astype("<i2").tobytes()


class RNNoiseSuppressor(BaseNoiseSuppressor):
    """Uses native RNNoise when installed, otherwise falls back cleanly to SpectralNoiseSuppressor."""

    name: str = "rnnoise"

    def __init__(self) -> None:
        self._native = RNNoiseProvider()
        self._fallback = SpectralNoiseSuppressor()

    def suppress(self, pcm: bytes, sample_rate: int = 8000, snr_db: Optional[float] = None) -> bytes:
        if self._native.is_available:
            res = self._native.suppress_pcm16(pcm, sample_rate=sample_rate)
            if res:
                return res
        return self._fallback.suppress(pcm, sample_rate=sample_rate, snr_db=snr_db)


def create_noise_suppressor(provider: str = "spectral") -> BaseNoiseSuppressor:
    """Factory returning configured noise suppressor."""
    normalized = (provider or "spectral").strip().lower()
    if normalized == "rnnoise":
        return RNNoiseSuppressor()
    return SpectralNoiseSuppressor()
