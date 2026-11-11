"""
Voice Activity Detection (VAD) processors for 8 kHz / 16 kHz telephony audio.

Provides:
  • AdaptiveTelephonyVAD — fast (<0.5 ms/frame) pure-NumPy VAD combining
    RMS energy, telephony speech-band ratio (300–3400 Hz), and spectral flatness.
  • SileroVADProcessor — pluggable Silero VAD adapter that uses locally cached
    Silero weights (when installed/cached) and transparently falls back to
    AdaptiveTelephonyVAD without blocking or downloading large weights unprompted.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class BaseVADProcessor(ABC):
    """Abstract interface for telephony VAD implementations."""

    name: str = "base"

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = float(threshold)

    @abstractmethod
    def frame_probability(self, frame_f32: np.ndarray, sample_rate: int = 8000) -> float:
        """Return speech likelihood in [0.0, 1.0] for one short audio frame."""

    def utterance_probabilities(
        self,
        samples_f32: np.ndarray,
        sample_rate: int = 8000,
        frame_ms: int = 20,
    ) -> List[float]:
        """Compute speech probabilities across non-overlapping frames of an utterance."""
        frame_len = max(1, int(sample_rate * frame_ms / 1000))
        n_samples = len(samples_f32)
        if n_samples == 0:
            return []
        probs: List[float] = []
        for start in range(0, n_samples, frame_len):
            chunk = samples_f32[start : start + frame_len]
            if len(chunk) < max(32, frame_len // 2):
                break
            probs.append(self.frame_probability(chunk, sample_rate=sample_rate))
        return probs


class AdaptiveTelephonyVAD(BaseVADProcessor):
    """
    Fast telephony-tuned VAD using voice-band energy, spectral flatness, and
    adaptive noise-floor awareness.

    Works on normalized float32 samples in [-1.0, 1.0] at 8 kHz or 16 kHz.
    """

    name: str = "adaptive_telephony"

    def __init__(
        self,
        threshold: float = 0.45,
        min_rms_floor: float = 0.0012,
    ) -> None:
        super().__init__(threshold=threshold)
        self.min_rms_floor = min_rms_floor

    def frame_probability(self, frame_f32: np.ndarray, sample_rate: int = 8000) -> float:
        if len(frame_f32) < 16:
            return 0.0

        # Remove frame DC bias before energy & spectral checks
        centered = frame_f32 - float(np.mean(frame_f32))
        rms = float(np.sqrt(np.mean(centered ** 2)))
        if rms < self.min_rms_floor:
            return 0.0

        # Normalized log-energy score (catches soft speech ~0.0025 through loud speech ~0.35)
        energy_score = float(np.clip((np.log10(rms + 1e-7) - np.log10(self.min_rms_floor)) / 1.15, 0.0, 1.0))

        # Power spectrum over telephony voice band (120 Hz - 3600 Hz, capturing F0 & formants)
        window = np.hanning(len(centered))
        spec = np.abs(np.fft.rfft(centered * window)) ** 2
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
        total_power = float(np.sum(spec)) + 1e-12

        voice_mask = (freqs >= 120.0) & (freqs <= 3600.0)
        low_hum_mask = freqs < 95.0
        voice_ratio = float(np.sum(spec[voice_mask])) / total_power
        hum_ratio = float(np.sum(spec[low_hum_mask])) / total_power

        # Spectral flatness within voice band (white/fan noise ~0.75-1.0; speech harmonics <0.40)
        voice_bins = spec[voice_mask] + 1e-12
        if len(voice_bins) > 2:
            geo_mean = float(np.exp(np.mean(np.log(voice_bins))))
            arith_mean = float(np.mean(voice_bins))
            flatness = float(np.clip(geo_mean / (arith_mean + 1e-12), 0.0, 1.0))
        else:
            flatness = 0.5

        harmonic_score = float(np.clip(1.0 - flatness, 0.0, 1.0))
        band_score = float(np.clip(voice_ratio - 0.40 * hum_ratio, 0.0, 1.0))

        # Weighted blend: energy + voice-band dominance + harmonic structure
        prob = 0.40 * energy_score + 0.30 * band_score + 0.30 * harmonic_score
        return float(np.clip(prob, 0.0, 1.0))

    def utterance_probabilities(
        self,
        samples_f32: np.ndarray,
        sample_rate: int = 8000,
        frame_ms: int = 20,
    ) -> List[float]:
        """Compute frame probabilities with utterance-level noise-floor adaptation."""
        frame_len = max(1, int(sample_rate * frame_ms / 1000))
        n_samples = len(samples_f32)
        if n_samples == 0:
            return []

        frames = [
            samples_f32[start : start + frame_len]
            for start in range(0, n_samples, frame_len)
            if len(samples_f32[start : start + frame_len]) >= max(32, frame_len // 2)
        ]
        if not frames:
            return []

        # Estimate stationary background floor using 15th percentile RMS when >= 6 frames
        frame_rms = [float(np.sqrt(np.mean((f - np.mean(f)) ** 2))) for f in frames]
        noise_floor = float(np.percentile(frame_rms, 15)) if len(frame_rms) >= 6 else self.min_rms_floor

        raw_probs: List[float] = []
        for f, rms in zip(frames, frame_rms):
            base_prob = self.frame_probability(f, sample_rate=sample_rate)
            # Penalize stationary broadband noise-floor frames (high flatness, no SNR rise)
            if noise_floor > self.min_rms_floor and rms < noise_floor * 1.65:
                centered = f - float(np.mean(f))
                spec = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2 + 1e-12
                freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
                vbins = spec[(freqs >= 120.0) & (freqs <= 3600.0)]
                flatness = float(np.exp(np.mean(np.log(vbins))) / (np.mean(vbins) + 1e-12)) if len(vbins) > 2 else 0.5
                if flatness > 0.42:
                    base_prob *= 0.35
            raw_probs.append(float(np.clip(base_prob, 0.0, 1.0)))

        return raw_probs


class SileroVADProcessor(BaseVADProcessor):
    """
    Silero VAD adapter with safe local-cache probing and AdaptiveTelephonyVAD fallback.
    Only loads Silero if `silero_vad` package is installed or local torch hub cache exists
    (unless `ALLOW_SILERO_DOWNLOAD=true`).
    """

    name: str = "silero"

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold)
        self._fallback = AdaptiveTelephonyVAD(threshold=threshold)
        self._model = None
        self._init_silero()

    def _init_silero(self) -> None:
        try:
            import silero_vad  # type: ignore

            self._model = silero_vad.load_silero_vad()
            logger.info("[VAD] Loaded silero_vad package model.")
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"[VAD] silero_vad package init failed ({exc})")

        # Only attempt torch.hub if cached locally or explicitly allowed
        allow_download = os.getenv("ALLOW_SILERO_DOWNLOAD", "false").lower() in ("1", "true", "yes")
        hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "snakers4_silero-vad_master")
        if os.path.isdir(hub_dir) or allow_download:
            try:
                import torch

                self._model, _ = torch.hub.load(
                    repo_or_dir=hub_dir if os.path.isdir(hub_dir) else "snakers4/silero-vad",
                    model="silero_vad",
                    source="local" if os.path.isdir(hub_dir) else "github",
                    force_reload=False,
                    trust_repo=True,
                )
                logger.info("[VAD] Loaded Silero VAD via torch.hub.")
                return
            except Exception as exc:
                logger.debug(f"[VAD] Silero torch.hub unavailable ({exc}); using adaptive fallback.")

        logger.debug("[VAD] Silero VAD not cached locally; using AdaptiveTelephonyVAD fallback.")

    @property
    def is_silero_active(self) -> bool:
        return self._model is not None

    def frame_probability(self, frame_f32: np.ndarray, sample_rate: int = 8000) -> float:
        if self._model is None:
            return self._fallback.frame_probability(frame_f32, sample_rate=sample_rate)

        try:
            import torch

            # Silero supports 8000 Hz (256 samples) and 16000 Hz (512 samples)
            win = 256 if sample_rate == 8000 else 512
            if len(frame_f32) < win:
                return self._fallback.frame_probability(frame_f32, sample_rate=sample_rate)
            tensor = torch.from_numpy(frame_f32[:win].astype(np.float32))
            with torch.no_grad():
                return float(self._model(tensor, sample_rate).item())
        except Exception as exc:
            logger.debug(f"[VAD] Silero frame inference failed ({exc}); using fallback.")
            return self._fallback.frame_probability(frame_f32, sample_rate=sample_rate)

    def utterance_probabilities(
        self,
        samples_f32: np.ndarray,
        sample_rate: int = 8000,
        frame_ms: int = 20,
    ) -> List[float]:
        if self._model is None:
            return self._fallback.utterance_probabilities(samples_f32, sample_rate=sample_rate, frame_ms=frame_ms)

        try:
            import torch

            win = 256 if sample_rate == 8000 else 512
            probs: List[float] = []
            if hasattr(self._model, "reset_states"):
                self._model.reset_states()
            for start in range(0, len(samples_f32) - win + 1, win):
                tensor = torch.from_numpy(samples_f32[start : start + win].astype(np.float32))
                with torch.no_grad():
                    probs.append(float(self._model(tensor, sample_rate).item()))
            return probs if probs else self._fallback.utterance_probabilities(samples_f32, sample_rate=sample_rate)
        except Exception as exc:
            logger.debug(f"[VAD] Silero utterance inference failed ({exc}); using fallback.")
            return self._fallback.utterance_probabilities(samples_f32, sample_rate=sample_rate, frame_ms=frame_ms)


def create_vad_processor(provider: str = "adaptive", threshold: float = 0.45) -> BaseVADProcessor:
    """Factory returning the configured VAD processor with safe local defaults."""
    normalized = (provider or "adaptive").strip().lower()
    if normalized == "silero":
        return SileroVADProcessor(threshold=threshold)
    return AdaptiveTelephonyVAD(threshold=threshold)
