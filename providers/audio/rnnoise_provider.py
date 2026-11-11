"""
Optional RNNoise native/Python wrapper with safe availability detection.
When native RNNoise bindings are not present on the host, the Audio Intelligence
pipeline transparently falls back to the built-in SpectralNoiseSuppressor.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RNNoiseProvider:
    """Thin adapter around optional `pyrnnoise` or `rnnoise` bindings."""

    def __init__(self) -> None:
        self._backend = None
        self._available = False
        self._probe_backend()

    def _probe_backend(self) -> None:
        for mod_name in ("pyrnnoise", "rnnoise_wrapper"):
            try:
                __import__(mod_name)
                self._available = True
                logger.info(f"[AudioProvider] Native RNNoise backend detected: {mod_name}")
                return
            except ImportError:
                continue
        logger.debug("[AudioProvider] RNNoise library not installed; using spectral fallback.")

    @property
    def is_available(self) -> bool:
        return self._available

    def suppress_pcm16(self, pcm: bytes, sample_rate: int = 8000) -> Optional[bytes]:
        """Attempt native RNNoise suppression; returns None if unavailable."""
        if not self._available or not pcm:
            return None
        try:
            import pyrnnoise  # type: ignore

            denoiser = pyrnnoise.RNNoise(sample_rate=sample_rate)
            # pyrnnoise yields tuples; concatenate denoised chunks if supported
            chunks = []
            for _, denoised_frame in denoiser.process_chunk(pcm):
                chunks.append(denoised_frame)
            return b"".join(chunks) if chunks else None
        except Exception as exc:
            logger.debug(f"[AudioProvider] RNNoise execution skipped ({exc})")
            return None
