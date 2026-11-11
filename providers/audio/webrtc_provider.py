"""
Optional WebRTC audio / VAD provider wrapper with safe fallback.
Detects optional `webrtcvad` or `webrtc_audio_processing` bindings without
introducing mandatory build dependencies on constrained environments.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WebRTCAudioProvider:
    """Adapter around optional `webrtcvad` / WebRTC audio processing libraries."""

    def __init__(self, aggressiveness: int = 2) -> None:
        self._aggressiveness = max(0, min(3, aggressiveness))
        self._vad = None
        self._apm = None
        self._probe_backends()

    def _probe_backends(self) -> None:
        try:
            import webrtcvad  # type: ignore

            self._vad = webrtcvad.Vad(self._aggressiveness)
            logger.info("[AudioProvider] webrtcvad backend loaded.")
        except ImportError:
            self._vad = None

        try:
            import webrtc_audio_processing  # type: ignore  # noqa: F401

            self._apm = webrtc_audio_processing
            logger.info("[AudioProvider] webrtc_audio_processing backend loaded.")
        except ImportError:
            self._apm = None

    @property
    def has_vad(self) -> bool:
        return self._vad is not None

    @property
    def has_apm(self) -> bool:
        return self._apm is not None

    def is_speech_frame(self, pcm_frame: bytes, sample_rate: int = 8000) -> Optional[bool]:
        """Classify a single 10/20/30ms PCM16 frame if webrtcvad is installed."""
        if self._vad is None:
            return None
        try:
            return bool(self._vad.is_speech(pcm_frame, sample_rate))
        except Exception as exc:
            logger.debug(f"[AudioProvider] webrtcvad frame check failed: {exc}")
            return None

