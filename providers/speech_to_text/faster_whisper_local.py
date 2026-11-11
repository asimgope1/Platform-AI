"""
Optional Faster-Whisper (CTranslate2) Speech-to-Text Adapter.

Exposes rich decoding diagnostics (`avg_logprob`, `no_speech_prob`,
`compression_ratio`, `segments`) when `faster_whisper` is installed, and
cleanly delegates to `WhisperLocalSpeechToTextAdapter` if unavailable.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from typing import Any, Dict, Optional

import numpy as np

from platform_core.contracts.providers import BaseSpeechToTextAdapter
from providers.speech_to_text.whisper_local import WhisperLocalSpeechToTextAdapter

logger = logging.getLogger(__name__)


class FasterWhisperLocalAdapter(BaseSpeechToTextAdapter):
    """Pluggable CTranslate2 Whisper STT adapter with local Whisper fallback."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.model_size = os.getenv("FASTER_WHISPER_MODEL", config.get("model") or "base")
        self.device = os.getenv("FASTER_WHISPER_DEVICE", config.get("device") or "cpu")
        self.compute_type = os.getenv("FASTER_WHISPER_COMPUTE", config.get("compute_type") or "int8")
        self._fw_model = None
        self._available = False
        self._fallback = WhisperLocalSpeechToTextAdapter(f"{provider_id}-fallback", config)
        self._probe_faster_whisper()

    def _probe_faster_whisper(self) -> None:
        try:
            import faster_whisper  # type: ignore  # noqa: F401

            self._available = True
            logger.info("[STT] faster-whisper package available.")
        except ImportError:
            self._available = False
            logger.debug("[STT] faster-whisper not installed; will use WhisperLocalSpeechToTextAdapter.")

    @property
    def is_faster_whisper_available(self) -> bool:
        return self._available

    def _get_model(self):
        if self._fw_model is None and self._available:
            from faster_whisper import WhisperModel  # type: ignore

            self._fw_model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._fw_model

    def _transcribe_fw_sync(
        self,
        audio_bytes: bytes,
        filename: str,
        language: Optional[str],
    ) -> Dict[str, Any]:
        model = self._get_model()
        if model is None:
            return self._fallback._transcribe_sync(audio_bytes, filename=filename, language=language)

        t0 = time.perf_counter()
        audio_arr = None
        sr = 16000
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    arr = arr.reshape(-1, ch).mean(axis=1)
                if sr != 16000 and len(arr) > 0:
                    import scipy.signal

                    arr = scipy.signal.resample(arr, int(len(arr) * 16000 / sr)).astype(np.float32)
                audio_arr = arr
        except Exception:
            return self._fallback._transcribe_sync(audio_bytes, filename=filename, language=language)

        lang_arg = "hi" if language in ("hi", "or") else (language if language in ("en",) else None)
        segments_iter, info = model.transcribe(audio_arr, language=lang_arg, beam_size=1)
        segs = list(segments_iter)
        text = " ".join(s.text.strip() for s in segs if s.text).strip()

        avg_logprob = float(np.mean([s.avg_logprob for s in segs])) if segs else -1.5
        no_speech_prob = float(np.mean([s.no_speech_prob for s in segs])) if segs else 0.0
        comp_ratio = float(np.mean([s.compression_ratio for s in segs])) if segs else 1.0

        return {
            "text": text,
            "language": getattr(info, "language", language or "en"),
            "duration": round(len(audio_arr) / 16000.0, 2),
            "words": [],
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob}
                for s in segs
            ],
            "diagnostics": {
                "provider": "faster_whisper",
                "avg_logprob": round(avg_logprob, 3),
                "no_speech_prob": round(no_speech_prob, 3),
                "compression_ratio": round(comp_ratio, 3),
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        }

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not audio_bytes:
            return {"text": "", "language": language or "en", "duration": 0.0, "words": []}
        if not self._available:
            return await self._fallback.transcribe(audio_bytes, filename=filename, model=model, language=language)
        return await asyncio.to_thread(self._transcribe_fw_sync, audio_bytes, filename, language)
