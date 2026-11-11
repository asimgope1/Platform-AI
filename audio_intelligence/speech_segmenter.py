"""
Speech Segmentation with Hangover & Pre-Roll Protection.

Trims excessive leading/trailing dead air from telephony utterances while
preserving 120 ms onset pre-roll (for unvoiced stops like /p/, /t/, /k/) and
180 ms trailing hangover (for quiet word endings).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from audio_intelligence.vad_processor import AdaptiveTelephonyVAD, BaseVADProcessor


@dataclass
class SegmentedUtterance:
    pcm: bytes
    start_ms: float
    end_ms: float
    speech_duration_ms: float
    original_duration_ms: float
    was_trimmed: bool


class SpeechSegmenter:
    """Segments raw s16le PCM to active speech boundaries with hangover padding."""

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_ms: int = 20,
        preroll_ms: int = 120,
        hangover_ms: int = 180,
        vad_processor: Optional[BaseVADProcessor] = None,
        vad_threshold: float = 0.42,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.preroll_frames = max(1, preroll_ms // frame_ms)
        self.hangover_frames = max(1, hangover_ms // frame_ms)
        self.vad_threshold = vad_threshold
        self.vad = vad_processor or AdaptiveTelephonyVAD(threshold=vad_threshold)

    def segment(self, pcm: bytes) -> SegmentedUtterance:
        """Trim excessive outer silence around detected speech frames."""
        if not pcm or len(pcm) < 4:
            return SegmentedUtterance(
                pcm=pcm,
                start_ms=0.0,
                end_ms=0.0,
                speech_duration_ms=0.0,
                original_duration_ms=0.0,
                was_trimmed=False,
            )

        usable_len = len(pcm) - (len(pcm) % 2)
        samples_i16 = np.frombuffer(pcm[:usable_len], dtype="<i2").astype(np.float32)
        original_duration_ms = (len(samples_i16) / self.sample_rate) * 1000.0
        samples_f32 = (samples_i16 - float(np.mean(samples_i16))) / 32768.0

        probs = self.vad.utterance_probabilities(
            samples_f32,
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
        )
        if not probs:
            return SegmentedUtterance(
                pcm=pcm[:usable_len],
                start_ms=0.0,
                end_ms=original_duration_ms,
                speech_duration_ms=0.0,
                original_duration_ms=original_duration_ms,
                was_trimmed=False,
            )

        active_indices = [idx for idx, p in enumerate(probs) if p >= self.vad_threshold]
        if not active_indices:
            return SegmentedUtterance(
                pcm=pcm[:usable_len],
                start_ms=0.0,
                end_ms=original_duration_ms,
                speech_duration_ms=0.0,
                original_duration_ms=original_duration_ms,
                was_trimmed=False,
            )

        start_frame = max(0, active_indices[0] - self.preroll_frames)
        end_frame = min(len(probs), active_indices[-1] + 1 + self.hangover_frames)

        frame_bytes = int(self.sample_rate * self.frame_ms / 1000) * 2
        start_byte = start_frame * frame_bytes
        end_byte = min(usable_len, end_frame * frame_bytes)

        trimmed = pcm[start_byte:end_byte]
        was_trimmed = start_frame > 0 or end_byte < usable_len
        speech_duration_ms = len(active_indices) * float(self.frame_ms)

        return SegmentedUtterance(
            pcm=trimmed if trimmed else pcm[:usable_len],
            start_ms=round(start_frame * float(self.frame_ms), 2),
            end_ms=round((end_byte / 2 / self.sample_rate) * 1000.0, 2),
            speech_duration_ms=round(speech_duration_ms, 2),
            original_duration_ms=round(original_duration_ms, 2),
            was_trimmed=was_trimmed,
        )
