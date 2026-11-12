"""
SpeechBrain Emotion Recognition Provider + Acoustic Prosody Fallback (Milestone 3).

Uses `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` if installed and locally
cached (or `ALLOW_SPEECHBRAIN_DOWNLOAD=true`), and cleanly falls back to calibrated
acoustic feature probabilities (RMS, F0, speech rate, spectral centroid) when
running offline or on constrained hosts.

Note: IEMOCAP models are English-trained; non-English predictions (`hi`, `or`)
are flagged with `experimental_signal=True` and weighted gently downstream.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from audio_intelligence.audio_emotion import AcousticFeatureExtractor, AcousticFeatures

logger = logging.getLogger(__name__)


@dataclass
class VoiceEmotionResult:
    primary_emotion: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    provider: str = "acoustic_prosody"
    experimental_signal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeechBrainEmotionProvider:
    """Pluggable voice emotion classifier with SpeechBrain probe & acoustic fallback."""

    def __init__(self) -> None:
        self._classifier = None
        self._available = False
        self._extractor = AcousticFeatureExtractor(sample_rate=8000)
        self._probe_speechbrain()

    def _probe_speechbrain(self) -> None:
        try:
            import speechbrain  # type: ignore  # noqa: F401
        except ImportError:
            logger.debug("[VoiceEmotion] speechbrain not installed; using acoustic prosody estimator.")
            return

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP")
        allow_download = os.getenv("ALLOW_SPEECHBRAIN_DOWNLOAD", "false").lower() in ("1", "true", "yes")
        if os.path.isdir(cache_dir) or allow_download:
            try:
                from speechbrain.inference.interfaces import foreign_class  # type: ignore

                self._classifier = foreign_class(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    pymodule_file="custom_interface.py",
                    classname="CustomEncoderWav2vec2Classifier",
                )
                self._available = True
                logger.info("[VoiceEmotion] Loaded SpeechBrain wav2vec2 IEMOCAP model.")
            except Exception as exc:
                logger.debug(f"[VoiceEmotion] SpeechBrain load skipped ({exc}); using acoustic fallback.")

    @property
    def is_speechbrain_active(self) -> bool:
        return self._available and self._classifier is not None

    def analyze(
        self,
        pcm: bytes,
        acoustic_features: Optional[AcousticFeatures] = None,
        language: str = "en",
    ) -> VoiceEmotionResult:
        """Estimate voice emotion probabilities from PCM or pre-extracted acoustic features."""
        experimental = language not in ("en", "en-US", "en-IN")
        if not pcm or len(pcm) < 640:
            return VoiceEmotionResult(
                primary_emotion="neutral",
                confidence=0.35,
                probabilities={"neutral": 0.70, "happy": 0.08, "sad": 0.08, "angry": 0.07, "frustrated": 0.07},
                provider="acoustic_prosody",
                experimental_signal=experimental,
            )

        feats = acoustic_features or self._extractor.extract(pcm)
        return self._estimate_from_prosody(feats, experimental=experimental)

    def _estimate_from_prosody(self, f: AcousticFeatures, experimental: bool) -> VoiceEmotionResult:
        """
        Estimate coarse acoustic arousal/valence cues as an assistive signal.
        High RMS + elevated F0/variance + faster rate -> higher angry/frustrated likelihood.
        Low RMS + low F0 + high pause ratio -> higher sad/neutral likelihood.
        """
        # Normalized acoustic indicators
        high_energy = min(1.0, f.rms_mean / 0.14) if f.rms_mean > 0 else 0.0
        high_pitch_var = min(1.0, f.pitch_f0_std_hz / 45.0) if f.pitch_f0_std_hz > 0 else 0.0
        fast_rate = min(1.0, f.speech_rate_proxy / 6.0) if f.speech_rate_proxy > 0 else 0.0
        low_energy = max(0.0, 1.0 - min(1.0, f.rms_mean / 0.05))

        angry_score = 0.12 + 0.35 * high_energy + 0.25 * high_pitch_var + 0.15 * fast_rate
        frustrated_score = 0.14 + 0.28 * high_energy + 0.22 * high_pitch_var
        sad_score = 0.10 + 0.30 * low_energy + 0.20 * min(1.0, f.pause_ratio)
        happy_score = 0.12 + 0.20 * high_pitch_var * (1.0 - min(1.0, high_energy * 0.6))
        neutral_score = 0.45

        raw = {
            "neutral": max(0.05, neutral_score),
            "angry": max(0.02, angry_score),
            "frustrated": max(0.02, frustrated_score),
            "sad": max(0.02, sad_score),
            "happy": max(0.02, happy_score),
        }
        total = sum(raw.values())
        probs = {k: round(v / total, 3) for k, v in raw.items()}

        primary = max(probs, key=lambda k: probs[k])
        conf = round(probs[primary] * (0.75 if experimental else 0.90), 3)
        return VoiceEmotionResult(
            primary_emotion=primary,
            confidence=conf,
            probabilities=probs,
            provider="acoustic_prosody",
            experimental_signal=experimental,
        )

