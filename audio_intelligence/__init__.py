"""Audio Intelligence Layer — Milestone 1 (quality, noise suppression, VAD, segmentation, acoustic features)."""

from audio_intelligence.audio_emotion import AcousticFeatureExtractor, AcousticFeatures
from audio_intelligence.audio_preprocessor import AudioIntelligencePipeline, EnhancedAudioResult
from audio_intelligence.audio_quality import AudioQualityAnalyzer, AudioQualityReport
from audio_intelligence.echo_canceller import PlaybackEchoCanceller
from audio_intelligence.noise_suppressor import (
    BaseNoiseSuppressor,
    RNNoiseSuppressor,
    SpectralNoiseSuppressor,
    create_noise_suppressor,
)
from audio_intelligence.speech_segmenter import SegmentedUtterance, SpeechSegmenter
from audio_intelligence.vad_processor import (
    AdaptiveTelephonyVAD,
    BaseVADProcessor,
    SileroVADProcessor,
    create_vad_processor,
)

__all__ = [
    "AcousticFeatureExtractor",
    "AcousticFeatures",
    "AdaptiveTelephonyVAD",
    "AudioIntelligencePipeline",
    "AudioQualityAnalyzer",
    "AudioQualityReport",
    "BaseNoiseSuppressor",
    "BaseVADProcessor",
    "EnhancedAudioResult",
    "PlaybackEchoCanceller",
    "RNNoiseSuppressor",
    "SegmentedUtterance",
    "SileroVADProcessor",
    "SpectralNoiseSuppressor",
    "SpeechSegmenter",
    "create_noise_suppressor",
    "create_vad_processor",
]
