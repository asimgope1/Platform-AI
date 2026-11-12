"""Speech emotion provider package (SpeechBrain wav2vec2 + acoustic fallback)."""

from providers.speech_emotion.speechbrain_emotion import SpeechBrainEmotionProvider, VoiceEmotionResult

__all__ = ["SpeechBrainEmotionProvider", "VoiceEmotionResult"]

