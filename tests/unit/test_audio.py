import pytest
from audio.service import AudioService
from providers.speech_to_text.base import MockSpeechToTextAdapter
from providers.text_to_speech.base import MockTextToSpeechAdapter


@pytest.mark.asyncio
async def test_mock_stt():
    adapter = MockSpeechToTextAdapter("mock_stt", {})
    audio_data = b"RIFF" + b"\x00" * 1000
    res = await adapter.transcribe(audio_data, filename="voice.mp3")
    assert "Transcribed speech content" in res["text"]
    assert res["language"] == "en"
    assert res["duration"] > 0


@pytest.mark.asyncio
async def test_mock_tts():
    adapter = MockTextToSpeechAdapter("mock_tts", {})
    audio_bytes = await adapter.synthesize(text="Hello from AI platform")
    assert len(audio_bytes) > 100
    assert audio_bytes.startswith((b"\xff", b"ID3", b"RIFF"))


@pytest.mark.asyncio
async def test_audio_service():
    service = AudioService()
    transcription = await service.transcribe(b"test-audio-bytes", filename="call.mp3")
    assert "call.mp3" in transcription["text"]

    speech = await service.synthesize("Testing TTS service")
    assert len(speech) > 0


@pytest.mark.asyncio
async def test_whisper_local_stt():
    from providers.speech_to_text.whisper_local import WhisperLocalSpeechToTextAdapter
    adapter = WhisperLocalSpeechToTextAdapter("whisper_local", {})
    
    # 1. Test empty audio
    empty_res = await adapter.transcribe(b"")
    assert empty_res["text"] == ""
    
    # 2. Test fallback for corrupt/dummy bytes
    dummy_res = await adapter.transcribe(b"dummy_corrupt_bytes", filename="recording.wav")
    assert "recording.wav" in dummy_res["text"]


def test_phonetic_noise_normalization():
    import os, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "comone.settings")
    django.setup()
    from apps.voice.cognitive_pipeline import CognitiveVoicePipeline
    raw_noisy_text = "UPA Reef Fund is to take care about this."
    normalized = CognitiveVoicePipeline._normalize_phonetic_transcription(raw_noisy_text)
    assert "UPI" in normalized
    assert "refund" in normalized

    intent, score, emotion = CognitiveVoicePipeline._analyze_sentiment_and_intent(raw_noisy_text)
    assert intent == "UPI_REFUND_INQUIRY"


def test_noise_hallucination_discard():
    import os, django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "comone.settings")
    django.setup()
    from apps.voice.cognitive_pipeline import CognitiveVoicePipeline
    assert CognitiveVoicePipeline._is_noise_hallucination("") is True
    assert CognitiveVoicePipeline._is_noise_hallucination("...") is True
    assert CognitiveVoicePipeline._is_noise_hallucination("(Ambient noise)") is True
    assert CognitiveVoicePipeline._is_noise_hallucination("the") is True
    assert CognitiveVoicePipeline._is_noise_hallucination("uh") is True
    assert CognitiveVoicePipeline._is_noise_hallucination("you you you you you") is True
    assert CognitiveVoicePipeline._is_noise_hallucination("UPI refund status") is False


