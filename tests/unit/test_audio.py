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

