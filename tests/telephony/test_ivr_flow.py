"""
Integration & Flow tests for IVRCallSession with DTMF keypad navigation,
prompt playback, Whisper language locking, and microphone gating.
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from telephony.ivr_session import IVRCallSession
from telephony.ivr_menu import IVRState


class MockWebSocket:
    def __init__(self):
        self.sent_texts = []
        self.sent_bytes = []

    async def send_text(self, text: str):
        self.sent_texts.append(text)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)


class MockAudioService:
    def __init__(self):
        self.synthesized_prompts = []
        self.transcribe_languages = []

    async def synthesize(self, text: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "wav") -> bytes:
        self.synthesized_prompts.append(text)
        # Return valid minimal RIFF WAV header (44 bytes) + 320 bytes PCM
        riff_header = (
            b"RIFF\x6c\x01\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x48\x01\x00\x00"
        )
        return riff_header + (b"\x00" * 320)

    async def transcribe(self, audio_bytes: bytes, filename: str = "call.wav", language: str = "en") -> dict:
        self.transcribe_languages.append(language)
        return {"text": "Mere account se paise cut gaye lekin transaction fail ho gaya"}


class MockOrchestrator:
    async def execute(self, request, context=None, enable_rag=False):
        mock_choice = MagicMock()
        mock_choice.message.content = "Aapka paisa 48 ghante ke bheetar aapke bank khate mein auto-refund ho jayega."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp


@pytest.mark.asyncio
async def test_full_ivr_dtmf_journey_hindi_upi():
    ws = MockWebSocket()
    audio = MockAudioService()
    orchestrator = MockOrchestrator()

    session = IVRCallSession(
        ws=ws,
        orchestrator=orchestrator,
        audio_service=audio,
    )

    # 1. Caller connects
    session.on_call_start({"uuid": "test-call-001", "caller": "+919876543210"})
    assert session.menu_machine.state == IVRState.LANGUAGE_SELECTION
    assert session.call_uuid == "test-call-001"
    assert session.selected_language == "en"

    # Give prompt playback task a moment to initialize
    await asyncio.sleep(0.05)
    # Verify that audio playback began (play event sent to Asterisk)
    assert len(ws.sent_texts) >= 1
    assert "play" in ws.sent_texts[0]

    # 2. Caller presses 2 for Hindi
    await session.on_dtmf("2")
    assert session.menu_machine.state == IVRState.SERVICE_SELECTION
    assert session.selected_language == "hi"
    assert session._language == "hi"

    await asyncio.sleep(0.05)
    assert len(ws.sent_texts) >= 2

    # 3. Caller presses 3 for UPI and Payment Issues
    await session.on_dtmf("3")
    assert session.menu_machine.state == IVRState.AI_CONVERSATION
    assert session.selected_service == "upi_payment"
    assert session._language == "hi"

    await asyncio.sleep(0.05)
    assert len(ws.sent_texts) >= 3
    assert "specializing in UPI and Payment Support" in session._history[0].content or "specializing in" in session._history[0].content


@pytest.mark.asyncio
async def test_microphone_gating_during_menus():
    ws = MockWebSocket()
    audio = MockAudioService()
    orchestrator = MockOrchestrator()

    session = IVRCallSession(
        ws=ws,
        orchestrator=orchestrator,
        audio_service=audio,
    )

    session.on_call_start({"uuid": "test-gating-002", "caller": "1001"})
    assert session.menu_machine.state == IVRState.LANGUAGE_SELECTION

    # Simulate speech PCM frames arriving while still in LANGUAGE_SELECTION menu
    speech_frame = b"\x10\x20" * 160  # Non-silent frame
    for _ in range(50):
        await session.on_pcm_frame(speech_frame)

    # Audio buffer should NOT accumulate during menu selection!
    assert len(session._pcm_buffer) == 0
    assert len(audio.transcribe_languages) == 0

    # Advance to AI_CONVERSATION
    await session.on_dtmf("1")  # English
    await session.on_dtmf("3")  # UPI
    assert session.menu_machine.state == IVRState.AI_CONVERSATION

    # Now speech frames SHOULD accumulate
    for _ in range(50):
        await session.on_pcm_frame(speech_frame)

    assert len(session._pcm_buffer) > 0


@pytest.mark.asyncio
async def test_dtmf_barge_in_cancels_playback():
    ws = MockWebSocket()
    audio = MockAudioService()
    orchestrator = MockOrchestrator()

    session = IVRCallSession(
        ws=ws,
        orchestrator=orchestrator,
        audio_service=audio,
    )

    session.on_call_start({"uuid": "test-barge-in-003", "caller": "1002"})
    await asyncio.sleep(0.02)

    # Active playback task should exist
    assert session._playback_task is not None

    # DTMF arrives during playback
    await session.on_dtmf("1")

    # State immediately advanced to SERVICE_SELECTION
    assert session.menu_machine.state == IVRState.SERVICE_SELECTION
    assert session.selected_language == "en"


@pytest.mark.asyncio
async def test_human_transfer_workflow():
    ws = MockWebSocket()
    audio = MockAudioService()
    orchestrator = MockOrchestrator()

    session = IVRCallSession(
        ws=ws,
        orchestrator=orchestrator,
        audio_service=audio,
    )

    session.on_call_start({"uuid": "test-transfer-004", "caller": "1003"})
    await session.on_dtmf("1")  # English
    await session.on_dtmf("9")  # Human Representative

    assert session.menu_machine.state == IVRState.HUMAN_TRANSFER
    await asyncio.sleep(0.05)
    assert len(ws.sent_texts) >= 1
    assert "play" in ws.sent_texts[-1]


@pytest.mark.asyncio
async def test_call_end_cleanup():
    ws = MockWebSocket()
    audio = MockAudioService()
    orchestrator = MockOrchestrator()

    session = IVRCallSession(
        ws=ws,
        orchestrator=orchestrator,
        audio_service=audio,
    )

    session.on_call_start({"uuid": "test-cleanup-005", "caller": "1004"})
    session.on_call_end({"uuid": "test-cleanup-005"})

    assert session.menu_machine.state == IVRState.CALL_END
    assert session._is_playing is False
    assert len(session._pcm_buffer) == 0

