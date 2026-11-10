import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock

from telephony.ivr_session import IVRCallSession
from telephony.ivr_menu import IVRState
from providers.chat.ivr_mini_model import ivr_mini_brain


class TestIVRVoiceBargeIn:

    def _generate_pcm_frame(self, amplitude: int = 5000, length: int = 160) -> bytes:
        samples = (np.sin(np.linspace(0, 3.14 * 4, length)) * amplitude).astype("<i2")
        return samples.tobytes()

    @pytest.mark.asyncio
    async def test_voice_barge_in_cancels_playback(self):
        mock_ws = AsyncMock()
        mock_audio = AsyncMock()
        mock_orch = AsyncMock()

        session = IVRCallSession(
            ws=mock_ws,
            orchestrator=mock_orch,
            audio_service=mock_audio,
        )
        session.menu_machine.state = IVRState.AI_CONVERSATION

        # Simulate bot actively speaking
        session._is_playing = True
        dummy_task = asyncio.create_task(asyncio.sleep(10))
        session._playback_task = dummy_task

        # Send high-energy caller audio (RMS > 650)
        loud_frame = self._generate_pcm_frame(amplitude=8000)
        await session.on_pcm_frame(loud_frame)
        assert session._is_playing is True  # Needs 2 frames for confirmation

        await session.on_pcm_frame(loud_frame)
        await asyncio.sleep(0)
        assert session._is_playing is False
        assert dummy_task.cancelling() > 0 or dummy_task.cancelled()
        assert len(session._pcm_buffer) > 0

    @pytest.mark.asyncio
    async def test_transcript_barge_in_cancels_playback(self):
        mock_ws = AsyncMock()
        mock_audio = AsyncMock()
        mock_orch = AsyncMock()

        session = IVRCallSession(
            ws=mock_ws,
            orchestrator=mock_orch,
            audio_service=mock_audio,
        )
        session.menu_machine.state = IVRState.AI_CONVERSATION
        session.ignore_pbx_transcript = False

        session._is_playing = True
        dummy_task = asyncio.create_task(asyncio.sleep(10))
        session._playback_task = dummy_task

        # Patch _run_ai_and_reply to avoid full synthesis in this unit test
        session._run_ai_and_reply = AsyncMock()

        await session.on_transcript({"text": "cancel that"})
        await asyncio.sleep(0)
        assert session._is_playing is False
        assert dummy_task.cancelling() > 0 or dummy_task.cancelled()
        session._run_ai_and_reply.assert_awaited_once_with("cancel that")

    @pytest.mark.asyncio
    async def test_transcript_ignored_when_flag_enabled(self):
        mock_ws = AsyncMock()
        mock_audio = AsyncMock()
        mock_orch = AsyncMock()

        session = IVRCallSession(
            ws=mock_ws,
            orchestrator=mock_orch,
            audio_service=mock_audio,
        )
        session.menu_machine.state = IVRState.AI_CONVERSATION
        session.ignore_pbx_transcript = True
        session._is_playing = True
        session._run_ai_and_reply = AsyncMock()

        await session.on_transcript({"text": "cancel that"})
        session._run_ai_and_reply.assert_not_called()
        assert session._is_playing is True  # Did not interrupt playback

    def test_substring_hi_eliminated(self):
        # Queries containing 'hi' as a substring must NOT trigger greeting intent
        assert not ivr_mini_brain._is_greeting_intent("what think you can do")
        assert not ivr_mini_brain._is_greeting_intent("which card is blocked")
        assert not ivr_mini_brain._is_greeting_intent("this is a problem")
        assert not ivr_mini_brain._is_greeting_intent("machine learning")

        # True greetings MUST trigger greeting intent
        assert ivr_mini_brain._is_greeting_intent("hi")
        assert ivr_mini_brain._is_greeting_intent("hello there")
        assert ivr_mini_brain._is_greeting_intent("नमस्ते")

    @pytest.mark.asyncio
    async def test_punctuation_and_noise_transcripts_ignored(self):
        mock_ws = AsyncMock()
        mock_audio = AsyncMock()
        mock_orch = AsyncMock()

        session = IVRCallSession(
            ws=mock_ws,
            orchestrator=mock_orch,
            audio_service=mock_audio,
        )
        session.menu_machine.state = IVRState.AI_CONVERSATION
        session.ignore_pbx_transcript = False
        session._run_ai_and_reply = AsyncMock()

        # Pure punctuation, noise or whitespace must be discarded
        await session.on_transcript({"text": "."})
        await session.on_transcript({"text": "..."})
        await session.on_transcript({"text": "   "})
        await session.on_transcript({"text": "!?"})
        # Single stopword noise from PBX
        await session.on_transcript({"text": "the"})
        await session.on_transcript({"text": "a ago"})
        session._run_ai_and_reply.assert_not_called()

        # Real speech passes through
        await session.on_transcript({"text": "Hello"})
        session._run_ai_and_reply.assert_awaited_once_with("Hello")
