import inspect
from typing import Dict, Any, Optional
from providers.registry import provider_registry


class AudioService:
    """Service layer coordinating Speech-to-Text and Text-to-Speech operations."""

    def __init__(
        self,
        stt_provider_id: str = "default",
        tts_provider_id: str = "default",
    ):
        self.stt_provider_id = stt_provider_id
        self.tts_provider_id = tts_provider_id

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        adapter = provider_registry.get_stt_adapter(self.stt_provider_id)
        return await adapter.transcribe(
            audio_bytes=audio_bytes,
            filename=filename,
            model=model,
            language=language,
        )

    async def synthesize(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        emotion: Optional[str] = None,
    ) -> bytes:
        adapter = provider_registry.get_tts_adapter(self.tts_provider_id)
        kwargs: Dict[str, Any] = {
            "text": text,
            "model": model,
            "voice": voice,
            "response_format": response_format,
        }
        try:
            params = inspect.signature(adapter.synthesize).parameters
            if emotion and ("emotion" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())):
                kwargs["emotion"] = emotion
        except (TypeError, ValueError):
            pass
        return await adapter.synthesize(**kwargs)
