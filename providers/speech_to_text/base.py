import os
from typing import Dict, Any, Optional
import httpx
from platform_core.contracts.providers import BaseSpeechToTextAdapter
from platform_core.errors.exceptions import ProviderError


class MockSpeechToTextAdapter(BaseSpeechToTextAdapter):
    """Deterministic mock speech-to-text adapter for tests and offline development."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        size_kb = len(audio_bytes) / 1024.0
        return {
            "text": f"Transcribed speech content from {filename} ({size_kb:.1f} KB) using {model}.",
            "language": language or "en",
            "duration": round(max(1.0, size_kb / 16.0), 2),
            "words": [],
        }


class OpenAISpeechToTextAdapter(BaseSpeechToTextAdapter):
    """Speech-to-text adapter targeting OpenAI-compatible Whisper endpoints."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.base_url = (
            os.getenv(config.get("base_url_env", "HOSTED_AI_BASE_URL"))
            or config.get("base_url")
            or "https://api.openai.com/v1"
        )
        self.api_key = (
            os.getenv(config.get("api_key_env", "HOSTED_AI_API_KEY"))
            or config.get("api_key")
        )
        self.timeout = float(config.get("timeout_seconds", 60.0))

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            # Fallback to mock adapter if upstream credentials are not configured
            mock = MockSpeechToTextAdapter(self.provider_id, self.config)
            return await mock.transcribe(audio_bytes, filename, model, language)

        url = f"{self.base_url.rstrip('/')}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": model}
        if language:
            data["language"] = language

        files = {"file": (filename, audio_bytes, "audio/mpeg")}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=headers, data=data, files=files)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                raise ProviderError(self.provider_id, f"STT upstream error: {exc}")
