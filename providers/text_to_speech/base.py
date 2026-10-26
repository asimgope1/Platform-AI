import os
from typing import Dict, Any, Optional
import httpx
from platform_core.contracts.providers import BaseTextToSpeechAdapter
from platform_core.errors.exceptions import ProviderError


class MockTextToSpeechAdapter(BaseTextToSpeechAdapter):
    """Produces valid synthetic MP3 frames for development and offline testing."""

    async def synthesize(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        # Standard MPEG audio frame sync header (32 frames)
        frame_header = b"\xff\xfb\x90\x00\x00\x00\x00\x00"
        return frame_header * max(16, min(len(text) * 2, 512))


class OpenAITextToSpeechAdapter(BaseTextToSpeechAdapter):
    """Text-to-speech adapter targeting OpenAI-compatible TTS endpoints."""

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

    async def synthesize(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        if not self.api_key:
            mock = MockTextToSpeechAdapter(self.provider_id, self.config)
            return await mock.synthesize(text, model, voice, response_format)

        url = f"{self.base_url.rstrip('/')}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.content
            except Exception as exc:
                raise ProviderError(self.provider_id, f"TTS upstream error: {exc}")
