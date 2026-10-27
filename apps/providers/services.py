"""
Unified Providers Bridge for Django AI Platform
Connects Django apps directly to swappable LLM, STT, TTS, and Embedding engines.
"""
import logging
from typing import List, Dict, Any, Optional
from platform_core.contracts.requests import NormalizedChatRequest, ChatMessage
from providers.registry import provider_registry

logger = logging.getLogger(__name__)

class ProviderBridge:
    _initialized = False

    @classmethod
    def ensure_initialized(cls):
        if not cls._initialized:
            try:
                provider_registry.initialize_from_file("config/providers.yaml")
                cls._initialized = True
                logger.info("ProviderBridge successfully initialized providers from config/providers.yaml")
            except Exception as e:
                logger.warning(f"ProviderBridge config load fallback: {e}")

    @classmethod
    async def generate_chat_response(
        cls,
        messages: List[Dict[str, str]],
        model_alias: str = "gemma-gpu",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate response via configured model (e.g. google/gemma-4-E2B-it or local smoke)."""
        cls.ensure_initialized()
        try:
            if model_alias in provider_registry.models:
                meta = provider_registry.get_model_metadata(model_alias)
                adapter = provider_registry.get_chat_adapter(meta.provider_id)
            else:
                meta = provider_registry.get_model_metadata("mock-chat")
                adapter = provider_registry.get_chat_adapter(meta.provider_id)

            req = NormalizedChatRequest(
                model_alias=model_alias,
                messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            res = await adapter.chat_completion(req, meta)
            content = res.choices[0].message.content if res.choices else ""
            return {
                "content": content,
                "model": res.model,
                "finish_reason": res.choices[0].finish_reason if res.choices else "stop",
                "prompt_tokens": res.usage.prompt_tokens,
                "completion_tokens": res.usage.completion_tokens,
            }
        except Exception as e:
            logger.warning(f"Chat model {model_alias} failed ({e}), falling back to mock provider.")
            try:
                meta = provider_registry.get_model_metadata("mock-chat")
                adapter = provider_registry.get_chat_adapter(meta.provider_id)
                req = NormalizedChatRequest(
                    model_alias="mock-chat",
                    messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                    max_tokens=max_tokens,
                )
                res = await adapter.chat_completion(req, meta)
                content = res.choices[0].message.content if res.choices else ""
                return {
                    "content": content,
                    "model": "fallback-mock",
                    "finish_reason": "stop",
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                }
            except Exception:
                return {
                    "content": "Customer support representative available. How can I assist you?",
                    "model": "static-fallback",
                    "finish_reason": "stop",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

    @classmethod
    async def transcribe_audio(cls, audio_bytes: bytes, filename: str = "caller_audio.wav") -> Dict[str, Any]:
        """Perceive Stage: Transcribe caller audio bytes with WhisperLocalSpeechToTextAdapter."""
        cls.ensure_initialized()
        stt_adapter = provider_registry.get_stt_adapter("whisper_local")
        return await stt_adapter.transcribe(audio_bytes=audio_bytes, filename=filename)

    @classmethod
    async def synthesize_speech(cls, text: str, voice: str = "alloy") -> bytes:
        """Respond Stage: Synthesize spoken response via TTS adapter."""
        cls.ensure_initialized()
        tts_adapter = provider_registry.get_tts_adapter("mock_tts")
        audio_data = await tts_adapter.synthesize(text=text, voice=voice, response_format="wav")
        if isinstance(audio_data, bytes):
            return audio_data
        elif hasattr(audio_data, "audio_bytes"):
            return audio_data.audio_bytes
        return bytes(audio_data)
