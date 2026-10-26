from typing import AsyncIterator, Dict, Any
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import NormalizedChatRequest, NormalizedChatResponse, StreamChunk
from platform_core.contracts.providers import BaseChatAdapter
from platform_core.errors.exceptions import ProviderError


class TransformersLocalChatAdapter(BaseChatAdapter):
    """Adapter for running local HuggingFace / Gemma models on GPU/CPU."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.device = config.get("device", "auto")
        self.trust_remote_code = config.get("trust_remote_code", False)

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
        except ImportError:
            raise ProviderError(
                self.provider_id,
                "Local Transformers adapter requires 'torch' and 'transformers'. Please install them on your GPU environment.",
            )

        raise ProviderError(
            self.provider_id,
            f"Local model '{model_meta.underlying_model}' is configured. Please provide local model weights or run in GPU container.",
        )

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> AsyncIterator[StreamChunk]:
        raise ProviderError(
            self.provider_id,
            "Streaming is not yet enabled for the local transformers adapter.",
        )
        yield  # make it an async generator
