from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict, Any, Optional
from platform_core.contracts.capabilities import ModelCapability, ModelMetadata
from platform_core.contracts.requests import NormalizedChatRequest, NormalizedChatResponse, StreamChunk


class BaseChatAdapter(ABC):
    """Abstract interface for chat model providers."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config

    @abstractmethod
    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        """Executes a non-streaming chat completion request."""
        pass

    @abstractmethod
    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> AsyncIterator[StreamChunk]:
        """Executes a streaming chat completion request."""
        pass


class BaseEmbeddingAdapter(ABC):
    """Abstract interface for embedding providers."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str,
    ) -> List[List[float]]:
        """Generates vector embeddings for a list of texts."""
        pass


class BaseSpeechToTextAdapter(ABC):
    """Abstract interface for Speech-to-Text (STT) providers."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribes speech audio into text."""
        pass


class BaseTextToSpeechAdapter(ABC):
    """Abstract interface for Text-to-Speech (TTS) providers."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
    ) -> bytes:
        """Synthesizes text input into audio bytes."""
        pass


class BaseVectorStoreAdapter(ABC):
    """Abstract interface for vector database storage and similarity search."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config

    @abstractmethod
    async def upsert(
        self,
        tenant_id: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Upsert embedded document chunks."""
        pass

    @abstractmethod
    async def query(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Execute nearest-neighbor vector search."""
        pass


class BaseRerankerAdapter(ABC):
    """Abstract interface for reranking retrieved documents."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.config = config

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates based on semantic relevance to query."""
        pass

