from typing import Dict, Any, Optional, List
from platform_core.contracts.capabilities import ModelMetadata, ModelCapability
from platform_core.contracts.providers import (
    BaseChatAdapter,
    BaseEmbeddingAdapter,
    BaseSpeechToTextAdapter,
    BaseTextToSpeechAdapter,
    BaseVectorStoreAdapter,
    BaseRerankerAdapter,
)
from platform_core.config.loader import load_providers_config, build_model_metadata_map
from platform_core.config.validation import ProvidersConfigFile
from platform_core.errors.exceptions import ModelNotFoundError, ProviderError
from providers.chat.mock import MockChatAdapter
from providers.chat.openai_compatible import OpenAICompatibleChatAdapter
from providers.chat.transformers_local import TransformersLocalChatAdapter
from providers.chat.gemma_transformers import GemmaTransformersChatAdapter
from providers.embeddings.base import MockEmbeddingAdapter, OpenAICompatibleEmbeddingAdapter
from providers.embeddings.sentence_transformers import SentenceTransformersEmbeddingAdapter
from providers.embeddings.google_genai import GoogleGenAIEmbeddingAdapter
from providers.speech_to_text.base import MockSpeechToTextAdapter, OpenAISpeechToTextAdapter
from providers.text_to_speech.base import MockTextToSpeechAdapter, OpenAITextToSpeechAdapter
from providers.vector_stores.base import InMemoryVectorStore
from providers.vector_stores.pgvector import PGVectorStore
from providers.rerankers.base import MockReranker


class ProviderRegistry:
    """Central registry managing model aliases, provider instances, capabilities, and peripheral services."""

    def __init__(self):
        self.chat_adapters: Dict[str, BaseChatAdapter] = {}
        self.embedding_adapters: Dict[str, BaseEmbeddingAdapter] = {}
        self.stt_adapters: Dict[str, BaseSpeechToTextAdapter] = {}
        self.tts_adapters: Dict[str, BaseTextToSpeechAdapter] = {}
        self.vector_stores: Dict[str, BaseVectorStoreAdapter] = {}
        self.rerankers: Dict[str, BaseRerankerAdapter] = {}
        self.models: Dict[str, ModelMetadata] = {}
        self.config: Optional[ProvidersConfigFile] = None

        # Seed defaults
        self._seed_default_adapters()

    def _seed_default_adapters(self) -> None:
        self.embedding_adapters["mock_embeddings"] = MockEmbeddingAdapter("mock_embeddings", {})
        self.embedding_adapters["sentence_transformers"] = SentenceTransformersEmbeddingAdapter("sentence_transformers", {})
        self.embedding_adapters["google_genai"] = GoogleGenAIEmbeddingAdapter("google_genai", {"dimension": 384})
        self.embedding_adapters["default"] = self.embedding_adapters["google_genai"]

        self.stt_adapters["mock_stt"] = MockSpeechToTextAdapter("mock_stt", {})
        self.stt_adapters["openai_stt"] = OpenAISpeechToTextAdapter("openai_stt", {})
        self.stt_adapters["default"] = self.stt_adapters["mock_stt"]

        self.tts_adapters["mock_tts"] = MockTextToSpeechAdapter("mock_tts", {})
        self.tts_adapters["openai_tts"] = OpenAITextToSpeechAdapter("openai_tts", {})
        self.tts_adapters["default"] = self.tts_adapters["mock_tts"]

        self.vector_stores["in_memory"] = InMemoryVectorStore("in_memory", {})
        self.vector_stores["pgvector"] = PGVectorStore("pgvector", {})
        self.vector_stores["default"] = self.vector_stores["pgvector"]

        self.rerankers["mock_reranker"] = MockReranker("mock_reranker", {})
        self.rerankers["default"] = self.rerankers["mock_reranker"]

    def initialize_from_file(self, config_path: str) -> None:
        cfg = load_providers_config(config_path)
        self.initialize_from_config(cfg)

    def initialize_from_config(self, cfg: ProvidersConfigFile) -> None:
        self.config = cfg
        self.models = build_model_metadata_map(cfg)

        # Seed defaults first so explicit configurations override or supplement
        self._seed_default_adapters()

        # Initialize chat adapters
        for p_id, p_cfg in cfg.providers.items():
            adapter_type = p_cfg.type.lower()
            if adapter_type == "mock":
                self.chat_adapters[p_id] = MockChatAdapter(p_id, p_cfg.model_dump())
            elif adapter_type == "openai_compatible":
                self.chat_adapters[p_id] = OpenAICompatibleChatAdapter(p_id, p_cfg.model_dump())
            elif adapter_type == "gemma_transformers":
                self.chat_adapters[p_id] = GemmaTransformersChatAdapter(p_id, p_cfg.model_dump())
            elif adapter_type == "transformers":
                self.chat_adapters[p_id] = TransformersLocalChatAdapter(p_id, p_cfg.model_dump())
            else:
                self.chat_adapters[p_id] = MockChatAdapter(p_id, p_cfg.model_dump())

        # Initialize configured embedding adapters
        if cfg.embeddings:
            for e_id, e_cfg in cfg.embeddings.items():
                p_name = e_cfg.provider.lower()
                if p_name in ("google_genai", "google_embeddings", "google"):
                    self.embedding_adapters[e_id] = GoogleGenAIEmbeddingAdapter(e_id, e_cfg.model_dump())
                elif p_name == "sentence_transformers":
                    self.embedding_adapters[e_id] = SentenceTransformersEmbeddingAdapter(e_id, e_cfg.model_dump())
                elif p_name == "mock_embeddings":
                    self.embedding_adapters[e_id] = MockEmbeddingAdapter(e_id, e_cfg.model_dump(), dimension=e_cfg.dimension)

        # Initialize configured vector stores
        if cfg.vector_store:
            for vs_id, vs_cfg in cfg.vector_store.items():
                p_name = vs_cfg.provider.lower()
                if p_name == "pgvector":
                    self.vector_stores[vs_id] = PGVectorStore(vs_id, vs_cfg.model_dump())
                elif p_name == "in_memory":
                    self.vector_stores[vs_id] = InMemoryVectorStore(vs_id, vs_cfg.model_dump())

    def get_model_metadata(self, model_alias: str) -> ModelMetadata:
        if model_alias not in self.models:
            raise ModelNotFoundError(model_alias)
        return self.models[model_alias]

    def get_chat_adapter(self, provider_id: str) -> BaseChatAdapter:
        if provider_id not in self.chat_adapters:
            raise ProviderError(provider_id, f"Provider '{provider_id}' is not registered.")
        return self.chat_adapters[provider_id]

    def get_embedding_adapter(self, provider_id: str = "default") -> BaseEmbeddingAdapter:
        return self.embedding_adapters.get(provider_id, self.embedding_adapters["default"])

    def get_stt_adapter(self, provider_id: str = "default") -> BaseSpeechToTextAdapter:
        return self.stt_adapters.get(provider_id, self.stt_adapters["default"])

    def get_tts_adapter(self, provider_id: str = "default") -> BaseTextToSpeechAdapter:
        return self.tts_adapters.get(provider_id, self.tts_adapters["default"])

    def get_vector_store(self, provider_id: str = "default") -> BaseVectorStoreAdapter:
        return self.vector_stores.get(provider_id, self.vector_stores["default"])

    def get_reranker(self, provider_id: str = "default") -> BaseRerankerAdapter:
        return self.rerankers.get(provider_id, self.rerankers["default"])

    def list_models(self) -> List[ModelMetadata]:
        return list(self.models.values())

    def register_model_alias(self, metadata: ModelMetadata, adapter: Optional[BaseChatAdapter] = None) -> None:
        self.models[metadata.alias] = metadata
        if adapter:
            self.chat_adapters[metadata.provider_id] = adapter


# Global registry instance
provider_registry = ProviderRegistry()
