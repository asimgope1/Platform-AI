from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ModelAliasConfig(BaseModel):
    provider: str
    model: str
    description: Optional[str] = ""
    capabilities: List[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    type: str  # e.g., 'openai_compatible', 'mock', 'transformers', 'gemma_transformers'
    model: Optional[str] = None
    base_url_env: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: float = 60.0
    device: Optional[str] = None
    trust_remote_code: Optional[bool] = False


class EmbeddingsConfig(BaseModel):
    provider: str
    model: str
    dimension: int = 1536


class VectorStoreConfig(BaseModel):
    provider: str = "pgvector"
    collection: str = "knowledge_chunks"
    dimensions: Optional[int] = 384
    distance_metric: Optional[str] = "cosine"


class ProvidersConfigFile(BaseModel):
    model_aliases: Dict[str, ModelAliasConfig]
    providers: Dict[str, ProviderConfig]
    embeddings: Optional[Dict[str, EmbeddingsConfig]] = None
    speech_to_text: Optional[Dict[str, Any]] = None
    text_to_speech: Optional[Dict[str, Any]] = None
    vector_store: Optional[Dict[str, VectorStoreConfig]] = None
