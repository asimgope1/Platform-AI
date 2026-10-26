import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from platform_core.contracts.providers import BaseEmbeddingAdapter
from providers.embeddings.sentence_transformers import SentenceTransformersEmbeddingAdapter

logger = logging.getLogger(__name__)


class GoogleGenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Google Gemini & GenAI Embedding Adapter supporting text-embedding-004.
    
    Supports dynamic output_dimensionality (e.g. 384 for compatibility with pgvector)
    and automatic graceful fallback to local multilingual SentenceTransformers
    if GEMINI_API_KEY is not configured or in offline environments.
    """

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.model_name = (
            config.get("model")
            or config.get("model_id")
            or "text-embedding-004"
        )
        self.dimension = int(config.get("dimension") or config.get("dimensions") or 384)
        self.api_key = (
            config.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.fallback_model = config.get("fallback_model") or "sentence-transformers/all-MiniLM-L6-v2"
        self._client = None
        self._fallback_adapter = None

    def _get_fallback_adapter(self) -> SentenceTransformersEmbeddingAdapter:
        if self._fallback_adapter is None:
            self._fallback_adapter = SentenceTransformersEmbeddingAdapter(
                f"{self.provider_id}_fallback",
                {"model": self.fallback_model, "dimension": self.dimension},
            )
        return self._fallback_adapter

    def _get_genai_client(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                logger.warning(f"Could not initialize Google GenAI Client: {exc}")
                return None
        return self._client

    def _embed_with_google(self, texts: List[str], model: str) -> List[List[float]]:
        from google.genai import types
        client = self._get_genai_client()
        if not client:
            raise ValueError("No Google GenAI client available")

        # Google embed_content supports batching
        config = types.EmbedContentConfig(output_dimensionality=self.dimension)
        
        # Call Google GenAI Embed API
        response = client.models.embed_content(
            model=model or self.model_name,
            contents=texts,
            config=config,
        )

        vectors = []
        if hasattr(response, "embeddings") and response.embeddings:
            for emb in response.embeddings:
                vectors.append(list(emb.values))
        elif hasattr(response, "embedding") and response.embedding:
            vectors.append(list(response.embedding.values))

        return vectors

    async def generate_embeddings(self, texts: List[str], model: str = "") -> List[List[float]]:
        if not texts:
            return []

        target_model = model or self.model_name

        # If API key is available, attempt Google GenAI embedding
        if self.api_key:
            try:
                vectors = await asyncio.to_thread(self._embed_with_google, texts, target_model)
                if vectors and len(vectors) == len(texts):
                    return vectors
            except Exception as exc:
                logger.warning(
                    f"Google GenAI embedding failed ({exc}). Falling back to local multilingual embedding model."
                )

        # Fallback to local embedding adapter
        fallback = self._get_fallback_adapter()
        return await fallback.generate_embeddings(texts, self.fallback_model)
