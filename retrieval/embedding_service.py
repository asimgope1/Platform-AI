from typing import List, Optional
from providers.registry import provider_registry


class EmbeddingService:
    """Service generating vector embeddings using the configured embedding adapter."""

    def __init__(self, provider_id: Optional[str] = None, model: Optional[str] = None):
        if provider_id is None:
            if model == "text-embedding-3-small":
                provider_id = "default"
            elif "semantic-default" in provider_registry.embedding_adapters:
                provider_id = "semantic-default"
                model = model or "sentence-transformers/all-MiniLM-L6-v2"
            else:
                provider_id = "default"
                model = model or "text-embedding-3-small"

        self.provider_id = provider_id
        self.model = model or "text-embedding-3-small"

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        adapter = provider_registry.get_embedding_adapter(self.provider_id)
        return await adapter.generate_embeddings(texts, self.model)

    async def embed_query(self, query: str) -> List[float]:
        res = await self.embed_texts([query])
        return res[0] if res else []
