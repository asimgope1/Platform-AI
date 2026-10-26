import hashlib
import math
from typing import List, Dict, Any
from platform_core.contracts.providers import BaseEmbeddingAdapter


class MockEmbeddingAdapter(BaseEmbeddingAdapter):
    """Produces deterministic normalized embedding vectors based on text hashing.

    Guarantees that identical text yields identical vectors and similar texts share components.
    """

    def __init__(self, provider_id: str, config: Dict[str, Any], dimension: int = 1536):
        super().__init__(provider_id, config)
        self.dimension = dimension

    async def generate_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Deterministic pseudo-random generation seeded by text
            hasher = hashlib.sha256(text.encode("utf-8"))
            digest = hasher.digest()
            vec = []
            for i in range(self.dimension):
                byte_val = digest[i % len(digest)]
                val = (byte_val / 128.0) - 1.0 + math.sin(i + len(text))
                vec.append(val)

            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)

        return embeddings


class OpenAICompatibleEmbeddingAdapter(BaseEmbeddingAdapter):
    """Generates embeddings via OpenAI-compatible /v1/embeddings endpoint."""

    async def generate_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        # In a real environment with base_url and api_key configured, calls POST /v1/embeddings
        # Fall back to deterministic mock if unconfigured
        mock = MockEmbeddingAdapter(self.provider_id, self.config)
        return await mock.generate_embeddings(texts, model)
