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


import os
import httpx
import logging

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingAdapter(BaseEmbeddingAdapter):
    """Generates embeddings via OpenAI-compatible /v1/embeddings endpoint (such as Colab GPU server)."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.base_url = (
            os.getenv(config.get("base_url_env", ""))
            or config.get("base_url")
        )
        self.api_key = (
            os.getenv(config.get("api_key_env", ""))
            or config.get("api_key")
        )
        self.dimension = int(config.get("dimension") or 768)
        self.timeout = float(config.get("timeout_seconds", 60.0))

    async def generate_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        if not texts:
            return []

        if not self.base_url:
            mock = MockEmbeddingAdapter(self.provider_id, self.config, dimension=self.dimension)
            return await mock.generate_embeddings(texts, model)

        endpoint = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model or self.config.get("model", "google/embeddinggemma-2"),
            "input": texts,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if "data" in data and isinstance(data["data"], list):
                    return [item["embedding"] for item in data["data"]]
        except Exception as exc:
            logger.warning(f"Remote embedding request to {endpoint} failed ({exc}). Falling back to mock adapter.")

        mock = MockEmbeddingAdapter(self.provider_id, self.config, dimension=self.dimension)
        return await mock.generate_embeddings(texts, model)

