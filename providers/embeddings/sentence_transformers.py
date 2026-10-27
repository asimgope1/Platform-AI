import asyncio
from threading import Lock
from typing import List, Dict, Any, Optional
from platform_core.contracts.providers import BaseEmbeddingAdapter
from platform_core.errors.exceptions import ProviderError


class SentenceTransformersEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for generating dense semantic embeddings using Sentence Transformers locally."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.model_name = (
            config.get("model")
            or config.get("model_id")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.dimension = int(config.get("dimension") or config.get("dimensions") or 384)
        self._model = None
        self._load_lock = Lock()

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self.model_name)
                    except Exception as exc:
                        raise ProviderError(
                            self.provider_id,
                            f"Failed to load SentenceTransformer model '{self.model_name}': {exc}"
                        )
        return self._model

    def warmup(self):
        """Pre-load model weights into RAM so first query takes 0.01s instead of 8s."""
        try:
            model = self._get_model()
            model.encode(["warmup"], convert_to_numpy=True)
        except Exception:
            pass

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    async def generate_embeddings(self, texts: List[str], model: str = "") -> List[List[float]]:
        if not texts:
            return []

        # Run model inference in thread pool to avoid blocking the async event loop
        return await asyncio.to_thread(self._encode_sync, texts)
