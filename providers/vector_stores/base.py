import math
from typing import List, Dict, Any, Optional
from platform_core.contracts.providers import BaseVectorStoreAdapter


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class InMemoryVectorStore(BaseVectorStoreAdapter):
    """In-memory vector store with tenant partitioning and cosine similarity search."""

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        # Partitioned as: self._storage[tenant_id][chunk_id] = chunk_dict
        self._storage: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def upsert(
        self,
        tenant_id: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        if tenant_id not in self._storage:
            self._storage[tenant_id] = {}

        for chunk in chunks:
            chunk_id = chunk.get("id") or f"{document_id}_{chunk.get('chunk_index', 0)}"
            self._storage[tenant_id][chunk_id] = {
                "id": chunk_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "chunk_index": chunk.get("chunk_index", 0),
                "content": chunk.get("content", ""),
                "embedding": chunk.get("embedding", []),
                "metadata": chunk.get("metadata", {}),
            }

    async def query(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        tenant_items = self._storage.get(tenant_id, {})
        if not tenant_items:
            return []

        scored = []
        for item in tenant_items.values():
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append({**item, "score": round(score, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        if tenant_id in self._storage:
            keys_to_remove = [
                k for k, v in self._storage[tenant_id].items()
                if v.get("document_id") == document_id
            ]
            for k in keys_to_remove:
                del self._storage[tenant_id][k]
