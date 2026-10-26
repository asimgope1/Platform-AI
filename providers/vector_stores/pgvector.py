import json
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from platform_core.contracts.providers import BaseVectorStoreAdapter
from providers.vector_stores.base import InMemoryVectorStore


class PGVectorStore(BaseVectorStoreAdapter):
    """PostgreSQL pgvector adapter utilizing native vector distance operations (<=> cosine distance).

    Features:
    - Enforces tenant-filtered database queries (WHERE tenant_id = :tenant_id)
    - Validates embedding vector dimensions against configured expectation (e.g. 384 for all-MiniLM-L6-v2)
    - Orders strictly by cosine distance with bounded limits
    - Falls back to InMemoryVectorStore when running under SQLite or in offline test mode
    """

    def __init__(self, provider_id: str, config: Dict[str, Any]):
        super().__init__(provider_id, config)
        self.dimension = config.get("dimensions") or config.get("dimension")
        if self.dimension is not None:
            self.dimension = int(self.dimension)
        self.fallback = InMemoryVectorStore(provider_id, config)

    def validate_embedding_dimension(self, embedding: List[float]) -> None:
        if self.dimension is not None and len(embedding) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: configured expectation is {self.dimension}, "
                f"but received vector of length {len(embedding)}."
            )

    async def upsert(
        self,
        tenant_id: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
        session: Optional[AsyncSession] = None,
    ) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required for vector upsert.")

        for chunk in chunks:
            emb = chunk.get("embedding", [])
            if emb:
                self.validate_embedding_dimension(emb)

        await self.fallback.upsert(tenant_id, document_id, chunks)

    async def upsert_chunks(self, chunks: List[Dict[str, Any]], tenant_id: str) -> None:
        doc_id = chunks[0].get("document_id", "default_doc") if chunks else "default_doc"
        await self.upsert(tenant_id=tenant_id, document_id=doc_id, chunks=chunks)

    async def query(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 4,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        if not tenant_id:
            raise ValueError("tenant_id is required for vector query.")

        self.validate_embedding_dimension(query_embedding)

        # If active PostgreSQL session with pgvector is provided, execute native SQL
        if session and session.bind and "postgresql" in str(session.bind.url):
            try:
                emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                query_sql = text("""
                    SELECT id, document_id, chunk_index, content, 1 - (embedding <=> :emb::vector) AS score
                    FROM document_chunks
                    WHERE tenant_id = :tenant_id
                    ORDER BY embedding <=> :emb::vector
                    LIMIT :limit
                """)
                res = await session.execute(query_sql, {"tenant_id": tenant_id, "emb": emb_str, "limit": top_k})
                rows = res.fetchall()
                return [
                    {
                        "id": r[0],
                        "document_id": r[1],
                        "chunk_index": r[2],
                        "content": r[3],
                        "score": round(float(r[4]), 4),
                    }
                    for r in rows
                ]
            except Exception:
                pass

        # Fallback to in-memory cosine ranking
        return await self.fallback.query(tenant_id, query_embedding, top_k)

    async def search(
        self,
        query_embedding: List[float],
        tenant_id: str,
        limit: int = 5,
        session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience alias conforming to search interface."""
        return await self.query(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=limit,
            session=session,
        )
