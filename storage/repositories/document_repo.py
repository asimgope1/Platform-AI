import math
import json
from typing import List, Optional, Tuple
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from platform_core.observability.logging import logger
from storage.models.document import Document, DocumentChunk


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class DocumentRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_document(
        self,
        tenant_id: str,
        title: str,
        source_type: str = "text",
    ) -> Document:
        doc = Document(tenant_id=tenant_id, title=title, source_type=source_type)
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def add_chunks(
        self,
        document_id: str,
        tenant_id: str,
        chunks_data: List[Tuple[int, str, List[float]]],
    ) -> List[DocumentChunk]:
        created_chunks = []
        for idx, text, emb in chunks_data:
            emb_384 = emb if len(emb) == 384 else None
            emb_768 = emb if len(emb) == 768 else None
            chunk = DocumentChunk(
                document_id=document_id,
                tenant_id=tenant_id,
                chunk_index=idx,
                content=text,
                embedding=emb_384,
                embedding_768=emb_768,
                embedding_json=json.dumps(emb),
            )
            self.session.add(chunk)
            created_chunks.append(chunk)

        # Update document count
        doc = await self.session.get(Document, document_id)
        if doc:
            doc.total_chunks = len(created_chunks)

        await self.session.commit()
        return created_chunks

    async def list_by_tenant(self, tenant_id: str) -> List[Document]:
        stmt = select(Document).where(Document.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, document_id: str, tenant_id: str) -> Optional[Document]:
        stmt = (
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id, Document.tenant_id == tenant_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_document(self, document_id: str, tenant_id: str) -> bool:
        doc = await self.get_by_id(document_id, tenant_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True

    async def search_similar_chunks(
        self,
        tenant_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        # Native pgvector cosine distance on PostgreSQL
        bind = self.session.bind
        is_postgres = bind and "postgresql" in str(bind.url)
        is_768 = len(query_embedding) == 768
        target_col = DocumentChunk.embedding_768 if is_768 else DocumentChunk.embedding

        if is_postgres:
            try:
                distance_expr = target_col.cosine_distance(query_embedding)
                stmt = (
                    select(DocumentChunk, (1.0 - distance_expr).label("score"))
                    .where(DocumentChunk.tenant_id == tenant_id, target_col != None)
                    .order_by(distance_expr.asc())
                    .limit(top_k)
                )
                res = await self.session.execute(stmt)
                rows = res.all()
                logger.info(f"Executed native pgvector cosine search for tenant {tenant_id}, found {len(rows)} matching chunks.")
                return [(row[0], round(float(row[1]), 4)) for row in rows]
            except Exception as e:
                logger.warning(f"Native pgvector search failed ({e}), falling back to in-memory cosine search.")

        # In-memory cosine similarity fallback (for SQLite unit tests)
        stmt = select(DocumentChunk).where(DocumentChunk.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        all_chunks = list(result.scalars().all())

        scored_chunks = []
        for chunk in all_chunks:
            chunk_emb = chunk.embedding_768 if is_768 else chunk.embedding
            if chunk_emb is None and chunk.embedding_json:
                chunk_emb = json.loads(chunk.embedding_json)
            if chunk_emb:
                score = cosine_similarity(query_embedding, chunk_emb)
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
