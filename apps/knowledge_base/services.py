"""
Knowledge Base Service for Com-one Platform
Performs real-time semantic retrieval over bilingual IVR & CRM knowledge chunks.
"""
import os
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost:5432/ai_platform_test")

class KnowledgeBaseService:
    _engine = None
    _session_factory = None
    _emb_service = None

    @classmethod
    def get_embedding_service(cls):
        if cls._emb_service is None:
            cls._emb_service = EmbeddingService(provider_id="google-embeddings")
        return cls._emb_service

    @classmethod
    def get_session_factory(cls):
        if not cls._session_factory:
            cls._engine = create_async_engine(POSTGRES_URL, echo=False)
            cls._session_factory = async_sessionmaker(cls._engine, class_=AsyncSession, expire_on_commit=False)
        return cls._session_factory

    @classmethod
    async def retrieve_ivr_context(
        cls,
        query: str,
        tenant_id: str = "879e9bda-8049-4824-9aa0-fc5e9f0012d9",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant IVR/CRM knowledge chunks using Google Embeddings + pgvector."""
        try:
            # 1. Embed query
            emb_service = cls.get_embedding_service()
            query_vector = await emb_service.embed_query(query)

            # 2. Search database
            session_factory = cls.get_session_factory()
            async with session_factory() as session:
                doc_repo = DocumentRepository(session)
                results = await doc_repo.search_similar_chunks(
                    tenant_id=tenant_id,
                    query_embedding=query_vector,
                    top_k=top_k,
                )

                retrieved = []
                for chunk, score in results:
                    retrieved.append({
                        "chunk_id": chunk.id,
                        "content": chunk.content,
                        "score": score,
                        "document_id": chunk.document_id,
                    })
                return retrieved
        except Exception as e:
            logger.warning(f"IVR context retrieval fallback: {e}")
            return []
