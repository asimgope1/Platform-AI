from typing import List, Optional, Tuple
from storage.repositories.document_repo import DocumentRepository
from storage.models.document import DocumentChunk
from retrieval.embedding_service import EmbeddingService


class SemanticSearchService:
    """Executes similarity and hybrid searches across tenant document chunks."""

    def __init__(self, doc_repo: DocumentRepository, embedding_service: EmbeddingService):
        self.doc_repo = doc_repo
        self.embedding_service = embedding_service

    async def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        query_embedding = await self.embedding_service.embed_query(query)
        return await self.doc_repo.search_similar_chunks(tenant_id=tenant_id, query_embedding=query_embedding, top_k=top_k)

    async def hybrid_search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 4,
        preferred_category: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Execute semantic search over a wider candidate pool and rerank with lexical/metadata signals."""
        from services.knowledge.retrieval_service import HybridRetrievalService

        raw_results = await self.search(tenant_id=tenant_id, query=query, top_k=max(top_k * 3, 12))
        return HybridRetrievalService.hybrid_rerank(
            query=query,
            raw_results=raw_results,
            preferred_category=preferred_category,
            top_k=top_k,
        )
