from typing import List, Tuple
from storage.repositories.document_repo import DocumentRepository
from storage.models.document import DocumentChunk
from retrieval.embedding_service import EmbeddingService


class SemanticSearchService:
    """Executes similarity searches across tenant document chunks."""

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
