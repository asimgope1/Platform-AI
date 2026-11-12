from typing import Dict, List, Optional, Tuple
from storage.repositories.document_repo import DocumentRepository
from storage.models.document import Document
from retrieval.chunking import chunk_text
from retrieval.embedding_service import EmbeddingService
from retrieval.parsers import assert_no_transactional_pii_dump


class DocumentIngestionService:
    """Ingests documents cleanly with PII filtering, chunking, embeddings, and re-indexing."""

    def __init__(self, doc_repo: DocumentRepository, embedding_service: EmbeddingService):
        self.doc_repo = doc_repo
        self.embedding_service = embedding_service

    async def ingest_document(
        self,
        tenant_id: str,
        title: str,
        content: str,
        source_type: str = "text",
        version: str = "1.0",
    ) -> Document:
        assert_no_transactional_pii_dump(content, filename=title)
        doc = await self.doc_repo.create_document(tenant_id=tenant_id, title=title, source_type=source_type)

        chunks = chunk_text(content, chunk_size=500, chunk_overlap=50)
        if not chunks:
            chunks = [content]

        embeddings = await self.embedding_service.embed_texts(chunks)
        chunks_data: List[Tuple[int, str, List[float]]] = [
            (idx, text, emb) for idx, (text, emb) in enumerate(zip(chunks, embeddings))
        ]

        await self.doc_repo.add_chunks(document_id=doc.id, tenant_id=tenant_id, chunks_data=chunks_data)
        return doc

    async def reindex_document(
        self,
        tenant_id: str,
        document_id: str,
        *,
        content: str,
        title: Optional[str] = None,
        source_type: Optional[str] = None,
        version: str = "2.0",
    ) -> Optional[Document]:
        assert_no_transactional_pii_dump(content, filename=title or document_id)
        chunks = chunk_text(content, chunk_size=500, chunk_overlap=50) or [content]
        embeddings = await self.embedding_service.embed_texts(chunks)
        chunks_data: List[Tuple[int, str, List[float]]] = [
            (idx, text, emb) for idx, (text, emb) in enumerate(zip(chunks, embeddings))
        ]
        return await self.doc_repo.replace_chunks(
            document_id=document_id,
            tenant_id=tenant_id,
            chunks_data=chunks_data,
            new_title=title,
            new_source_type=source_type,
            meta={"doc_version": version, "verified": True},
        )
