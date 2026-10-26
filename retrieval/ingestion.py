from typing import List, Tuple
from storage.repositories.document_repo import DocumentRepository
from storage.models.document import Document
from retrieval.chunking import chunk_text
from retrieval.embedding_service import EmbeddingService


class DocumentIngestionService:
    """Ingests raw text documents, chunks them, generates vector embeddings, and persists them."""

    def __init__(self, doc_repo: DocumentRepository, embedding_service: EmbeddingService):
        self.doc_repo = doc_repo
        self.embedding_service = embedding_service

    async def ingest_document(
        self,
        tenant_id: str,
        title: str,
        content: str,
        source_type: str = "text",
    ) -> Document:
        # Create parent document record
        doc = await self.doc_repo.create_document(tenant_id=tenant_id, title=title, source_type=source_type)

        # Chunk content
        chunks = chunk_text(content, chunk_size=500, chunk_overlap=50)
        if not chunks:
            chunks = [content]

        # Generate embeddings in batch
        embeddings = await self.embedding_service.embed_texts(chunks)

        # Pair chunks with embeddings
        chunks_data: List[Tuple[int, str, List[float]]] = [
            (idx, text, emb) for idx, (text, emb) in enumerate(zip(chunks, embeddings))
        ]

        # Save to database
        await self.doc_repo.add_chunks(document_id=doc.id, tenant_id=tenant_id, chunks_data=chunks_data)
        return doc
