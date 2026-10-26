from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas.documents import (
    DocumentUploadRequest,
    DocumentResponse,
    DocumentListResponse,
    SearchRequestSchema,
    SearchResponseSchema,
    SearchResultItem,
)
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from storage.database import get_db_session
from storage.repositories.document_repo import DocumentRepository
from retrieval.ingestion import DocumentIngestionService
from retrieval.search import SemanticSearchService
from retrieval.embedding_service import EmbeddingService

router = APIRouter(prefix="/v1", tags=["Documents"])


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    payload: DocumentUploadRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Upload and index a document for RAG retrieval within tenant scope."""
    doc_repo = DocumentRepository(session)
    embedding_service = EmbeddingService()
    ingestion_service = DocumentIngestionService(doc_repo, embedding_service)

    doc = await ingestion_service.ingest_document(
        tenant_id=context.tenant_id,
        title=payload.title,
        content=payload.content,
        source_type=payload.source_type or "text",
    )

    return DocumentResponse(
        id=doc.id,
        tenant_id=doc.tenant_id,
        title=doc.title,
        source_type=doc.source_type,
        total_chunks=doc.total_chunks,
        created_at=doc.created_at,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """List all documents owned by the caller's tenant."""
    doc_repo = DocumentRepository(session)
    docs = await doc_repo.list_by_tenant(context.tenant_id)

    return DocumentListResponse(
        object="list",
        data=[
            DocumentResponse(
                id=d.id,
                tenant_id=d.tenant_id,
                title=d.title,
                source_type=d.source_type,
                total_chunks=d.total_chunks,
                created_at=d.created_at,
            )
            for d in docs
        ],
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Delete a document and all indexed chunks."""
    doc_repo = DocumentRepository(session)
    deleted = await doc_repo.delete_document(document_id, context.tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return {"id": document_id, "deleted": True}


@router.post("/documents/search", response_model=SearchResponseSchema)
async def search_documents(
    payload: SearchRequestSchema,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Run vector similarity search against tenant chunks."""
    doc_repo = DocumentRepository(session)
    embedding_service = EmbeddingService()
    search_service = SemanticSearchService(doc_repo, embedding_service)

    results = await search_service.search(
        tenant_id=context.tenant_id,
        query=payload.query,
        top_k=payload.top_k or 4,
    )

    return SearchResponseSchema(
        object="list",
        data=[
            SearchResultItem(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=round(score, 4),
            )
            for chunk, score in results
        ],
    )
