from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from api.schemas.documents import (
    DocumentReindexRequest,
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
from retrieval.parsers import extract_document_text
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


@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Upload and index a multi-format knowledge file (.pdf, .docx, .txt, .md, .csv) for RAG."""
    raw_bytes = await file.read()
    filename = file.filename or "document.txt"
    source_type, extracted_text = extract_document_text(filename, raw_bytes)

    doc_repo = DocumentRepository(session)
    ingestion_service = DocumentIngestionService(doc_repo, EmbeddingService())
    doc = await ingestion_service.ingest_document(
        tenant_id=context.tenant_id,
        title=(title or filename).strip(),
        content=extracted_text,
        source_type=source_type,
    )
    return DocumentResponse(
        id=doc.id,
        tenant_id=doc.tenant_id,
        title=doc.title,
        source_type=doc.source_type,
        total_chunks=doc.total_chunks,
        created_at=doc.created_at,
    )


@router.post("/documents/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: str,
    payload: DocumentReindexRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Re-chunk and re-embed an updated version of an existing tenant document."""
    doc_repo = DocumentRepository(session)
    ingestion_service = DocumentIngestionService(doc_repo, EmbeddingService())
    doc = await ingestion_service.reindex_document(
        tenant_id=context.tenant_id,
        document_id=document_id,
        content=payload.content,
        title=payload.title,
        source_type=payload.source_type,
        version=payload.version or "2.0",
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return DocumentResponse(
        id=doc.id,
        tenant_id=doc.tenant_id,
        title=doc.title,
        source_type=doc.source_type,
        total_chunks=doc.total_chunks,
        created_at=doc.created_at,
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
    """Run hybrid or semantic similarity search against tenant chunks with source citations."""
    doc_repo = DocumentRepository(session)
    embedding_service = EmbeddingService()
    search_service = SemanticSearchService(doc_repo, embedding_service)

    top_k = payload.top_k or 4
    if payload.hybrid is False:
        results = await search_service.search(tenant_id=context.tenant_id, query=payload.query, top_k=top_k)
    else:
        results = await search_service.hybrid_search(
            tenant_id=context.tenant_id,
            query=payload.query,
            top_k=top_k,
            preferred_category=payload.category,
        )

    return SearchResponseSchema(
        object="list",
        data=[
            SearchResultItem(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=round(score, 4),
                citation=f"doc:{chunk.document_id}#chunk-{chunk.chunk_index}",
            )
            for chunk, score in results
        ],
    )
