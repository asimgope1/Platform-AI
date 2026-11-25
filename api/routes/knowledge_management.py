"""
Dynamic AI Knowledge Management API
===================================
Provides tenant-isolated REST endpoints for:
  - Dashboard metrics
  - Dynamic Category Management (CRUD + hierarchy)
  - Asynchronous Document Upload & Ingestion Jobs (PDF/DOCX/TXT/CSV)
  - Document Review, Publish & Archive lifecycle
  - RAG Playground & Search Testing
  - AI / RAG Configuration
  - Audit Logs
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.security.authentication import get_current_tenant
from storage.database import get_db_session, async_session
from platform_core.security.tenant_context import TenantContext
from retrieval.chunking import chunk_text
from retrieval.embedding_service import EmbeddingService
from retrieval.parsers import extract_document_text
from services.knowledge.retrieval_service import HybridRetrievalService
from storage.database import async_session
from storage.models.document import Document, DocumentChunk
from storage.models.knowledge import (
    KnowledgeAuditLog,
    KnowledgeCategory,
    KnowledgeConfiguration,
    KnowledgeDocumentVersion,
    KnowledgeIngestionJob,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Management"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class CategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    slug: str = Field(..., min_length=1, max_length=120)
    domain: str = Field(default="citizen_services", max_length=80)
    description: Optional[str] = None
    parent_id: Optional[str] = None
    supported_languages: List[str] = Field(default_factory=lambda: ["en", "hi", "or"])
    keywords: List[str] = Field(default_factory=list)
    is_active: bool = True


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None
    supported_languages: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    is_active: Optional[bool] = None


class ConfigUpdate(BaseModel):
    chunk_size: Optional[int] = Field(default=None, ge=100, le=4000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=0.99)
    reranking_enabled: Optional[bool] = None
    supported_languages: Optional[List[str]] = None


class PlaygroundRequest(BaseModel):
    query: str = Field(..., min_length=1)
    language: str = "en"
    category_id: Optional[str] = None
    category_slug: Optional[str] = None
    top_k: int = 4


async def _log_audit(
    session: AsyncSession,
    tenant_id: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    actor: str = "admin",
) -> None:
    session.add(
        KnowledgeAuditLog(
            tenant_id=tenant_id,
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details_json=json.dumps(details or {}),
        )
    )


# ── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    tid = context.tenant_id
    doc_count = await session.scalar(select(func.count()).select_from(Document).where(Document.tenant_id == tid)) or 0
    chunk_count = await session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.tenant_id == tid)) or 0
    cat_count = await session.scalar(select(func.count()).select_from(KnowledgeCategory).where(KnowledgeCategory.tenant_id == tid, KnowledgeCategory.is_active == True)) or 0
    job_count = await session.scalar(select(func.count()).select_from(KnowledgeIngestionJob).where(KnowledgeIngestionJob.tenant_id == tid)) or 0
    failed_jobs = await session.scalar(select(func.count()).select_from(KnowledgeIngestionJob).where(KnowledgeIngestionJob.tenant_id == tid, KnowledgeIngestionJob.status == "FAILED")) or 0

    recent_docs_res = await session.execute(
        select(Document).where(Document.tenant_id == tid).order_by(Document.created_at.desc()).limit(8)
    )
    recent_docs = [
        {
            "id": d.id,
            "title": d.title,
            "source_type": d.source_type,
            "total_chunks": d.total_chunks,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in recent_docs_res.scalars().all()
    ]

    return {
        "tenant_id": tid,
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "active_categories": cat_count,
        "total_jobs": job_count,
        "failed_jobs": failed_jobs,
        "supported_languages": ["en", "hi", "or"],
        "recent_documents": recent_docs,
    }


# ── Categories CRUD ──────────────────────────────────────────────────────────

@router.get("/categories")
async def list_categories(
    domain: Optional[str] = None,
    include_inactive: bool = False,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    q = select(KnowledgeCategory).where(KnowledgeCategory.tenant_id == context.tenant_id)
    if domain:
        q = q.where(KnowledgeCategory.domain == domain)
    if not include_inactive:
        q = q.where(KnowledgeCategory.is_active == True)
    q = q.order_by(KnowledgeCategory.name.asc())
    rows = (await session.execute(q)).scalars().all()
    return [
        {
            "id": c.id,
            "tenant_id": c.tenant_id,
            "name": c.name,
            "slug": c.slug,
            "domain": c.domain,
            "description": c.description,
            "parent_id": c.parent_id,
            "supported_languages": json.loads(c.supported_languages or '["en","hi","or"]'),
            "keywords": json.loads(c.keywords or "[]"),
            "is_active": c.is_active,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in rows
    ]


@router.post("/categories", status_code=status.HTTP_201_CREATED)
async def create_category(
    payload: CategoryCreate,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    existing = await session.scalar(
        select(KnowledgeCategory).where(
            KnowledgeCategory.tenant_id == context.tenant_id,
            KnowledgeCategory.slug == payload.slug.strip().lower(),
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"Category slug '{payload.slug}' already exists.")

    cat = KnowledgeCategory(
        tenant_id=context.tenant_id,
        name=payload.name.strip(),
        slug=payload.slug.strip().lower(),
        domain=payload.domain.strip(),
        description=payload.description,
        parent_id=payload.parent_id,
        supported_languages=json.dumps(payload.supported_languages),
        keywords=json.dumps(payload.keywords),
        is_active=payload.is_active,
    )
    session.add(cat)
    await _log_audit(session, context.tenant_id, "CREATE_CATEGORY", "category", cat.id, {"slug": cat.slug})
    await session.commit()
    await session.refresh(cat)
    return {"id": cat.id, "name": cat.name, "slug": cat.slug, "domain": cat.domain, "is_active": cat.is_active}


@router.patch("/categories/{category_id}")
async def update_category(
    category_id: str,
    payload: CategoryUpdate,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    cat = await session.scalar(
        select(KnowledgeCategory).where(KnowledgeCategory.id == category_id, KnowledgeCategory.tenant_id == context.tenant_id)
    )
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found.")

    if payload.name is not None:
        cat.name = payload.name.strip()
    if payload.domain is not None:
        cat.domain = payload.domain.strip()
    if payload.description is not None:
        cat.description = payload.description
    if payload.parent_id is not None:
        cat.parent_id = payload.parent_id
    if payload.supported_languages is not None:
        cat.supported_languages = json.dumps(payload.supported_languages)
    if payload.keywords is not None:
        cat.keywords = json.dumps(payload.keywords)
    if payload.is_active is not None:
        cat.is_active = payload.is_active

    await _log_audit(session, context.tenant_id, "UPDATE_CATEGORY", "category", cat.id, payload.model_dump(exclude_none=True))
    await session.commit()
    return {"id": cat.id, "name": cat.name, "slug": cat.slug, "is_active": cat.is_active}


@router.delete("/categories/{category_id}")
async def delete_category(
    category_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    cat = await session.scalar(
        select(KnowledgeCategory).where(KnowledgeCategory.id == category_id, KnowledgeCategory.tenant_id == context.tenant_id)
    )
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found.")
    cat.is_active = False
    await _log_audit(session, context.tenant_id, "ARCHIVE_CATEGORY", "category", cat.id, {"slug": cat.slug})
    await session.commit()
    return {"id": cat.id, "status": "archived"}


# ── Async Document Upload & Ingestion Worker ─────────────────────────────────

async def _run_ingestion_job(
    job_id: str,
    tenant_id: str,
    filename: str,
    raw_bytes: bytes,
    category_id: Optional[str],
    category_slug: str,
    language: str,
    source: str,
    source_version: str,
    effective_date: str,
    verified: bool,
) -> None:
    async with async_session() as session:
        job = await session.scalar(select(KnowledgeIngestionJob).where(KnowledgeIngestionJob.id == job_id))
        if not job:
            return
        try:
            job.status = "PROCESSING"
            await session.commit()

            cfg = await session.scalar(
                select(KnowledgeConfiguration).where(
                    KnowledgeConfiguration.tenant_id == tenant_id,
                    KnowledgeConfiguration.category_id == category_id,
                )
            ) or await session.scalar(
                select(KnowledgeConfiguration).where(
                    KnowledgeConfiguration.tenant_id == tenant_id,
                    KnowledgeConfiguration.category_id.is_(None),
                )
            )
            c_size = cfg.chunk_size if cfg else 500
            c_overlap = cfg.chunk_overlap if cfg else 50

            source_type, extracted_text = extract_document_text(filename, raw_bytes)
            chunks = chunk_text(extracted_text, chunk_size=c_size, chunk_overlap=c_overlap) or [extracted_text]

            emb_service = EmbeddingService()
            embeddings = await emb_service.embed_batch(chunks)

            # Validate 384-dim embeddings
            for emb in embeddings:
                if len(emb) != 384:
                    raise ValueError(f"Expected 384-dim embedding, got {len(emb)}")

            doc = Document(
                tenant_id=tenant_id,
                title=filename,
                source_type=source_type,
                total_chunks=len(chunks),
            )
            session.add(doc)
            await session.flush()

            pub_status = "PUBLISHED" if verified else "REVIEW_REQUIRED"
            for idx, (chunk_txt, emb) in enumerate(zip(chunks, embeddings)):
                meta = {
                    "category": category_slug,
                    "category_id": category_id,
                    "language": language,
                    "source": source,
                    "source_version": source_version,
                    "effective_date": effective_date,
                    "verified": verified,
                    "status": pub_status,
                }
                session.add(
                    DocumentChunk(
                        document_id=doc.id,
                        tenant_id=tenant_id,
                        chunk_index=idx,
                        content=chunk_txt,
                        embedding=emb,
                        embedding_json=json.dumps(emb),
                        meta_json=json.dumps(meta),
                    )
                )

            session.add(
                KnowledgeDocumentVersion(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    version=source_version,
                    content_hash=job.file_hash or "",
                    status=pub_status,
                )
            )
            job.document_id = doc.id
            job.total_chunks = len(chunks)
            job.embedded_chunks = len(embeddings)
            job.status = pub_status
            await _log_audit(session, tenant_id, "INGEST_COMPLETE", "document", doc.id, {"chunks": len(chunks), "status": pub_status})
            await session.commit()
        except Exception as exc:
            logger.error(f"[Knowledge Ingestion] Job {job_id} failed: {exc}")
            job.status = "FAILED"
            job.error_message = str(exc)[:500]
            await session.commit()


@router.post("/documents/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_knowledge_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category_id: Optional[str] = Form(None),
    language: str = Form("en"),
    source: str = Form("Admin Upload"),
    source_version: str = Form("1.0"),
    effective_date: str = Form(""),
    verified: bool = Form(False),
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "document.txt"
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    cat_slug = "uncategorized"
    if category_id:
        cat = await session.scalar(
            select(KnowledgeCategory).where(KnowledgeCategory.id == category_id, KnowledgeCategory.tenant_id == context.tenant_id)
        )
        if not cat:
            raise HTTPException(status_code=404, detail="Selected category not found.")
        cat_slug = cat.slug

    job = KnowledgeIngestionJob(
        tenant_id=context.tenant_id,
        category_id=category_id,
        filename=filename,
        file_hash=file_hash,
        status="QUEUED",
        metadata_json=json.dumps({
            "category_slug": cat_slug,
            "language": language,
            "source": source,
            "source_version": source_version,
            "effective_date": effective_date,
            "verified": verified,
        }),
    )
    session.add(job)
    await _log_audit(session, context.tenant_id, "UPLOAD_DOCUMENT", "ingestion_job", job.id, {"filename": filename, "category": cat_slug})
    await session.commit()

    background_tasks.add_task(
        _run_ingestion_job,
        job.id,
        context.tenant_id,
        filename,
        raw_bytes,
        category_id,
        cat_slug,
        language,
        source,
        source_version,
        effective_date,
        verified,
    )
    return {"job_id": job.id, "status": "QUEUED", "filename": filename, "category": cat_slug}


@router.get("/documents")
async def list_knowledge_documents(
    limit: int = 50,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    res = await session.execute(
        select(Document).where(Document.tenant_id == context.tenant_id).order_by(Document.created_at.desc()).limit(limit)
    )
    docs = res.scalars().all()
    return [
        {
            "id": d.id,
            "title": d.title,
            "source_type": d.source_type,
            "total_chunks": d.total_chunks,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.post("/documents/{document_id}/publish")
async def publish_document(
    document_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    doc = await session.scalar(select(Document).where(Document.id == document_id, Document.tenant_id == context.tenant_id))
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = (await session.execute(select(DocumentChunk).where(DocumentChunk.document_id == document_id))).scalars().all()
    for ch in chunks:
        meta = json.loads(ch.meta_json or "{}")
        meta["verified"] = True
        meta["status"] = "PUBLISHED"
        ch.meta_json = json.dumps(meta)

    job = await session.scalar(select(KnowledgeIngestionJob).where(KnowledgeIngestionJob.document_id == document_id))
    if job:
        job.status = "PUBLISHED"

    await _log_audit(session, context.tenant_id, "PUBLISH_DOCUMENT", "document", document_id)
    await session.commit()
    return {"document_id": document_id, "status": "PUBLISHED", "chunks_updated": len(chunks)}


@router.post("/documents/{document_id}/archive")
async def archive_document(
    document_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    doc = await session.scalar(select(Document).where(Document.id == document_id, Document.tenant_id == context.tenant_id))
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = (await session.execute(select(DocumentChunk).where(DocumentChunk.document_id == document_id))).scalars().all()
    for ch in chunks:
        meta = json.loads(ch.meta_json or "{}")
        meta["verified"] = False
        meta["status"] = "ARCHIVED"
        ch.meta_json = json.dumps(meta)

    job = await session.scalar(select(KnowledgeIngestionJob).where(KnowledgeIngestionJob.document_id == document_id))
    if job:
        job.status = "ARCHIVED"

    await _log_audit(session, context.tenant_id, "ARCHIVE_DOCUMENT", "document", document_id)
    await session.commit()
    return {"document_id": document_id, "status": "ARCHIVED"}


# ── Jobs ─────────────────────────────────────────────────────────────────────

@router.get("/jobs")
async def list_ingestion_jobs(
    limit: int = 50,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    res = await session.execute(
        select(KnowledgeIngestionJob)
        .where(KnowledgeIngestionJob.tenant_id == context.tenant_id)
        .order_by(KnowledgeIngestionJob.created_at.desc())
        .limit(limit)
    )
    return [
        {
            "id": j.id,
            "document_id": j.document_id,
            "category_id": j.category_id,
            "filename": j.filename,
            "status": j.status,
            "total_chunks": j.total_chunks,
            "embedded_chunks": j.embedded_chunks,
            "error_message": j.error_message,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in res.scalars().all()
    ]


# ── RAG Playground & Search ──────────────────────────────────────────────────

@router.post("/playground")
async def rag_playground(
    payload: PlaygroundRequest,
    context: TenantContext = Depends(get_current_tenant),
):
    retrieval = HybridRetrievalService()
    res = await retrieval.search(
        query=payload.query,
        tenant_id=context.tenant_id,
        top_k=payload.top_k,
        preferred_category=payload.category_slug,
    )
    from providers.chat.ivr_mini_model import ivr_mini_brain
    rag_ctx = "\n\n".join(res.chunk_texts)
    answer = ivr_mini_brain.synthesize(
        user_query=payload.query,
        rag_context=rag_ctx,
        language=payload.language,
        service_category=payload.category_slug,
    )
    chunks_out = []
    for chunk_obj, score in res.scored_chunks:
        meta = {}
        try:
            meta = json.loads(getattr(chunk_obj, "meta_json", "{}") or "{}")
        except Exception:
            pass
        chunks_out.append({
            "content": getattr(chunk_obj, "content", ""),
            "score": round(float(score), 4),
            "category": meta.get("category", "unknown"),
            "source": meta.get("source", "knowledge_base"),
            "page": meta.get("page"),
        })
    return {
        "query": payload.query,
        "language": payload.language,
        "confidence_tier": res.evaluation.tier.value,
        "top_score": round(float(res.evaluation.top_score), 4),
        "cache_hit": res.cache_hit,
        "answer": answer,
        "chunks": chunks_out,
    }


# ── AI Configuration ─────────────────────────────────────────────────────────

@router.get("/config")
async def get_ai_config(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    cfg = await session.scalar(
        select(KnowledgeConfiguration).where(
            KnowledgeConfiguration.tenant_id == context.tenant_id,
            KnowledgeConfiguration.category_id.is_(None),
        )
    )
    if not cfg:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "top_k": 4,
            "confidence_threshold": 0.55,
            "reranking_enabled": True,
            "supported_languages": ["en", "hi", "or"],
        }
    return {
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "embedding_model": cfg.embedding_model,
        "embedding_dim": cfg.embedding_dim,
        "top_k": cfg.top_k,
        "confidence_threshold": cfg.confidence_threshold,
        "reranking_enabled": cfg.reranking_enabled,
        "supported_languages": json.loads(cfg.supported_languages or '["en","hi","or"]'),
    }


@router.patch("/config")
async def update_ai_config(
    payload: ConfigUpdate,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    cfg = await session.scalar(
        select(KnowledgeConfiguration).where(
            KnowledgeConfiguration.tenant_id == context.tenant_id,
            KnowledgeConfiguration.category_id.is_(None),
        )
    )
    if not cfg:
        cfg = KnowledgeConfiguration(tenant_id=context.tenant_id)
        session.add(cfg)

    if payload.chunk_size is not None:
        cfg.chunk_size = payload.chunk_size
    if payload.chunk_overlap is not None:
        cfg.chunk_overlap = payload.chunk_overlap
    if payload.top_k is not None:
        cfg.top_k = payload.top_k
    if payload.confidence_threshold is not None:
        cfg.confidence_threshold = payload.confidence_threshold
    if payload.reranking_enabled is not None:
        cfg.reranking_enabled = payload.reranking_enabled
    if payload.supported_languages is not None:
        cfg.supported_languages = json.dumps(payload.supported_languages)

    await _log_audit(session, context.tenant_id, "UPDATE_CONFIG", "knowledge_config", cfg.id, payload.model_dump(exclude_none=True))
    await session.commit()
    return {"status": "updated"}


@router.get("/audit-logs")
async def list_audit_logs(
    limit: int = 50,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    res = await session.execute(
        select(KnowledgeAuditLog)
        .where(KnowledgeAuditLog.tenant_id == context.tenant_id)
        .order_by(KnowledgeAuditLog.created_at.desc())
        .limit(limit)
    )
    return [
        {
            "id": a.id,
            "actor": a.actor,
            "action": a.action,
            "resource_type": a.resource_type,
            "resource_id": a.resource_id,
            "details": json.loads(a.details_json or "{}"),
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in res.scalars().all()
    ]
