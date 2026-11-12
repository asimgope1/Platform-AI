"""
Knowledge Dataset Ingestion Service
===================================
Ingests structured domain policy JSON files from data/banking_knowledge/
into PostgreSQL pgvector with versioning, tenant isolation, and rich metadata.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from storage.database import async_session
from storage.models.document import Document, DocumentChunk
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class KnowledgeIngestionService:
    """Ingests structured verified banking policy documents into pgvector."""

    def __init__(self, embedding_provider_id: str = "semantic-default") -> None:
        self.embedding_provider_id = embedding_provider_id

    @staticmethod
    def load_dataset_directory(dataset_dir: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                records.extend(payload)
            elif isinstance(payload, dict):
                records.append(payload)
        return records

    async def ingest_directory(
        self,
        dataset_dir: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Idempotently ingest or upgrade structured policy documents by (tenant_id, title).
        Skips already up-to-date source_version documents so repeated runs are safe.
        """
        records = self.load_dataset_directory(dataset_dir)
        if not records:
            return {"loaded": 0, "inserted": 0, "updated": 0, "skipped": 0}

        emb_service = EmbeddingService(provider_id=self.embedding_provider_id)
        inserted = 0
        updated = 0
        skipped = 0

        async with async_session() as session:
            doc_repo = DocumentRepository(session)
            for item in records:
                item_tenant = tenant_id or str(item.get("tenant_id") or "")
                title = str(item.get("title") or item.get("topic") or "Untitled Policy")
                content = str(item.get("content") or "").strip()
                if not item_tenant or not content:
                    continue

                meta = {
                    "category": item.get("category", "general"),
                    "topic": item.get("topic", "general"),
                    "title": title,
                    "language": item.get("language", ["en"]),
                    "source": item.get("source", "official_bank_policy"),
                    "source_version": str(item.get("source_version", "1.0")),
                    "verified": bool(item.get("verified", True)),
                    "effective_date": item.get("effective_date", "2026-01-01"),
                    "ask_count": int(item.get("ask_count", 1)),
                }

                # Embed full formatted document with title + category + body for high recall
                searchable_text = f"[{meta['category']}] {title}\n{content}"
                doc = await doc_repo.get_or_create_document(
                    tenant_id=item_tenant,
                    title=title,
                    source_type=str(meta["source"]),
                )

                existing_res = await session.execute(
                    select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index.asc())
                )
                existing_chunks = list(existing_res.scalars().all())

                if existing_chunks:
                    first = existing_chunks[0]
                    first_meta = json.loads(first.meta_json) if first.meta_json else {}
                    if (
                        first.content == searchable_text
                        and str(first_meta.get("source_version", "")) == meta["source_version"]
                    ):
                        skipped += 1
                        continue

                    emb = await emb_service.embed_query(searchable_text)
                    first.content = searchable_text
                    first.meta_json = json.dumps(meta, ensure_ascii=False)
                    if emb:
                        first.embedding = emb if len(emb) == 384 else None
                        first.embedding_768 = emb if len(emb) == 768 else None
                        first.embedding_json = json.dumps(emb)
                    await session.commit()
                    updated += 1
                else:
                    emb = await emb_service.embed_query(searchable_text)
                    await doc_repo.append_chunk(
                        document_id=doc.id,
                        tenant_id=item_tenant,
                        content=searchable_text,
                        embedding=emb,
                        meta=meta,
                    )
                    inserted += 1

        logger.info(
            f"[KnowledgeIngestion] Processed {len(records)} policy docs: "
            f"inserted={inserted}, updated={updated}, skipped={skipped}"
        )
        return {
            "loaded": len(records),
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
        }

