#!/usr/bin/env python3
"""
Dump Learned Knowledge & Chunks Script.
======================================
Exports all documents and pgvector chunks from PostgreSQL into `data/ivr_chunks_dump.json`
so that live telephone call learnings can be committed to Git and shared seamlessly.

Usage:
    python scripts/dump_learned_knowledge.py
"""
import os
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select
from storage.database import async_session
from storage.models.document import Document, DocumentChunk
from storage.repositories.tenant_repo import TenantRepository
from storage.repositories.document_repo import DocumentRepository

IVR_TENANT_NAME = "tenant-ivr-support"
DUMP_FILE = Path(__file__).resolve().parent.parent / "data" / "ivr_chunks_dump.json"


async def dump():
    print(f"Dumping all learned knowledge for tenant '{IVR_TENANT_NAME}'...")
    async with async_session() as session:
        tenant_repo = TenantRepository(session)
        doc_repo = DocumentRepository(session)

        tenant = await tenant_repo.get_by_name(IVR_TENANT_NAME)
        if not tenant:
            print(f"[!] Tenant '{IVR_TENANT_NAME}' not found.")
            return

        docs = await doc_repo.list_by_tenant(tenant.id)
        all_chunks = []
        for doc in docs:
            stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)
            res = await session.execute(stmt)
            chunks = res.scalars().all()
            for ch in chunks:
                meta = {}
                if ch.meta_json:
                    try:
                        meta = json.loads(ch.meta_json)
                    except Exception:
                        pass
                all_chunks.append({
                    "chunk_id": str(ch.id),
                    "document_title": doc.title,
                    "source_type": doc.source_type,
                    "chunk_index": ch.chunk_index,
                    "content": ch.content,
                    "metadata": meta,
                })

        DUMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "total_chunks": len(all_chunks),
            "tenant_id": str(tenant.id),
            "tenant_name": tenant.name,
            "chunks": all_chunks,
        }
        with open(DUMP_FILE, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Successfully exported {len(all_chunks)} chunks across {len(docs)} documents to {DUMP_FILE.name}!")


if __name__ == "__main__":
    asyncio.run(dump())
