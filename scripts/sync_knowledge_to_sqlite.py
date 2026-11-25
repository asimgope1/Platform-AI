"""
Sync all Banking Knowledge into SQLite ai_platform.db
=====================================================
Ensures local SQLite fallback has 100% identical rich knowledge chunks.
"""
import glob
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.embedding_service import EmbeddingService

SQLITE_PATH = os.path.join(ROOT_DIR, "ai_platform.db")
DATA_DIR = os.path.join(ROOT_DIR, "data", "banking_knowledge")
TENANTS = [
    ("879e9bda-8049-4824-9aa0-fc5e9f0012d9", "tenant-ivr-support"),
    ("e6ce1783-a1b4-48b7-adc9-b3443c14d6d2", "DemoTenant"),
]

import asyncio

async def sync():
    emb_service = EmbeddingService(provider_id="semantic-default")
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()

    # Ensure tenants exist in SQLite
    for tid, tname in TENANTS:
        c.execute("SELECT id FROM tenants WHERE id = ?", (tid,))
        if not c.fetchone():
            c.execute(
                "INSERT INTO tenants (id, name, is_active, created_at) VALUES (?, ?, ?, ?)",
                (tid, tname, 1, datetime.now().isoformat())
            )
    conn.commit()

    records = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)

    print(f"Loaded {len(records)} knowledge records from {DATA_DIR}")

    for tid, tname in TENANTS:
        inserted = 0
        updated = 0
        for item in records:
            title = str(item.get("title") or item.get("topic") or "Untitled Policy")
            content = str(item.get("content") or "").strip()
            category = item.get("category", "general")
            source_type = str(item.get("source", "official_bank_policy"))
            searchable_text = f"[{category}] {title}\n{content}"

            meta = {
                "category": category,
                "topic": item.get("topic", "general"),
                "title": title,
                "language": item.get("language", ["en"]),
                "source": source_type,
                "source_version": str(item.get("source_version", "2.0")),
                "verified": bool(item.get("verified", True)),
                "effective_date": item.get("effective_date", "2026-01-01"),
                "ask_count": int(item.get("ask_count", 1)),
            }

            # Find or create doc
            c.execute("SELECT id FROM documents WHERE tenant_id = ? AND title = ?", (tid, title))
            row = c.fetchone()
            if row:
                doc_id = row[0]
            else:
                doc_id = str(uuid.uuid4())
                c.execute(
                    "INSERT INTO documents (id, tenant_id, title, source_type, total_chunks, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, tid, title, source_type, 1, datetime.utcnow().isoformat())
                )

            # Check chunk
            c.execute("SELECT id, content FROM document_chunks WHERE document_id = ?", (doc_id,))
            chunk_row = c.fetchone()

            if chunk_row:
                chunk_id, existing_content = chunk_row
                if existing_content != searchable_text:
                    emb = await emb_service.embed_query(searchable_text)
                    c.execute(
                        "UPDATE document_chunks SET content = ?, embedding_json = ?, meta_json = ? WHERE id = ?",
                        (searchable_text, json.dumps(emb), json.dumps(meta, ensure_ascii=False), chunk_id)
                    )
                    updated += 1
            else:
                chunk_id = str(uuid.uuid4())
                emb = await emb_service.embed_query(searchable_text)
                c.execute(
                    "INSERT INTO document_chunks (id, document_id, tenant_id, chunk_index, content, embedding_json, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, tid, 0, searchable_text, json.dumps(emb), json.dumps(meta, ensure_ascii=False))
                )
                inserted += 1

        conn.commit()
        print(f"Tenant {tname} ({tid}): inserted={inserted}, updated={updated}")

    c.execute("SELECT tenant_id, count(*) FROM document_chunks GROUP BY tenant_id")
    print("Final SQLite chunk counts:", c.fetchall())
    conn.close()

if __name__ == "__main__":
    asyncio.run(sync())
