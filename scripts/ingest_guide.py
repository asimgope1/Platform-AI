#!/usr/bin/env python3
"""
Universal Dynamic Knowledge Guide Ingestion Tool
================================================
Ingests ANY service guide, policy manual, or enterprise document (PDF, DOCX, MD, TXT, CSV, JSON)
into PostgreSQL pgvector and SQLite.

Usage:
    .venv/bin/python scripts/ingest_guide.py --file /Users/asimgope/Downloads/Jana_Seva_Odisha_Service_Guide.pdf \
        --tenant-id 879e9bda-8049-4824-9aa0-fc5e9f0012d9 \
        --domain government_services \
        --category citizen_welfare
"""
import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.parsers import extract_document_text
from retrieval.embedding_service import EmbeddingService
from storage.database import async_session, init_db
from storage.models.document import DocumentChunk
from storage.repositories.document_repo import DocumentRepository
from sqlalchemy import select


def smart_split_guide_sections(text: str, default_title: str) -> List[Tuple[str, str]]:
    """
    Splits text by structural markers (numbered headings, markdown headers, or major topics).
    Falls back to window chunking if no natural sections are detected.
    """
    # Remove repetitive page footers/headers (e.g., 'Page 1', 'Page 2')
    cleaned = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    # Pattern for numbered sections (e.g. '1. Scope', '2. Subhadra Scheme') or Markdown (## Heading)
    heading_pattern = r'\n(?=(?:[#]{1,3}\s+|\d{1,2}\.\s+[A-Z]))'
    raw_sections = re.split(heading_pattern, cleaned)

    sections: List[Tuple[str, str]] = []
    for s in raw_sections:
        s_clean = s.strip()
        if not s_clean or len(s_clean) < 40:
            continue
        lines = [l.strip() for l in s_clean.splitlines() if l.strip()]
        first_line = lines[0] if lines else default_title
        # Clean heading label
        heading = re.sub(r'^[#\d\.\s]+', '', first_line).strip()
        if not heading:
            heading = default_title

        # If a single section is very long (> 2500 chars), subdivide it into 1200-char subchunks
        if len(s_clean) > 2500:
            words = s_clean.split()
            chunk_words = 350
            for i in range(0, len(words), chunk_words - 40):
                sub_text = " ".join(words[i : i + chunk_words])
                sub_heading = f"{heading} (Part {i // (chunk_words - 40) + 1})"
                sections.append((sub_heading, sub_text))
        else:
            sections.append((heading, s_clean))

    if not sections:
        sections = [(default_title, cleaned)]
    return sections


async def ingest_guide_file(
    file_path: str,
    tenant_id: str,
    domain: str,
    category: str,
    language: List[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    base_title = os.path.splitext(filename)[0].replace("_", " ").title()
    languages = language or ["en", "hi", "or"]

    print(f"\n[Dynamic Ingestion] Reading file: {file_path}")
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    source_type, extracted_text = extract_document_text(filename, raw_bytes)
    print(f"✓ Extracted {len(extracted_text)} characters (Format: {source_type})")

    sections = smart_split_guide_sections(extracted_text, default_title=base_title)
    print(f"✓ Parsed {len(sections)} high-coherence sections.")

    await init_db()
    emb_service = EmbeddingService(provider_id="semantic-default")

    inserted_count = 0
    updated_count = 0

    async with async_session() as session:
        doc_repo = DocumentRepository(session)

        for heading, body in sections:
            doc_title = f"{base_title} - {heading}" if heading != base_title else base_title
            searchable_text = f"[{domain} / {category}] {doc_title}\n{body}"

            meta = {
                "domain": domain,
                "category": category,
                "topic": heading.lower().replace(" ", "_"),
                "title": doc_title,
                "language": languages,
                "source": filename,
                "source_type": source_type,
                "source_version": "1.0",
                "verified": True,
                "effective_date": datetime.now().strftime("%Y-%m-%d"),
                "ask_count": 1,
            }

            doc = await doc_repo.get_or_create_document(
                tenant_id=tenant_id,
                title=doc_title,
                source_type=f"guide_{source_type}",
            )

            # Check if chunk exists
            chunk_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
            res = await session.execute(chunk_stmt)
            existing_chunks = list(res.scalars().all())
            if existing_chunks:
                first = existing_chunks[0]
                if first.content != searchable_text:
                    emb = await emb_service.embed_query(searchable_text)
                    first.content = searchable_text
                    first.meta_json = json.dumps(meta, ensure_ascii=False)
                    first.embedding = emb if len(emb) == 384 else None
                    first.embedding_768 = emb if len(emb) == 768 else None
                    first.embedding_json = json.dumps(emb)
                    await session.commit()
                    updated_count += 1
            else:
                emb = await emb_service.embed_query(searchable_text)
                await doc_repo.append_chunk(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    content=searchable_text,
                    embedding=emb,
                    meta=meta,
                )
                inserted_count += 1

            print(f"  + Vectorized: '{doc_title[:55]}...' ({len(searchable_text)} chars)")

    print(f"\n✓ Ingestion complete for tenant {tenant_id}:")
    print(f"  Inserted: {inserted_count} new chunks | Updated: {updated_count} chunks")
    return {
        "filename": filename,
        "tenant_id": tenant_id,
        "domain": domain,
        "category": category,
        "sections": len(sections),
        "inserted": inserted_count,
        "updated": updated_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Universal Dynamic Knowledge Guide Ingestion Tool")
    parser.add_argument("--file", required=True, help="Path to guide file (.pdf, .docx, .md, .txt, .csv, .json)")
    parser.add_argument("--tenant-id", default="879e9bda-8049-4824-9aa0-fc5e9f0012d9", help="Tenant ID")
    parser.add_argument("--domain", default="citizen_services", help="Domain slug (e.g. government, telecom, healthcare)")
    parser.add_argument("--category", default="guide", help="Category slug (e.g. welfare_schemes, billing, certificates)")
    args = parser.parse_args()

    result = asyncio.run(
        ingest_guide_file(
            file_path=args.file,
            tenant_id=args.tenant_id,
            domain=args.domain,
            category=args.category,
        )
    )
    print("\nResult summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
