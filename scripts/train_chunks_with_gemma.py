#!/usr/bin/env python3
"""
Gemma Knowledge Chunk Training & Category Updater CLI
====================================================
Uses Gemma 4 to distill, train, and categorize knowledge chunks,
updating PostgreSQL pgvector and the AI Platform Category Registry.

Usage:
    .venv/bin/python scripts/train_chunks_with_gemma.py \
        --file /Users/asimgope/Downloads/Jana_Seva_Odisha_Service_Guide.pdf \
        --tenant-id 879e9bda-8049-4824-9aa0-fc5e9f0012d9
"""
import argparse
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval.parsers import extract_document_text
from scripts.ingest_guide import smart_split_guide_sections
from services.knowledge.gemma_knowledge_trainer import GemmaKnowledgeTrainer


async def run_training(file_path: str, tenant_id: str, max_sections: int = 5):
    print("\n" + "=" * 70)
    print("GEMMA KNOWLEDGE CHUNK TRAINING & DYNAMIC CATEGORY UPDATER")
    print("=" * 70)

    filename = os.path.basename(file_path)
    base_title = os.path.splitext(filename)[0].replace("_", " ").title()

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    _, text = extract_document_text(filename, raw_bytes)
    sections = smart_split_guide_sections(text, default_title=base_title)

    trainer = GemmaKnowledgeTrainer()
    print(f"✓ Connected to Gemma 4 via: {trainer.base_url} (Model: {trainer.model_name})")
    print(f"✓ Found {len(sections)} sections in '{filename}'. Training first {min(len(sections), max_sections)} sections...\n")

    results = []
    for idx, (heading, body) in enumerate(sections[:max_sections], start=1):
        print(f"[{idx}/{min(len(sections), max_sections)}] Training with Gemma: '{heading}'...")
        res = await trainer.train_and_update_chunk(
            tenant_id=tenant_id,
            section_text=body,
            title=base_title,
            source_name=filename,
        )
        print(f"    ✓ Auto-classified Category: '{res['category']}'")
        print(f"    ✓ Generated Q&As: {res['qa_count']}")
        print(f"    ✓ Chunk Vectorized & Updated in pgvector (Doc ID: {res['doc_id']})\n")
        results.append(res)

    print("=" * 70)
    print(f"✓ Completed Gemma training for {len(results)} chunks.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train knowledge chunks using Gemma 4")
    parser.add_argument("--file", required=True, help="Path to guide file")
    parser.add_argument("--tenant-id", default="879e9bda-8049-4824-9aa0-fc5e9f0012d9", help="Tenant ID")
    parser.add_argument("--sections", type=int, default=3, help="Number of sections to train in this batch")
    args = parser.parse_args()

    asyncio.run(run_training(args.file, args.tenant_id, max_sections=args.sections))


if __name__ == "__main__":
    main()
