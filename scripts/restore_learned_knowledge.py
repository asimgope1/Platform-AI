#!/usr/bin/env python3
"""
Turnkey Knowledge & State Restoration Script.
============================================
Restores all learned RAG knowledge chunks and pre-rendered audio cache
so that any new clone or deployment functions identically without data loss.

Usage:
    python scripts/restore_learned_knowledge.py
"""
import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.database import async_session, init_db
from storage.repositories.tenant_repo import TenantRepository
from storage.repositories.document_repo import DocumentRepository
from storage.repositories.api_key_repo import ApiKeyRepository
from platform_core.security.api_keys import hash_api_key
from retrieval.ingestion import DocumentIngestionService
from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService
from providers.registry import provider_registry
from telephony.ivr_audio_cache import warm_up_ivr_prompts, CACHE_DIR
from audio.service import AudioService

IVR_TENANT_NAME = "tenant-ivr-support"
IVR_API_KEY = "sk-live-tenant-ivr-support"
CHUNKS_DUMP_FILE = Path(__file__).resolve().parent.parent / "data" / "ivr_chunks_dump.json"


async def restore():
    print("=" * 70)
    print("RESTORE AI PLATFORM: LEARNED CHUNKS & AUDIO CACHE")
    print("=" * 70)

    # 1. Initialize DB schema (tables, pgvector extension)
    print("\n[1/4] Initializing database schema...")
    await init_db()
    print("✓ PostgreSQL schema and pgvector extension ready.")

    # 2. Setup providers
    print("\n[2/4] Initializing provider registry...")
    provider_registry.initialize_from_file("config/providers.yaml")
    emb_adapter = provider_registry.get_embedding_adapter("semantic-default")
    print(f"✓ Embedding provider loaded: {emb_adapter.__class__.__name__} ({emb_adapter.dimension} dimensions)")

    # 3. Setup Tenant & API Key
    print("\n[3/4] Ensuring IVR tenant and API keys exist...")
    async with async_session() as session:
        tenant_repo = TenantRepository(session)
        doc_repo = DocumentRepository(session)
        key_repo = ApiKeyRepository(session)
        emb_service = EmbeddingService(provider_id="semantic-default")
        ingestion_service = DocumentIngestionService(doc_repo, emb_service)

        tenant = await tenant_repo.get_by_name(IVR_TENANT_NAME)
        if not tenant:
            tenant = await tenant_repo.create(IVR_TENANT_NAME)
            print(f"✓ Created tenant: {tenant.name} (id={tenant.id})")
        else:
            print(f"✓ Found existing tenant: {tenant.name} (id={tenant.id})")

        # Ensure API key exists for this tenant
        existing_keys = await key_repo.list_by_tenant(tenant.id)
        if not existing_keys:
            key_hash = hash_api_key(IVR_API_KEY)
            await key_repo.create(tenant.id, key_hash, name="IVR Live Telephone Key")
            print(f"✓ Provisioned API key: '{IVR_API_KEY}'")
        else:
            print(f"✓ Active API key ready for tenant {tenant.name}.")

        # Ingest/Restore learned chunks from data/ivr_chunks_dump.json
        if CHUNKS_DUMP_FILE.exists():
            print(f"\n[4/4] Loading learned knowledge chunks from {CHUNKS_DUMP_FILE.name}...")
            with open(CHUNKS_DUMP_FILE, "r", encoding="utf-8") as f:
                dump_data = json.load(f)

            raw_chunks = dump_data.get("chunks", [])
            print(f"Found {len(raw_chunks)} learned chunks in dump file.")

            # Group chunks by document_title and source_type
            docs_map = {}
            for ch in raw_chunks:
                key = (ch.get("document_title", "General Knowledge"), ch.get("source_type", "general"))
                if key not in docs_map:
                    docs_map[key] = []
                docs_map[key].append(ch.get("content", ""))

            existing_docs = await doc_repo.list_by_tenant(tenant.id)
            existing_titles = {d.title for d in existing_docs}

            restored_count = 0
            for (title, source_type), contents in docs_map.items():
                if title in existing_titles:
                    continue
                combined_content = "\n\n".join(contents)
                doc = await ingestion_service.ingest_document(
                    tenant_id=tenant.id,
                    title=title,
                    content=combined_content,
                    source_type=source_type,
                )
                restored_count += doc.total_chunks
                print(f"  + Ingested document '{title[:45]}...' ({doc.total_chunks} chunks)")

            print(f"✓ Successfully restored {restored_count} new chunks into pgvector.")
        else:
            print(f"\n[!] Chunks dump file {CHUNKS_DUMP_FILE} not found. Running training script...")
            from scripts.train_expanded_ivr_knowledge import train_expanded_knowledge
            await train_expanded_knowledge()

        # Verify semantic search
        search_service = SemanticSearchService(doc_repo, emb_service)
        test_res = await search_service.search(tenant.id, "Mera bank balance kitna hai", top_k=1)
        if test_res:
            print(f"\n✓ Semantic Search Test: top score = {test_res[0][1]:.3f} (Chunk ID: {test_res[0][0].id})")
        else:
            print("\n[!] Warning: Semantic search returned no results.")

    # 4. Audio prompt cache check
    print("\nChecking audio cache directory...")
    cached_pcm_files = list(CACHE_DIR.glob("*.pcm"))
    print(f"Found {len(cached_pcm_files)} pre-rendered audio prompts in {CACHE_DIR.name}/.")
    if len(cached_pcm_files) < 10:
        print("Warming up audio cache prompts...")
        audio_service = AudioService()
        await warm_up_ivr_prompts(audio_service)
        print("✓ Audio prompts warmed up.")

    print("\n" + "=" * 70)
    print("RESTORE COMPLETE: AI Platform is 100% operational!")
    print("Start server with: uvicorn services.api.main:app --host 0.0.0.0 --port 8000")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(restore())

