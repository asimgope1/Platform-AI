#!/usr/bin/env python3
"""
Live Verification of Real Gemma 4 Model Generation and Full API-Level RAG
Connecting to Live Colab GPU Runtime (google/gemma-4-E2B-it on NVIDIA T4)
with Real PostgreSQL 16 + pgvector and Real Sentence Transformers Embeddings.
"""

import sys
import os
import asyncio
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platform_core.security.api_keys import hash_api_key
from storage.database import get_db_session
from storage.models.tenant import Tenant
from storage.models.api_key import ApiKey
from storage.models.document import Document, DocumentChunk
from providers.registry import provider_registry
from services.api.main import app

POSTGRES_TEST_URL = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://localhost/ai_platform_test")


async def main():
    print("=" * 65)
    print("LIVE VERIFICATION: GEMMA 4 GPU RUNTIME & FULL POSTGRESQL RAG")
    print("=" * 65)

    # 1. Initialize Providers
    provider_registry.initialize_from_file("config/providers.yaml")
    model_meta = provider_registry.get_model_metadata("gemma-gpu")
    print(f"✓ Model alias registered: {model_meta.alias}")
    print(f"  - Underlying model: {model_meta.underlying_model}")
    print(f"  - Provider ID: {model_meta.provider_id}")

    # 2. Setup PostgreSQL test session
    engine = create_async_engine(POSTGRES_TEST_URL, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    tenant_a_id = "tenant-gpu-live-alpha"
    tenant_b_id = "tenant-gpu-live-beta"
    key_a_raw = "sk-live-gpu-alpha-key-12345"
    key_b_raw = "sk-live-gpu-beta-key-67890"

    async with session_factory() as session:
        # Clean previous test entries
        await session.execute(text(f"DELETE FROM document_chunks WHERE tenant_id IN ('{tenant_a_id}', '{tenant_b_id}')"))
        await session.execute(text(f"DELETE FROM documents WHERE tenant_id IN ('{tenant_a_id}', '{tenant_b_id}')"))
        await session.execute(text(f"DELETE FROM api_keys WHERE tenant_id IN ('{tenant_a_id}', '{tenant_b_id}')"))
        await session.execute(text(f"DELETE FROM tenants WHERE id IN ('{tenant_a_id}', '{tenant_b_id}')"))

        # Seed tenants
        t_a = Tenant(id=tenant_a_id, name="GPU Live Alpha Tenant", is_active=True)
        t_b = Tenant(id=tenant_b_id, name="GPU Live Beta Tenant", is_active=True)
        session.add_all([t_a, t_b])

        # Seed API keys
        k_a = ApiKey(
            id="key-gpu-alpha",
            tenant_id=tenant_a_id,
            key_hash=hash_api_key(key_a_raw),
            name="Key Alpha",
            scopes="*",
            is_active=True,
        )
        k_b = ApiKey(
            id="key-gpu-beta",
            tenant_id=tenant_b_id,
            key_hash=hash_api_key(key_b_raw),
            name="Key Beta",
            scopes="*",
            is_active=True,
        )
        session.add_all([k_a, k_b])
        await session.commit()
        print("✓ Test tenants and API keys seeded in PostgreSQL.")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 3. Test Health
        health_resp = await client.get("/health")
        print(f"✓ Health Check: {health_resp.status_code} - {health_resp.json()['status']}")

        # 4. Direct Chat Completion on Colab GPU Gemma 4
        print("\n--- STEP 1: Direct Real Gemma 4 Text Generation (No RAG) ---")
        direct_resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key_a_raw}"},
            json={
                "model": "gemma-gpu",
                "messages": [{"role": "user", "content": "What is the capital of France? Answer with just the name."}],
                "enable_rag": False,
                "max_tokens": 10,
            },
        )
        print(f"Status Code: {direct_resp.status_code}")
        assert direct_resp.status_code == 200, direct_resp.text
        direct_data = direct_resp.json()
        direct_answer = direct_data["choices"][0]["message"]["content"]
        print(f"✓ Gemma 4 GPU Answer: '{direct_answer.strip()}'")
        print(f"  Response ID: {direct_data['id']}")
        print(f"  Model: {direct_data['model']}")

        # 5. Document Ingestion with Real Sentence Transformers into PostgreSQL pgvector
        print("\n--- STEP 2: Ingesting Domain Document via API into PostgreSQL pgvector ---")
        aurora_content = (
            "The Aurora quantum sensor operates at 4 Kelvin and detects magnetic anomalies with sub-nanotesla precision. "
            "It is engineered for deep geophysical exploration and navigation in GPS-denied subterranean tunnels. "
            "The sensor utilizes laser-pumped rubidium vapor cells to achieve ultra-high sensitivity and stable bias."
        )
        doc_resp = await client.post(
            "/v1/documents",
            headers={"Authorization": f"Bearer {key_a_raw}"},
            json={
                "title": "Aurora Quantum Sensor Tech Spec",
                "content": aurora_content,
            },
        )
        print(f"Status Code: {doc_resp.status_code}")
        assert doc_resp.status_code in (200, 201), doc_resp.text
        doc_data = doc_resp.json()
        print(f"✓ Document Ingested: ID={doc_data['id']}, chunks={doc_data.get('total_chunks', 1)}")

        # 6. Full RAG Query with Real Retrieval and Real Gemma 4 Generation
        print("\n--- STEP 3: Complete RAG Query through API (Tenant Alpha) ---")
        rag_resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key_a_raw}"},
            json={
                "model": "gemma-gpu",
                "messages": [
                    {"role": "user", "content": "What is the operating temperature of the Aurora quantum sensor? State the temperature clearly."}
                ],
                "enable_rag": True,
                "max_tokens": 50,
            },
        )
        print(f"Status Code: {rag_resp.status_code}")
        assert rag_resp.status_code == 200, rag_resp.text
        rag_data = rag_resp.json()
        rag_answer = rag_data["choices"][0]["message"]["content"]
        print(f"✓ Gemma 4 GPU RAG Answer: '{rag_answer.strip()}'")
        has_temp = "4 kelvin" in rag_answer.lower() or "4 k" in rag_answer.lower() or "4k" in rag_answer.lower()
        print(f"  Document-consistent answer ('4 Kelvin'): {has_temp}")
        assert has_temp, f"Gemma answer did not contain expected temperature: {rag_answer}"

        # 7. Strict Multi-Tenant Isolation Verification
        print("\n--- STEP 4: Adversarial Cross-Tenant Query via API (Tenant Beta) ---")
        rival_resp = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key_b_raw}"},
            json={
                "model": "gemma-gpu",
                "messages": [
                    {"role": "user", "content": "What is the operating temperature of the Aurora quantum sensor?"}
                ],
                "enable_rag": True,
                "max_tokens": 50,
            },
        )
        print(f"Status Code: {rival_resp.status_code}")
        assert rival_resp.status_code == 200, rival_resp.text
        rival_data = rival_resp.json()
        rival_answer = rival_data["choices"][0]["message"]["content"]
        print(f"✓ Tenant Beta Answer (without context): '{rival_answer.strip()}'")
        assert "4 kelvin" not in rival_answer.lower(), "Tenant Beta leaked Tenant Alpha's context!"
        print("✓ Strict Tenant Isolation Verified: Tenant Beta did not receive Tenant Alpha's document context.")

    print("\n" + "=" * 65)
    print("ALL LIVE GEMMA 4 GPU & POSTGRESQL RAG TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
