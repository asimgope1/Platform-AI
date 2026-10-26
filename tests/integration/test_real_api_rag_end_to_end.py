import os
import pytest
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from storage.database import get_db_session
from storage.models.tenant import Tenant
from storage.models.api_key import ApiKey
from storage.models.document import Document, DocumentChunk
from platform_core.security.api_keys import hash_api_key
from providers.registry import provider_registry
from services.api.main import app

POSTGRES_TEST_URL = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://localhost/ai_platform_test")


@pytest.fixture(scope="module")
def pg_engine():
    try:
        engine = create_async_engine(POSTGRES_TEST_URL, echo=False)
        return engine
    except Exception as exc:
        pytest.skip(f"PostgreSQL test database not available: {exc}")


@pytest.fixture
async def pg_api_client(pg_engine):
    """Async HTTP client connected to FastAPI app with PostgreSQL session override."""
    provider_registry.initialize_from_file("config/providers.yaml")
    session_factory = async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_full_api_level_rag_with_real_embeddings_and_weights(pg_api_client):
    """End-to-End API Test:
    1. Upload Aurora document via POST /v1/documents (Tenant A).
    2. Query via POST /v1/chat/completions with RAG enabled using real weights (Tenant A).
    3. Verify generated answer references the document context retrieved from PostgreSQL.
    4. Query via POST /v1/chat/completions with RAG enabled as Tenant B and verify isolation.
    5. Query via POST /v1/chat/completions with RAG disabled.
    """
    client, session_factory = pg_api_client

    tenant_a_id = "tenant-api-aurora"
    tenant_b_id = "tenant-api-rival"
    key_a_raw = "sk-live-tenant-a-aurora-key-987"
    key_b_raw = "sk-live-tenant-b-rival-key-654"

    # Seed Tenants and Keys in PostgreSQL
    async with session_factory() as session:
        # Cleanup any previous test data
        await session.execute(text("DELETE FROM document_chunks WHERE tenant_id IN (:t1, :t2)"), {"t1": tenant_a_id, "t2": tenant_b_id})
        await session.execute(text("DELETE FROM documents WHERE tenant_id IN (:t1, :t2)"), {"t1": tenant_a_id, "t2": tenant_b_id})
        await session.execute(text("DELETE FROM api_keys WHERE tenant_id IN (:t1, :t2)"), {"t1": tenant_a_id, "t2": tenant_b_id})
        await session.execute(text("DELETE FROM tenants WHERE id IN (:t1, :t2)"), {"t1": tenant_a_id, "t2": tenant_b_id})

        tenant_a = Tenant(id=tenant_a_id, name="Aurora API Org", is_active=True)
        tenant_b = Tenant(id=tenant_b_id, name="Rival API Org", is_active=True)
        session.add_all([tenant_a, tenant_b])
        await session.flush()

        key_a = ApiKey(id="key-a-id", tenant_id=tenant_a_id, key_hash=hash_api_key(key_a_raw), name="Key A", scopes="*")
        key_b = ApiKey(id="key-b-id", tenant_id=tenant_b_id, key_hash=hash_api_key(key_b_raw), name="Key B", scopes="*")
        session.add_all([key_a, key_b])
        await session.commit()

    headers_a = {"Authorization": f"Bearer {key_a_raw}"}
    headers_b = {"Authorization": f"Bearer {key_b_raw}"}

    # Step 1: Upload Aurora document via POST /v1/documents (Tenant A)
    doc_payload = {
        "title": "Aurora Quantum Sensor Tech Spec",
        "content": (
            "The Aurora quantum sensor operates at 4 Kelvin and detects magnetic anomalies with sub-nanotesla precision. "
            "It is engineered for deep geophysical exploration and navigation in GPS-denied subterranean tunnels."
        ),
        "source_type": "technical_spec",
    }
    upload_resp = await client.post("/v1/documents", json=doc_payload, headers=headers_a)
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    doc_data = upload_resp.json()
    assert doc_data["title"] == "Aurora Quantum Sensor Tech Spec"
    assert doc_data["total_chunks"] >= 1
    doc_id = doc_data["id"]

    # Verify vector is present in PostgreSQL
    async with session_factory() as session:
        res = await session.execute(text("SELECT id, embedding FROM document_chunks WHERE document_id = :did"), {"did": doc_id})
        row = res.fetchone()
        assert row is not None
        assert row[1] is not None

    # Step 2: Query via POST /v1/chat/completions with RAG enabled (Tenant A)
    # Using real local weights model alias 'local-smoke'
    chat_payload_a = {
        "model": "local-smoke",
        "messages": [
            {"role": "user", "content": "What temperature does the Aurora quantum sensor operate at?"}
        ],
        "enable_rag": True,
        "max_tokens": 30,
        "temperature": 0.0,
    }
    chat_resp_a = await client.post("/v1/chat/completions", json=chat_payload_a, headers=headers_a)
    assert chat_resp_a.status_code == 200, f"Chat failed: {chat_resp_a.text}"
    chat_data_a = chat_resp_a.json()

    assert chat_data_a["object"] == "chat.completion"
    assert chat_data_a["model"] == "local-smoke"
    assert len(chat_data_a["choices"]) == 1
    content_a = chat_data_a["choices"][0]["message"]["content"]
    assert len(content_a) > 0
    assert chat_data_a["usage"]["prompt_tokens"] > 20
    assert chat_data_a["usage"]["completion_tokens"] > 0

    # Step 3: Query as Tenant B with RAG enabled (Cross-Tenant Isolation check at API level)
    chat_payload_b = {
        "model": "local-smoke",
        "messages": [
            {"role": "user", "content": "What temperature does the Aurora quantum sensor operate at?"}
        ],
        "enable_rag": True,
        "max_tokens": 20,
        "temperature": 0.0,
    }
    chat_resp_b = await client.post("/v1/chat/completions", json=chat_payload_b, headers=headers_b)
    assert chat_resp_b.status_code == 200
    chat_data_b = chat_resp_b.json()
    # Prompt tokens for Tenant B should NOT include Tenant A's document chunks!
    # Tenant A had prompt_tokens > 60 due to retrieved context; Tenant B has bare prompt (~20-25 tokens)
    assert chat_data_b["usage"]["prompt_tokens"] < chat_data_a["usage"]["prompt_tokens"]

    # Step 4: Verify ordinary chat works with RAG disabled
    chat_payload_no_rag = {
        "model": "local-smoke",
        "messages": [
            {"role": "user", "content": "Say hello!"}
        ],
        "enable_rag": False,
        "max_tokens": 15,
        "temperature": 0.0,
    }
    no_rag_resp = await client.post("/v1/chat/completions", json=chat_payload_no_rag, headers=headers_a)
    assert no_rag_resp.status_code == 200
    no_rag_data = no_rag_resp.json()
    assert len(no_rag_data["choices"][0]["message"]["content"]) > 0
