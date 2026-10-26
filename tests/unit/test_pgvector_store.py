import pytest
from providers.vector_stores.pgvector import PGVectorStore


@pytest.mark.asyncio
async def test_pgvector_store_upsert_and_dimension_validation():
    store = PGVectorStore("pgvector", {"dimensions": 384})

    # Valid 384-dimension chunk
    valid_chunk = {
        "id": "c_valid",
        "document_id": "doc_1",
        "content": "Valid embedding content",
        "embedding": [0.1] * 384,
    }
    await store.upsert(tenant_id="tenant_x", document_id="doc_1", chunks=[valid_chunk])

    # Query with valid 384 dimension
    results = await store.search(query_embedding=[0.1] * 384, tenant_id="tenant_x", limit=5)
    assert len(results) == 1
    assert results[0]["id"] == "c_valid"

    # Invalid dimension (128 values instead of 384) must be rejected
    invalid_chunk = {
        "id": "c_invalid",
        "document_id": "doc_1",
        "content": "Invalid embedding content",
        "embedding": [0.1] * 128,
    }
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        await store.upsert(tenant_id="tenant_x", document_id="doc_1", chunks=[invalid_chunk])

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        await store.search(query_embedding=[0.1] * 128, tenant_id="tenant_x")


@pytest.mark.asyncio
async def test_pgvector_store_strict_tenant_isolation():
    store = PGVectorStore("pgvector", {"dimensions": 384})

    chunk_a = {
        "id": "chunk_tenant_a",
        "document_id": "doc_a",
        "content": "Tenant A exclusive engineering manual",
        "embedding": [0.5] * 384,
    }
    chunk_b = {
        "id": "chunk_tenant_b",
        "document_id": "doc_b",
        "content": "Tenant B competitor product list",
        "embedding": [0.5] * 384,
    }

    await store.upsert(tenant_id="tenant_a", document_id="doc_a", chunks=[chunk_a])
    await store.upsert(tenant_id="tenant_b", document_id="doc_b", chunks=[chunk_b])

    # Tenant A searches: must ONLY return Tenant A's chunk
    res_a = await store.search(query_embedding=[0.5] * 384, tenant_id="tenant_a")
    assert len(res_a) == 1
    assert res_a[0]["id"] == "chunk_tenant_a"
    assert res_a[0]["tenant_id"] == "tenant_a"

    # Tenant B searches: must ONLY return Tenant B's chunk
    res_b = await store.search(query_embedding=[0.5] * 384, tenant_id="tenant_b")
    assert len(res_b) == 1
    assert res_b[0]["id"] == "chunk_tenant_b"
    assert res_b[0]["tenant_id"] == "tenant_b"


@pytest.mark.asyncio
async def test_pgvector_store_limit_enforcement():
    store = PGVectorStore("pgvector", {"dimensions": 384})

    chunks = [
        {"id": f"chunk_{i}", "document_id": "doc_multi", "content": f"Content {i}", "embedding": [0.1 * i] * 384}
        for i in range(10)
    ]
    await store.upsert(tenant_id="tenant_multi", document_id="doc_multi", chunks=chunks)

    # Search with limit 3
    res = await store.search(query_embedding=[0.5] * 384, tenant_id="tenant_multi", limit=3)
    assert len(res) == 3
