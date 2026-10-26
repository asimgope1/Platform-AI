import pytest
from retrieval.chunking import chunk_text
from providers.embeddings.base import MockEmbeddingAdapter
from providers.vector_stores.base import InMemoryVectorStore, cosine_similarity
from providers.rerankers.base import MockReranker


def test_chunking():
    text = "Paragraph one is here.\n\nParagraph two has more information.\n\nParagraph three concludes it."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


@pytest.mark.asyncio
async def test_mock_embedding_adapter_deterministic():
    adapter = MockEmbeddingAdapter("mock", {}, dimension=128)
    res1 = await adapter.generate_embeddings(["Antigravity AI Platform"], "text-embedding-3-small")
    res2 = await adapter.generate_embeddings(["Antigravity AI Platform"], "text-embedding-3-small")

    assert len(res1) == 1
    assert len(res1[0]) == 128
    assert res1[0] == res2[0]


@pytest.mark.asyncio
async def test_in_memory_vector_store():
    store = InMemoryVectorStore("mock_store", {})
    tenant_id = "tenant-test"

    chunks = [
        {"id": "c1", "content": "Revenue was up 20% in Q3", "embedding": [1.0, 0.0, 0.0]},
        {"id": "c2", "content": "Database latency was 5ms", "embedding": [0.0, 1.0, 0.0]},
    ]
    await store.upsert(tenant_id=tenant_id, document_id="doc1", chunks=chunks)

    # Query with vector matching c1
    results = await store.query(tenant_id=tenant_id, query_embedding=[1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "c1"
    assert results[0]["score"] == 1.0


@pytest.mark.asyncio
async def test_mock_reranker():
    reranker = MockReranker("mock_reranker", {})
    docs = [
        "Unrelated document about cooking recipes.",
        "Sales report showing high revenue growth.",
        "Weather forecast for tomorrow.",
    ]
    ranked = await reranker.rerank(query="sales revenue growth", documents=docs, top_n=2)
    assert len(ranked) == 2
    assert ranked[0]["index"] == 1
    assert ranked[0]["relevance_score"] > 0
