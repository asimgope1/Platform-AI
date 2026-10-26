import pytest
import math
from providers.embeddings.sentence_transformers import SentenceTransformersEmbeddingAdapter
from providers.embeddings.base import MockEmbeddingAdapter


@pytest.mark.asyncio
async def test_embedding_dimensions_and_batching():
    """Verify that Sentence Transformers adapter returns 384 dimensions and preserves batch length."""
    adapter = SentenceTransformersEmbeddingAdapter(
        provider_id="sentence_transformers",
        config={"model_id": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384},
    )

    inputs = [
        "First test document about AI systems.",
        "Second document discussing vector databases.",
        "Third document on prompt engineering.",
    ]

    # Test batch embedding
    vectors = await adapter.generate_embeddings(inputs)
    assert len(vectors) == 3

    for vec in vectors:
        # Expected dimension is 384
        assert len(vec) == 384
        # All values must be finite floats
        assert all(isinstance(x, float) and not math.isnan(x) and not math.isinf(x) for x in vec)
        # Verify L2 normalization (norm approximately 1.0)
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-3


@pytest.mark.asyncio
async def test_embedding_empty_input():
    adapter = SentenceTransformersEmbeddingAdapter(
        provider_id="sentence_transformers",
        config={"model_id": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384},
    )

    vectors = await adapter.generate_embeddings([])
    assert vectors == []


@pytest.mark.asyncio
async def test_mock_embedding_adapter_parity():
    """Ensure mock embedding adapter also correctly respects custom dimension parameter."""
    adapter = MockEmbeddingAdapter("mock", {}, dimension=384)
    vectors = await adapter.generate_embeddings(["Parity check text"], "all-MiniLM-L6-v2")
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
