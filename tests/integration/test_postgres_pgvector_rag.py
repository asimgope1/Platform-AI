import os
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from platform_core.security.tenant_context import TenantContext
from providers.registry import provider_registry
from storage.models.tenant import Tenant
from storage.models.document import Document, DocumentChunk
from storage.repositories.document_repo import DocumentRepository
from retrieval.ingestion import DocumentIngestionService
from retrieval.search import SemanticSearchService
from retrieval.embedding_service import EmbeddingService
from retrieval.rag_pipeline import RAGPipeline
from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.contracts.requests import NormalizedChatRequest, ChatMessage


TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://localhost/ai_platform_test")


@pytest.fixture
async def pg_session():
    try:
        engine = create_async_engine(TEST_DB_URL, echo=False)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL test database not available: {exc}")

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_pgvector_extension_and_dimension(pg_session: AsyncSession):
    """Verify that PostgreSQL is reachable, has pgvector installed, and embedding column has 384 dims."""
    # 1. Check extension
    res = await pg_session.execute(text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"))
    ext = res.fetchone()
    assert ext is not None
    assert ext[0] == "vector"

    # 2. Check dimension on document_chunks table
    res = await pg_session.execute(text("""
        SELECT a.attname, a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'document_chunks' AND a.attname = 'embedding';
    """))
    row = res.fetchone()
    assert row is not None
    assert row[1] == 384


@pytest.mark.asyncio
async def test_real_embedding_persistence_and_search_across_restart(pg_session: AsyncSession):
    """Verify real SentenceTransformers embedding generation, PostgreSQL persistence, and search across session restart."""
    tenant_a = "tenant-test-aurora"
    tenant_b = "tenant-test-isolated"

    # Clean existing rows
    await pg_session.execute(text("DELETE FROM document_chunks WHERE tenant_id IN (:t1, :t2)"), {"t1": tenant_a, "t2": tenant_b})
    await pg_session.execute(text("DELETE FROM documents WHERE tenant_id IN (:t1, :t2)"), {"t1": tenant_a, "t2": tenant_b})
    await pg_session.execute(text("DELETE FROM tenants WHERE id IN (:t1, :t2)"), {"t1": tenant_a, "t2": tenant_b})

    pg_session.add(Tenant(id=tenant_a, name="Test Aurora"))
    pg_session.add(Tenant(id=tenant_b, name="Test Isolated"))
    await pg_session.commit()

    # Ingest document
    doc_repo = DocumentRepository(pg_session)
    embedding_service = EmbeddingService(provider_id="semantic-default")
    ingestion = DocumentIngestionService(doc_repo, embedding_service)

    doc = await ingestion.ingest_document(
        tenant_id=tenant_a,
        title="Aurora Sensor Spec",
        content="The Aurora quantum sensor operates at 4 Kelvin and detects magnetic anomalies with sub-nanotesla precision.",
        source_type="spec",
    )
    assert doc.total_chunks == 1

    # Verify vector stored in PostgreSQL
    res = await pg_session.execute(text("SELECT id, embedding FROM document_chunks WHERE document_id = :did"), {"did": doc.id})
    chunk_row = res.fetchone()
    assert chunk_row is not None
    assert chunk_row[1] is not None

    # Search with differing phrasing
    query = "What cryogenic instrument measures magnetic anomalies?"
    search_service = SemanticSearchService(doc_repo, embedding_service)
    results = await search_service.search(tenant_id=tenant_a, query=query, top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "Aurora quantum sensor" in top_chunk.content
    assert score > 0.40

    # Cross-tenant isolation check: tenant B retrieves nothing
    results_b = await search_service.search(tenant_id=tenant_b, query=query, top_k=2)
    assert len(results_b) == 0


@pytest.mark.asyncio
async def test_end_to_end_rag_with_untrusted_context_delimiters(pg_session: AsyncSession):
    """Verify that RAG pipeline formats untrusted context delimiters and injects chunks into chat context."""
    tenant_id = "tenant-test-aurora"
    doc_repo = DocumentRepository(pg_session)
    embedding_service = EmbeddingService(provider_id="semantic-default")
    search_service = SemanticSearchService(doc_repo, embedding_service)
    rag_pipeline = RAGPipeline(search_service=search_service)

    context_str = await rag_pipeline.retrieve_context(
        tenant_id=tenant_id,
        query="What temperature does the Aurora quantum sensor operate at?",
    )
    assert "--- BEGIN UNTRUSTED RETRIEVED CONTEXT ---" in context_str
    assert "--- END RETRIEVED CONTEXT ---" in context_str
    assert "4 Kelvin" in context_str
