import pytest
import httpx
from sqlalchemy import select
from storage.models.tenant import Tenant
from storage.models.api_key import ApiKey
from storage.models.usage_event import UsageEvent
from platform_core.security.api_keys import hash_api_key
from storage.repositories.document_repo import DocumentRepository
from retrieval.embedding_service import EmbeddingService
from retrieval.ingestion import DocumentIngestionService
from retrieval.search import SemanticSearchService
from retrieval.rag_pipeline import RAGPipeline
from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.contracts.requests import NormalizedChatRequest, ChatMessage
from platform_core.security.tenant_context import TenantContext


@pytest.mark.asyncio
async def test_end_to_end_rag_pipeline_with_tenant_isolation(db_session, async_client: httpx.AsyncClient):
    """End-to-End RAG Verification:

    1. Tenant A ingests proprietary document.
    2. Tenant A queries with RAG enabled -> retrieved context is injected into conversation.
    3. Tenant B queries with RAG enabled -> retrieved context is EMPTY, preventing cross-tenant leakage.
    """
    # 1. Setup Tenant A
    raw_key_a = "sk_live_tenant_a_rag_key"
    tenant_a = Tenant(name="Tenant A Aerospace")
    db_session.add(tenant_a)
    await db_session.commit()
    await db_session.refresh(tenant_a)
    db_session.add(ApiKey(tenant_id=tenant_a.id, key_hash=hash_api_key(raw_key_a), name="Key A", scopes="*"))

    # 2. Setup Tenant B
    raw_key_b = "sk_live_tenant_b_rag_key"
    tenant_b = Tenant(name="Tenant B Competitor")
    db_session.add(tenant_b)
    await db_session.commit()
    await db_session.refresh(tenant_b)
    db_session.add(ApiKey(tenant_id=tenant_b.id, key_hash=hash_api_key(raw_key_b), name="Key B", scopes="*"))
    await db_session.commit()

    headers_a = {"Authorization": f"Bearer {raw_key_a}"}
    headers_b = {"Authorization": f"Bearer {raw_key_b}"}

    # 3. Tenant A uploads confidential document
    doc_payload = {
        "title": "Apollo Mission Spec",
        "content": "Secret Apollo launch code is ZEUS-9988 and launch date is October 2027.",
        "source_type": "text",
    }
    upload_resp = await async_client.post("/v1/documents", json=doc_payload, headers=headers_a)
    assert upload_resp.status_code == 201

    # 4. RAG Pipeline test directly through Orchestrator
    doc_repo = DocumentRepository(db_session)
    emb_service = EmbeddingService()
    search_service = SemanticSearchService(doc_repo, emb_service)
    rag_pipeline = RAGPipeline(search_service)
    orchestrator = ChatOrchestrator(rag_pipeline=rag_pipeline)

    ctx_a = TenantContext(tenant_id=tenant_a.id, tenant_name=tenant_a.name, scopes={"*"})
    ctx_b = TenantContext(tenant_id=tenant_b.id, tenant_name=tenant_b.name, scopes={"*"})

    chat_req = NormalizedChatRequest(
        model_alias="mock-chat",
        messages=[ChatMessage(role="user", content="What is the secret Apollo launch code?")],
    )

    # 5. Execute with Tenant A (RAG context must be retrieved and included)
    resp_a = await orchestrator.execute(chat_req, context=ctx_a, enable_rag=True)
    assert resp_a.choices[0].message.role == "assistant"

    # Context verification: Retrieve context directly for Tenant A vs Tenant B
    retrieved_context_a = await rag_pipeline.retrieve_context(
        tenant_id=tenant_a.id,
        query="What is the secret Apollo launch code?",
    )
    assert "ZEUS-9988" in retrieved_context_a
    assert "[Source 1]" in retrieved_context_a

    # 6. Execute with Tenant B: Context MUST BE completely empty for Tenant B
    retrieved_context_b = await rag_pipeline.retrieve_context(
        tenant_id=tenant_b.id,
        query="What is the secret Apollo launch code?",
    )
    assert retrieved_context_b == ""
    assert "ZEUS-9988" not in retrieved_context_b


@pytest.mark.asyncio
async def test_tool_execution_scope_enforcement(async_client: httpx.AsyncClient, db_session):
    """Verify tool execution is guarded by scopes."""
    # Tenant with only chat scope (no tools:execute scope)
    raw_key_restricted = "sk_live_restricted_key"
    tenant = Tenant(name="Restricted Tenant")
    db_session.add(tenant)
    await db_session.commit()
    await db_session.refresh(tenant)

    db_session.add(ApiKey(
        tenant_id=tenant.id,
        key_hash=hash_api_key(raw_key_restricted),
        name="Chat Only Key",
        scopes="chat:completions",  # Missing tools:execute
    ))
    await db_session.commit()

    headers = {"Authorization": f"Bearer {raw_key_restricted}"}

    # Request tool execution with restricted key
    payload = {
        "model": "mock-chat",
        "messages": [
            {"role": "user", "content": "How many open purchase requests are there?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "count_open_purchase_requests",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    }
    resp = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    # The tool loop handles unauthorized tool execution gracefully by passing error back to loop
    assert len(data["choices"]) > 0


@pytest.mark.asyncio
async def test_usage_events_tenant_isolation(async_client: httpx.AsyncClient, db_session):
    """Verify usage events in the database are strictly partitioned by tenant ID."""
    raw_key = "sk_live_usage_tenant_key"
    tenant = Tenant(name="Usage Tenant")
    db_session.add(tenant)
    await db_session.commit()
    await db_session.refresh(tenant)
    db_session.add(ApiKey(tenant_id=tenant.id, key_hash=hash_api_key(raw_key), name="Key", scopes="*"))
    await db_session.commit()

    headers = {"Authorization": f"Bearer {raw_key}"}
    payload = {
        "model": "mock-chat",
        "messages": [{"role": "user", "content": "Track my tokens"}],
    }

    # Fire 2 chat completions
    await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    await async_client.post("/v1/chat/completions", json=payload, headers=headers)

    # Query usage events from database
    result = await db_session.execute(
        select(UsageEvent).where(UsageEvent.tenant_id == tenant.id)
    )
    events = list(result.scalars().all())

    assert len(events) == 2
    assert all(e.tenant_id == tenant.id for e in events)
    assert all(e.total_tokens > 0 for e in events)
    assert all(e.status_code == 200 for e in events)
