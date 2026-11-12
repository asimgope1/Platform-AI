import asyncio
import io
import json
import sqlite3
import tempfile
import uuid
import zipfile
import zlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from data_access.entity_mapping import EntityMappingConfig, EntityMappingSpec
from data_access.service import UniversalDataAccessService
from data_access.sync.models import SyncDeadLetter, SyncEvent, SyncRecordState
from data_access.sync.service import RealTimeSyncService
from orchestration.chat_orchestrator import ChatOrchestrator
from orchestration.tool_loop import ToolLoopRunner
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import (
    ChatMessage,
    FunctionCall,
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ToolCall,
    UsageInfo,
)
from platform_core.errors.exceptions import PermissionDeniedError
from platform_core.security.api_keys import hash_api_key
from platform_core.security.tenant_context import TenantContext
from providers.registry import provider_registry
from retrieval.parsers import extract_document_text
from services.api.main import create_app
from services.conversation.response_generator import RAGFirstResponseGenerator
from services.evaluation.retrieval_evaluator import RetrievalConfidenceTier, RetrievalEvaluation
from services.knowledge.retrieval_service import RetrievedContext
from storage.database import Base, get_db_session
from storage.models.api_key import ApiKey
from storage.models.tenant import Tenant
from telephony.ivr_menu import IVRState
from telephony.ivr_prompts import WELCOME_AND_LANGUAGE
from telephony.ivr_session import BARGE_IN_MIN_CONSECUTIVE_FRAMES, IVRCallSession
from tools.executor import ToolExecutor
from tools.registry import tool_registry


def _build_minimal_docx(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
        )
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


def _build_minimal_pdf(text: str) -> bytes:
    stream = zlib.compress(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream\nendobj\n%%EOF"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 7: Dynamic Tool Orchestration, Time Budgets, Clarification & Dedup
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stage7_tool_loop_time_budget_and_iteration_caps():
    class _LoopingAdapter:
        def __init__(self):
            self.calls = 0

        async def chat_completion(self, request, model_meta):
            self.calls += 1
            return NormalizedChatResponse(
                id=f"c-{self.calls}",
                model=model_meta.alias,
                created=1,
                choices=[
                    NormalizedChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id=f"tc-{self.calls}",
                                    function=FunctionCall(
                                        name="count_open_purchase_requests",
                                        arguments='{"filter_status": "open"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(),
            )

    meta = ModelMetadata(alias="mock-chat", provider_id="mock", underlying_model="mock", capabilities=["chat", "tool_calling"])
    executor = ToolExecutor()
    runner = ToolLoopRunner(tool_executor=executor, max_iterations=3, max_execution_seconds=30.0)
    adapter = _LoopingAdapter()
    req = NormalizedChatRequest(model_alias="mock-chat", messages=[ChatMessage(role="user", content="loop test")])
    ctx = TenantContext(tenant_id="t1", tenant_name="T1", api_key_id="k1", scopes={"*"})

    resp = await runner.run(adapter, req, meta, context=ctx)
    assert adapter.calls == 3
    assert resp.id == "c-3"

    # Execution time budget stops early cleanly
    slow_runner = ToolLoopRunner(tool_executor=executor, max_iterations=5, max_execution_seconds=0.01)

    class _SlowAdapter(_LoopingAdapter):
        async def chat_completion(self, request, model_meta):
            await asyncio.sleep(0.02)
            return await super().chat_completion(request, model_meta)

    slow_adapter = _SlowAdapter()
    await slow_runner.run(slow_adapter, req, meta, context=ctx)
    assert slow_adapter.calls == 1


# ──────────────────────────────────────────────────────────────────────────────
# Stage 8: Multi-Format Document Ingestion, Re-indexing, Citations, PII Guard
# ──────────────────────────────────────────────────────────────────────────────


def test_stage8_document_parsers_and_pii_guard():
    st_docx, txt_docx = extract_document_text("guide.docx", _build_minimal_docx("Warranty lasts 24 months."))
    assert st_docx == "docx" and "Warranty lasts 24 months." in txt_docx

    st_pdf, txt_pdf = extract_document_text("terms.pdf", _build_minimal_pdf("Standard shipping takes 3 days."))
    assert st_pdf == "pdf" and "Standard shipping takes 3 days." in txt_pdf

    st_csv, txt_csv = extract_document_text("faq.csv", b"topic,answer\nRefund,Processed in 48 hours\n")
    assert st_csv == "csv" and "Processed in 48 hours" in txt_csv

    # Reject raw transactional PII CSV dump
    with pytest.raises(PermissionDeniedError, match="transactional PII"):
        extract_document_text(
            "cards.csv",
            b"customer_id,card_number,cvv\nC1,4111111111111111,123\n",
        )


@pytest.mark.asyncio
async def test_stage8_api_upload_reindex_search_delete_and_tenant_isolation():
    provider_registry.initialize_from_file("config/providers.yaml")
    app = create_app()
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tf.name}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async def _override_db():
            async with session_factory() as sess:
                yield sess

        app.dependency_overrides[get_db_session] = _override_db

        raw_k1 = "sk_live_tenant1_stage8_key"
        raw_k2 = "sk_live_tenant2_stage8_key"
        async with session_factory() as sess:
            t1 = Tenant(id=str(uuid.uuid4()), name="TenantOne", is_active=True)
            t2 = Tenant(id=str(uuid.uuid4()), name="TenantTwo", is_active=True)
            sess.add_all([t1, t2])
            await sess.flush()
            sess.add_all(
                [
                    ApiKey(tenant_id=t1.id, key_hash=hash_api_key(raw_k1), name="k1", scopes="documents:read,documents:write,chat:write"),
                    ApiKey(tenant_id=t2.id, key_hash=hash_api_key(raw_k2), name="k2", scopes="documents:read,documents:write,chat:write"),
                ]
            )
            await sess.commit()

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            h1 = {"Authorization": f"Bearer {raw_k1}"}
            h2 = {"Authorization": f"Bearer {raw_k2}"}

            up = await client.post(
                "/v1/documents/upload",
                headers=h1,
                files={"file": ("policy.docx", _build_minimal_docx("Express return window is 14 days for electronics."), "application/octet-stream")},
            )
            assert up.status_code == 201, up.text
            doc_id = up.json()["id"]

            # Search as tenant 1 -> returns match + citation
            s1 = await client.post(
                "/v1/documents/search",
                headers=h1,
                json={"query": "return window electronics", "top_k": 3, "hybrid": True},
            )
            assert s1.status_code == 200
            items1 = s1.json()["data"]
            assert len(items1) >= 1 and "14 days" in items1[0]["content"]
            assert items1[0]["citation"].startswith(f"doc:{doc_id}#chunk-")

            # Cross-tenant isolation: tenant 2 gets 0 matches
            s2 = await client.post("/v1/documents/search", headers=h2, json={"query": "return window electronics"})
            assert s2.status_code == 200 and s2.json()["data"] == []

            # Reindex with v2 content
            ri = await client.post(
                f"/v1/documents/{doc_id}/reindex",
                headers=h1,
                json={"content": "Updated express return window is 30 days for all items.", "version": "2.0"},
            )
            assert ri.status_code == 200
            s1_v2 = await client.post("/v1/documents/search", headers=h1, json={"query": "return window"})
            assert "30 days" in s1_v2.json()["data"][0]["content"]

            # Tenant 2 cannot delete tenant 1 doc; tenant 1 can
            d2 = await client.delete(f"/v1/documents/{doc_id}", headers=h2)
            assert d2.status_code == 404
            d1 = await client.delete(f"/v1/documents/{doc_id}", headers=h1)
            assert d1.status_code == 200 and d1.json()["deleted"] is True

        await engine.dispose()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 9: Real-Time Sync Pipeline (Outbox, Idempotency, Version, DLQ, Tombstone)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stage9_sync_idempotency_version_tombstone_dlq_reconcile_and_polling():
    with tempfile.NamedTemporaryFile(suffix=".db") as internal_db, tempfile.NamedTemporaryFile(suffix=".db") as ext_db:
        conn = sqlite3.connect(ext_db.name)
        conn.execute("CREATE TABLE catalog_products (pid TEXT PRIMARY KEY, sku TEXT, title TEXT, cat TEXT, price REAL, stock TEXT)")
        conn.execute("INSERT INTO catalog_products VALUES ('P10', 'SKU-10', 'Smart Gateway v1', 'IoT', 299.0, 'in_stock')")
        conn.commit()
        conn.close()

        engine = create_async_engine(f"sqlite+aiosqlite:///{internal_db.name}")
        async with engine.begin() as c:
            await c.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with session_factory() as session:
            tid = str(uuid.uuid4())
            session.add(Tenant(id=tid, name="SyncTenant", is_active=True))
            await session.commit()

            svc = RealTimeSyncService(session, base_retry_seconds=0.01)

            # Reject transactional entities cleanly
            with pytest.raises(PermissionDeniedError, match="Transactional"):
                await svc.enqueue_event(
                    tenant_id=tid,
                    entity_type="order",
                    record_id="ORD-1",
                    event_type="created",
                    payload={"status": "shipped"},
                )

            # 1. Enqueue v2 first + duplicate idempotency check
            ev2, created2 = await svc.enqueue_event(
                tenant_id=tid,
                entity_type="faq_policy",
                record_id="FAQ-1",
                event_type="created",
                version=2,
                title="UPI Refund Window",
                payload={"content": "UPI refunds are settled within 24 hours."},
                idempotency_key="faq-1-v2",
            )
            assert created2 is True
            _, dup = await svc.enqueue_event(
                tenant_id=tid,
                entity_type="faq_policy",
                record_id="FAQ-1",
                event_type="created",
                version=2,
                title="UPI Refund Window",
                payload={"content": "UPI refunds are settled within 24 hours."},
                idempotency_key="faq-1-v2",
            )
            assert dup is False
            assert await svc.process_event(ev2) == "completed"

            # 2. Out-of-order stale v1 ignored cleanly
            ev1, _ = await svc.enqueue_event(
                tenant_id=tid,
                entity_type="faq_policy",
                record_id="FAQ-1",
                event_type="updated",
                version=1,
                title="UPI Refund Window Old",
                payload={"content": "Old 7 day text"},
            )
            assert await svc.process_event(ev1) == "ignored_stale"

            # 3. Soft-delete tombstone removes chunks & records tombstone
            ev_del, _ = await svc.enqueue_event(
                tenant_id=tid,
                entity_type="faq_policy",
                record_id="FAQ-1",
                event_type="deleted",
                version=3,
            )
            assert await svc.process_event(ev_del) == "completed"
            status_after_del = await svc.get_status(tid)
            assert status_after_del["tombstones"] == 1
            assert status_after_del["active_synced_records"] == 0

            # 4. Retry + exponential backoff -> Dead Letter Queue
            ev_fail, _ = await svc.enqueue_event(
                tenant_id=tid,
                entity_type="support_article",
                record_id="ART-FAIL",
                event_type="created",
                version=1,
                payload={"content": "Valid article text"},
                max_retries=2,
            )

            async def _boom(_ev):
                raise RuntimeError("Transient embedding timeout")

            assert await svc.process_event(ev_fail, ingestion_override=_boom) == "retry_scheduled"
            assert ev_fail.next_retry_at is not None
            assert await svc.process_event(ev_fail, ingestion_override=_boom) == "dead_letter"
            st_dlq = await svc.get_status(tid)
            assert len(st_dlq["dead_letters"]) == 1

            # 5. Reconcile snapshot adds missing & tombstones absent
            rec_out = await svc.reconcile_snapshot(
                tid,
                entity_type="support_article",
                records=[{"record_id": "ART-NEW", "version": 1, "title": "Guide", "payload": {"content": "Hello"}}],
            )
            assert rec_out["upserts_enqueued"] == 1 and rec_out["completed"] == 1

            # 6. Incremental product polling via UniversalDataAccessService
            uda = UniversalDataAccessService(session)
            cfg = await uda.register_config(tenant_id=tid, name="ext-cat", engine="sqlite", database_name=ext_db.name)
            await uda.test_config_connection(tid, cfg.id)
            await uda.discover_and_save_schema(tid, cfg.id)
            await uda.update_mapping(
                tid,
                cfg.id,
                EntityMappingConfig(
                    entities={
                        "product": EntityMappingSpec(
                            table="catalog_products",
                            primary_key="pid",
                            fields={"product_id": "pid", "sku": "sku", "name": "title", "category": "cat", "price": "price", "stock_status": "stock"},
                        )
                    }
                ),
            )
            await uda.validate_mapping(tid, cfg.id)
            await uda.activate_config(tid, cfg.id)
            poll_out = await svc.poll_incremental_product_catalog(tid)
            assert poll_out["polled_rows"] == 1 and poll_out["enqueued"] == 1

        await engine.dispose()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 10: Voice / IVR Multi-Language (EN/HI/OR) & Resilience Matrix
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stage10_voice_ivr_multilingual_barge_in_failures_and_context():
    # 1. Confirm EN, HI, OR prompt dictionaries exist cleanly
    for lang in ("en", "hi", "or"):
        assert lang in WELCOME_AND_LANGUAGE

    # 2. Voice barge-in with sustained speech >= BARGE_IN_MIN_CONSECUTIVE_FRAMES + transcript barge-in
    session = IVRCallSession(ws=AsyncMock(), orchestrator=AsyncMock(), audio_service=AsyncMock())
    session.menu_machine.state = IVRState.AI_CONVERSATION
    session._is_playing = True
    dummy_task = asyncio.create_task(asyncio.sleep(10))
    session._playback_task = dummy_task

    loud = (np.sin(np.linspace(0, 12.0, 160)) * 8000).astype("<i2").tobytes()
    for _ in range(BARGE_IN_MIN_CONSECUTIVE_FRAMES):
        await session.on_pcm_frame(loud)
    await asyncio.sleep(0)
    assert session._is_playing is False
    assert dummy_task.cancelling() > 0 or dummy_task.cancelled()

    # 3. Silence / no-speech frame does not accumulate speech utterance
    silent_pcm = np.zeros(160, dtype="<i2").tobytes()
    session._pcm_buffer.clear()
    await session.on_pcm_frame(silent_pcm)
    assert len(session._pcm_buffer) == 0

    class _MockRetrieval:
        async def search(self, **_kwargs):
            return RetrievedContext(
                chunks=[],
                chunk_texts=[],
                scored_chunks=[],
                evaluation=RetrievalEvaluation(
                    tier=RetrievalConfidenceTier.NONE, top_score=0.0, verified_hits=0, total_hits=0, reason="none"
                ),
            )

    # 4. RAGFirstResponseGenerator with backend unavailable never fabricates orders in EN/HI/OR
    class _FailOrch:
        async def execute(self, *_args, **_kwargs):
            raise RuntimeError("remote_backend_unavailable")

    gen = RAGFirstResponseGenerator(retrieval_service=_MockRetrieval())
    for lang in ("en", "hi", "or"):
        res = await gen.generate_response(
            user_message="Where is my order ORD-2005?",
            conversation_history=[],
            tenant_id="879e9bda-8049-4824-9aa0-fc5e9f0012d9",
            language=lang,
            orchestrator=_FailOrch(),
            tools=[{"type": "function", "function": {"name": "get_order_status"}}],
            authorized_customer_id="CUST-100",
        )
        assert res.source == "tool_unavailable_fallback"
        assert "ORD-2005" not in res.text
        assert res.used_fallback is True

    # 5. Multi-turn context & authorized_customer_id propagation when tools active
    captured_reqs = []

    class _RecordingOrch:
        async def execute(self, req, context=None, intelligence_context=None):
            captured_reqs.append((req, context))
            return NormalizedChatResponse(
                id="ok-1",
                model="gemma4",
                created=1,
                choices=[NormalizedChoice(index=0, message=ChatMessage(role="assistant", content="Your order is out for delivery."))],
                usage=UsageInfo(),
            )

    gen_ok = RAGFirstResponseGenerator(retrieval_service=_MockRetrieval())
    ctx = TenantContext(tenant_id="879e9bda-8049-4824-9aa0-fc5e9f0012d9", tenant_name="VoiceTest", api_key_id="k", scopes={"*"})
    hist = [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="How can I help?")]
    out = await gen_ok.generate_response(
        user_message="Check my order please",
        conversation_history=hist,
        tenant_id=ctx.tenant_id,
        language="en",
        orchestrator=_RecordingOrch(),
        context=ctx,
        tools=[{"type": "function", "function": {"name": "get_order_status"}}],
        authorized_customer_id="CUST_456",
    )
    assert out.text == "Your order is out for delivery."
    sent_req, sent_ctx = captured_reqs[0]
    assert sent_req.metadata["authorized_customer_id"] == "CUST_456"
    assert len(sent_req.messages) >= 3
    assert sent_ctx.tenant_id == ctx.tenant_id
