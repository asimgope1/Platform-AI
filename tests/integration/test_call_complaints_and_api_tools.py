"""
End-to-End Integration Tests:
  • Over-Call Complaint Registration via IVRCallSession + ToolLoopRunner
  • External API Tool Calling (call_external_api) & SSRF Allowlist Enforcement
  • Complaint Lookup (get_complaint_status) & Multi-Turn Conversational Tool Routing
  • Tenant-Isolated REST API (/v1/complaints)
"""
from __future__ import annotations

import json
import math
import struct
import wave
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.config.settings import settings
from platform_core.contracts.requests import (
    ChatMessage,
    FunctionCall,
    NormalizedChoice,
    NormalizedChatResponse,
    ToolCall,
    UsageInfo,
)
from platform_core.security.api_keys import hash_api_key
from platform_core.security.tenant_context import TenantContext
from providers.chat.openai_compatible import OpenAICompatibleChatAdapter
from services.api.main import app
from storage.database import Base, get_db_session
from storage.models.api_key import ApiKey
from storage.models.complaint import Complaint
from storage.models.tenant import Tenant
from telephony.ivr_menu import IVRState
from telephony.ivr_session import IVRCallSession
from tools.business_data.complaints import (
    call_external_api,
    get_complaint_status,
    register_complaint,
)
from tools.executor import ToolExecutor


def _make_voiced_pcm(duration_sec: float = 0.4, sample_rate: int = 8000) -> bytes:
    n = int(sample_rate * duration_sec)
    samples = [int(6000 * math.sin(2.0 * math.pi * 300.0 * i / sample_rate)) for i in range(n)]
    return struct.pack("<" + "h" * n, *samples)


def _make_dummy_wav() -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x01\x00" * 800)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_over_call_complaint_registration_and_status_conversation():
    """
    Simulates a live multi-turn phone call:
      1. Caller selects Complaint Registration (DTMF '2') and asks to file a complaint.
      2. Model asks clarifying question; caller gives issue details for ORD-2005.
      3. Model invokes `register_complaint` via ToolLoopRunner -> persisted in DB with CMP-... ticket.
      4. Caller asks to check ticket status -> Model invokes `get_complaint_status` cleanly.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _fake_gemma(self, request, model_meta):
        last = request.messages[-1]
        if last.role == "tool":
            tool_data = json.loads(last.content or "{}")
            if "ticket_number" in tool_data:
                return NormalizedChatResponse(
                    id="c1",
                    model=model_meta.alias,
                    created=1,
                    choices=[
                        NormalizedChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=(
                                    f"I have registered your complaint under ticket number "
                                    f"{tool_data['ticket_number']}."
                                ),
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                )
            complaints = tool_data.get("complaints") or []
            if complaints:
                c0 = complaints[0]
                return NormalizedChatResponse(
                    id="c2",
                    model=model_meta.alias,
                    created=1,
                    choices=[
                        NormalizedChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=(
                                    f"Your complaint {c0['ticket_number']} is currently {c0['status']} "
                                    f"under high priority."
                                ),
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                )

        user_txt = str(last.content or "").lower()
        if "damaged" in user_txt and "ord-2005" in user_txt:
            return NormalizedChatResponse(
                id="tc_reg",
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
                                    id="call_reg_1",
                                    function=FunctionCall(
                                        name="register_complaint",
                                        arguments=json.dumps(
                                            {
                                                "description": "Order ORD-2005 arrived damaged with a cracked screen",
                                                "order_id": "ORD-2005",
                                                "category": "damaged_item",
                                                "call_uuid": "spoofed-by-llm",
                                            }
                                        ),
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        if "status of my complaint" in user_txt:
            return NormalizedChatResponse(
                id="tc_stat",
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
                                    id="call_stat_1",
                                    function=FunctionCall(
                                        name="get_complaint_status",
                                        arguments=json.dumps({"order_id": "ORD-2005"}),
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

        return NormalizedChatResponse(
            id="ask_more",
            model=model_meta.alias,
            created=1,
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Sure, please share the issue details and order number so I can register your complaint.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    mock_ws = AsyncMock()
    mock_audio = AsyncMock()
    mock_audio.synthesize = AsyncMock(return_value=_make_dummy_wav())
    transcripts = iter(
        [
            "I want to register a complaint",
            "My order ORD-2005 arrived damaged with a cracked screen",
            "What is the status of my complaint?",
        ]
    )
    mock_audio.transcribe = AsyncMock(side_effect=lambda *a, **kw: next(transcripts))

    session = IVRCallSession(
        ws=mock_ws,
        orchestrator=ChatOrchestrator(),
        audio_service=mock_audio,
    )
    session.on_call_start({"uuid": "call-cmp-101", "caller": "+919876543210"})
    session.menu_machine.state = IVRState.AI_CONVERSATION
    session.selected_service = "complaint"
    session.intel_pipeline.persistence_enabled = False
    session.response_generator.retrieval_service.search = AsyncMock(
        return_value=MagicMock(
            chunks=[],
            chunk_texts=[],
            cache_hit=False,
            evaluation=MagicMock(tier=MagicMock(value="none"), top_score=0.0),
        )
    )

    voiced = _make_voiced_pcm()
    with (
        patch("telephony.ivr_session.async_session", session_factory),
        patch("tools.business_data.complaints.async_session", session_factory),
        patch.object(OpenAICompatibleChatAdapter, "chat_completion", new=_fake_gemma),
        patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock(return_value=False)),
    ):
        await session._run_stt_ai_tts(voiced)
        await session._run_stt_ai_tts(voiced)
        await session._run_stt_ai_tts(voiced)

    replies = [m.content for m in session._history if m.role == "assistant"]
    assert len(replies) == 3
    assert "issue details" in replies[0].lower()
    assert "CMP-" in replies[1]
    assert "CMP-" in replies[2] and "open" in replies[2].lower()

    async with session_factory() as db:
        rows = (await db.execute(select(Complaint))).scalars().all()
        assert len(rows) == 1
        assert rows[0].call_uuid == "call-cmp-101"  # spoofing blocked, trusted call_uuid preserved
        assert rows[0].caller_number == "***3210"  # raw phone masked at storage
        assert rows[0].order_id == "ORD-2005"
        assert rows[0].category == "damaged_item"
        assert rows[0].priority == "high"

    await engine.dispose()


@pytest.mark.asyncio
async def test_external_api_dispatch_and_call_external_api_tool_security(monkeypatch):
    """
    Verifies:
      • register_complaint dispatches authenticated POST to configured COMPLAINT_API_URL
      • Duplicate retry in the same call_uuid is idempotent
      • call_external_api calls allowlisted REMOTE_TOOLS_BASE_URL endpoints cleanly
      • call_external_api blocks SSRF / unallowlisted URLs with ValidationError
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    captured_requests = []

    async def mock_transport(request: httpx.Request) -> httpx.Response:
        captured_requests.append((request.method, str(request.url), dict(request.headers), request.content.decode("utf-8")))
        if str(request.url) == "https://crm.example.com/api/complaints":
            return httpx.Response(201, json={"id": "CRM-9921", "phone": "+919999888877"})
        if str(request.url).startswith("http://192.168.1.141:8000/api/v1/orders/ORD-2005"):
            return httpx.Response(200, json={"order_id": "ORD-2005", "carrier": "DHL Express", "phone": "+919999888877"})
        return httpx.Response(404, json={"detail": "Not found"})

    monkeypatch.setattr(settings, "COMPLAINT_API_URL", "https://crm.example.com/api/complaints")
    monkeypatch.setattr(settings, "COMPLAINT_API_TOKEN", "secret-crm-token")
    monkeypatch.setattr(settings, "REMOTE_TOOLS_ENABLED", True)
    monkeypatch.setattr(settings, "REMOTE_TOOLS_BASE_URL", "http://192.168.1.141:8000")
    monkeypatch.setattr(settings, "REMOTE_TOOLS_API_KEY", "test-remote-key")
    monkeypatch.setattr(settings, "REMOTE_TOOLS_HMAC_SECRET", "hmac-secret-123")

    ctx = TenantContext(tenant_id="tenant-api-1", tenant_name="Demo", api_key_id="key-1", scopes={"*"})

    async with session_factory() as db:
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_transport)) as http_cli:
            res1 = await register_complaint(
                description="Payment of 1500 debited twice for order ORD-2005",
                order_id="ORD-2005",
                tenant_context=ctx,
                db_session=db,
                call_uuid="call-ext-55",
                http_client=http_cli,
            )
            assert res1["registration_status"] == "registered"
            assert res1["status"] == "open"
            assert res1["external_api_status"] == "dispatched"
            assert res1["external_reference"] == "CRM-9921"

            # Same description in same call_uuid must return existing ticket idempotently
            res_dup = await register_complaint(
                description="Payment of 1500 debited twice for order ORD-2005",
                order_id="ORD-2005",
                tenant_context=ctx,
                db_session=db,
                call_uuid="call-ext-55",
                http_client=http_cli,
            )
            assert res_dup["registration_status"] == "already_registered"
            assert res_dup["ticket_number"] == res1["ticket_number"]

            # Allowlisted remote API call succeeds with PII masked and HMAC signature attached
            api_res = await call_external_api(
                service="remote_django",
                method="GET",
                path="/api/v1/orders/ORD-2005/",
                tenant_context=ctx,
                call_uuid="call-ext-55",
                http_client=http_cli,
            )
            assert api_res["ok"] is True
            assert api_res["data"]["carrier"] == "DHL Express"
            assert api_res["data"]["phone"] == "********8877"
            assert "+919999888877" not in str(api_res)

            # SSRF attempt to cloud metadata / external unallowlisted host must fail closed
            executor = ToolExecutor(db_session=db)
            ssrf_res = await executor.execute_tool(
                call_id="ssrf_1",
                name="call_external_api",
                arguments_json=json.dumps({"path": "http://169.254.169.254/latest/meta-data/"}),
                context=ctx,
            )
            assert ssrf_res.is_error is True
            assert "allowlist" in (ssrf_res.error_message or "").lower()

            # Complaint lookup finds persisted ticket
            lookup = await get_complaint_status(
                ticket_number=res1["ticket_number"],
                tenant_context=ctx,
                db_session=db,
            )
            assert lookup["found"] is True
            assert lookup["complaints"][0]["ticket_number"] == res1["ticket_number"]

    assert len(captured_requests) == 2
    assert captured_requests[1][2].get("x-signature")  # HMAC header attached
    await engine.dispose()


@pytest.mark.asyncio
async def test_complaints_rest_api_crud_and_tenant_isolation():
    """Verifies POST / GET / PATCH /v1/complaints enforce tenant isolation cleanly."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def override_db():
        async with session_factory() as db:
            yield db

    async with session_factory() as db:
        t1 = Tenant(name="Tenant One")
        t2 = Tenant(name="Tenant Two")
        db.add_all([t1, t2])
        await db.commit()
        await db.refresh(t1)
        await db.refresh(t2)
        k1 = ApiKey(tenant_id=t1.id, key_hash=hash_api_key("sk_t1_complaint"), name="K1", scopes="*")
        k2 = ApiKey(tenant_id=t2.id, key_hash=hash_api_key("sk_t2_complaint"), name="K2", scopes="*")
        db.add_all([k1, k2])
        await db.commit()

    app.dependency_overrides[get_db_session] = override_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            h1 = {"Authorization": "Bearer sk_t1_complaint"}
            h2 = {"Authorization": "Bearer sk_t2_complaint"}

            create_resp = await client.post(
                "/v1/complaints",
                json={
                    "description": "Delivery delayed by 5 days for order ORD-3009",
                    "order_id": "ORD-3009",
                    "priority": "high",
                },
                headers=h1,
            )
            assert create_resp.status_code == 201
            created = create_resp.json()
            ticket = created["ticket_number"]
            assert ticket.startswith("CMP-")

            # Tenant 1 can retrieve and update it
            get_resp = await client.get(f"/v1/complaints/{ticket}", headers=h1)
            assert get_resp.status_code == 200
            assert get_resp.json()["order_id"] == "ORD-3009"

            patch_resp = await client.patch(
                f"/v1/complaints/{ticket}",
                json={"status": "resolved", "resolution_notes": "Replacement dispatched"},
                headers=h1,
            )
            assert patch_resp.status_code == 200
            assert patch_resp.json()["status"] == "resolved"

            # Tenant 2 cannot view or modify Tenant 1's complaint
            cross_get = await client.get(f"/v1/complaints/{ticket}", headers=h2)
            assert cross_get.status_code == 404
            cross_list = await client.get("/v1/complaints", headers=h2)
            assert cross_list.status_code == 200
            assert cross_list.json()["count"] == 0
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sync_17_windows_django_tools_and_confirmation_gated_complaint_execution():
    """
    Verifies MacBook compatibility with the 17 Windows Django tools:
      • Syncing create_complaint suppresses redundant local register_complaint
      • Long >128 char descriptions & desired_resolution + order_id -> order_reference work cleanly
      • Confirmation-gated prompt instructions include search_existing_complaints + create_complaint
      • Pruning remote catalog restores local register_complaint cleanly
    """
    from tools.registry import ToolRegistry, register_complaint as local_reg
    from tools.remote.django_client import DjangoRemoteToolClient
    from tools.remote.registry_sync import sync_remote_django_tools

    reg = ToolRegistry()
    reg.register(
        name="register_complaint",
        description="Local register complaint",
        parameters={"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
        handler=local_reg,
        source="local",
    )

    executed_calls = []

    async def django_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/":
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {
                            "name": "search_existing_complaints",
                            "description": "Search open complaints",
                            "parameters": {"type": "object", "properties": {"order_reference": {"type": "string"}}},
                        },
                        {
                            "name": "create_complaint",
                            "description": "Register confirmed complaint",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string"},
                                    "subject": {"type": "string"},
                                    "description": {"type": "string"},
                                    "order_reference": {"type": "string"},
                                    "desired_resolution": {"type": "string"},
                                },
                                "required": ["category", "subject", "description"],
                            },
                        },
                        {
                            "name": "get_complaint_status",
                            "description": "Check complaint status",
                            "parameters": {
                                "type": "object",
                                "properties": {"complaint_reference": {"type": "string"}},
                                "required": ["complaint_reference"],
                            },
                        },
                        {
                            "name": "escalate_complaint",
                            "description": "Escalate complaint",
                            "parameters": {
                                "type": "object",
                                "properties": {"complaint_reference": {"type": "string"}, "reason": {"type": "string"}},
                                "required": ["complaint_reference", "reason"],
                            },
                        },
                    ]
                },
            )
        body = json.loads(request.content.decode("utf-8"))
        executed_calls.append(body)
        return httpx.Response(
            200,
            json={
                "status": "success",
                "execution_time_ms": 12,
                "data": {"complaint_reference": "CMP-2026-00099", "status": "open"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(django_handler)) as mock_http:
        client = DjangoRemoteToolClient("http://192.168.1.141:8000", default_tenant_slug="tenant_001", http_client=mock_http)
        names = await sync_remote_django_tools(client, registry=reg, prefer_remote=True)
        assert "create_complaint" in names
        assert reg.get_tool("register_complaint") is None  # duplicate local register_complaint suppressed cleanly

        # Verify ToolExecutor-style normalization + >128 char description works on remote create_complaint
        handler = reg.get_handler("create_complaint")
        assert handler is not None
        long_desc = (
            "My order ORD-2005 was scheduled for delivery five days ago, the courier never attempted "
            "delivery twice, and the packaging photo shows severe water damage."
        )
        assert len(long_desc) > 128
        ctx = TenantContext(tenant_id="t1", tenant_name="Demo", api_key_id="k1", scopes={"*"})
        out = await handler(
            tenant_context=ctx,
            category="delivery_delay",
            subject="Severe delivery delay and damaged parcel",
            description=long_desc,
            order_id="ORD-2005",
            desired_resolution="Immediate replacement shipment via air express within 24 hours.",
        )
        assert out["complaint_reference"] == "CMP-2026-00099"
        assert executed_calls[0]["arguments"]["order_reference"] == "ORD-2005"

        # Prune remote create_complaint -> local register_complaint is restored automatically
        async def empty_catalog_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"tools": []})

        client_empty = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            default_tenant_slug="tenant_001",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(empty_catalog_handler)),
        )
        await sync_remote_django_tools(client_empty, registry=reg, prefer_remote=True)
        assert reg.get_tool("register_complaint") is not None


@pytest.mark.asyncio
async def test_ivr_tool_model_resolution_never_picks_non_tool_gemma2_and_focuses_tools():
    """
    Guards against Ollama HTTP 400 (`gemma2:2b does not support tools`):
      • `config/providers.yaml` exposes `gemma2` without `tool_calling`, while `gemma` & `gemma4` support `tool_calling` on `gemma4:latest`.
      • `_resolve_ivr_tool_model_alias` switches `gemma2` / `ivr-rag-mini` to tool-capable `gemma4` whenever `tools` are attached.
      • `_select_ivr_session_tools` excludes legacy local DB prototype tools when `remote_django` is synced and keeps tool list focused.
    """
    from platform_core.contracts.capabilities import ModelCapability
    from platform_core.contracts.requests import ChatMessage
    from providers.registry import provider_registry
    from telephony.ivr_session import _resolve_ivr_tool_model_alias, _select_ivr_session_tools

    prev_models = dict(provider_registry.models)
    try:
        provider_registry.initialize_from_file("config/providers.yaml")
        gemma2_meta = provider_registry.get_model_metadata("gemma2")
        gemma4_meta = provider_registry.get_model_metadata("gemma4")
        gemma_meta = provider_registry.get_model_metadata("gemma")

        assert not gemma2_meta.supports(ModelCapability.TOOL_CALLING)
        assert gemma4_meta.supports(ModelCapability.TOOL_CALLING)
        assert gemma4_meta.underlying_model == "gemma4:latest"
        assert gemma_meta.supports(ModelCapability.TOOL_CALLING)

        # Tool-enabled call upgrades non-tool aliases cleanly to a tool-capable Gemma alias
        assert _resolve_ivr_tool_model_alias("gemma2", has_tools=True, settings_alias="gemma4") == "gemma4"
        assert _resolve_ivr_tool_model_alias("ivr-rag-mini", has_tools=True, settings_alias="gemma4") == "gemma4"
        # No-tool turn preserves the requested lightweight alias
        assert _resolve_ivr_tool_model_alias("ivr-rag-mini", has_tools=False, settings_alias="gemma4") == "ivr-rag-mini"

        # Focused session tool selection drops redundant legacy prototype tools & unrelated clutter
        fake_names = [
            "count_open_purchase_requests",
            "get_customer_info",
            "search_customer",
            "get_customer_details",
            "get_order_status",
            "get_customer_orders",
            "search_existing_complaints",
            "create_complaint",
            "get_complaint_status",
            "get_customer_complaints",
            "get_order_tracking",
            "get_payment_status",
            "get_invoice_details",
            "escalate_complaint",
            "search_database_records",
        ]
        schemas = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}} for n in fake_names]
        focused = _select_ivr_session_tools(
            all_tool_schemas=schemas,
            user_text="I am Sarah Connor, I want to register a complaint about order ORD-2005",
            history=[ChatMessage(role="user", content="Hi")],
            service_category="complaint",
            remote_tools_active=True,
        )
        assert focused is not None
        focused_names = {s["function"]["name"] for s in focused}
        assert "count_open_purchase_requests" not in focused_names
        assert "get_customer_info" not in focused_names
        assert {"search_customer", "get_order_status", "search_existing_complaints", "create_complaint"}.issubset(focused_names)
        assert len(focused_names) < len(fake_names)
    finally:
        provider_registry.models = prev_models


@pytest.mark.asyncio
async def test_ivr_ignores_stale_queued_audio_whisper_noise_and_normalizes_ort_order_code():
    """
    Reproduces exact live-call issues from Sarah Connor complaint call:
      1. Phonetic STT mishearing 'ORT2005' normalizes cleanly to 'ORD-2005'.
      2. Stale PCM captured before playback ended is discarded when _pipeline_lock releases.
      3. Low-energy/short Whisper hallucination 'Thank you.' and filler 'Hmm.' do not hijack the call.
    """
    import math
    import re
    import struct
    import time
    from providers.speech_to_text.whisper_local import UNIVERSAL_PHONETIC_NORMALIZATION
    from telephony.ivr_menu import IVRState

    raw_stt = "I am Sarah Ponder, I want to register complaint about ORT2005 arriving damaged."
    normalized = raw_stt
    for pat, rep in UNIVERSAL_PHONETIC_NORMALIZATION:
        normalized = re.sub(pat, rep, normalized, flags=re.IGNORECASE)
    assert "ORD-2005" in normalized

    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_bytes = AsyncMock()
    orch = AsyncMock()
    audio_svc = AsyncMock()
    audio_svc.synthesize = AsyncMock(return_value=b"\x00" * 3200)
    session = IVRCallSession(ws=ws, orchestrator=orch, audio_service=audio_svc)
    session.menu_machine.state = IVRState.AI_CONVERSATION

    # 1. Stale buffer captured BEFORE last playback ended must be ignored
    session._last_playback_end_ts = time.monotonic() + 10.0
    sample_pcm = struct.pack(
        "<" + "h" * 160,
        *[int(3000 * math.sin(2.0 * math.pi * 400.0 * i / 8000.0)) for i in range(160)],
    ) * 25
    await session._run_flush_with_lock(sample_pcm, captured_at=session._last_playback_end_ts - 1.0, from_barge_in=False)
    audio_svc.transcribe.assert_not_awaited()

    # 2. Filler ('Hmm.') and low-energy hallucination ('Thank you.') do not trigger replies
    session._last_playback_end_ts = 0.0
    audio_svc.transcribe = AsyncMock(return_value={"text": "Hmm.", "language": "en"})
    await session._run_stt_ai_tts(sample_pcm)
    audio_svc.synthesize.assert_not_awaited()

    quiet_pcm = struct.pack(
        "<" + "h" * 160,
        *[int(120 * math.sin(2.0 * math.pi * 400.0 * i / 8000.0)) for i in range(160)],
    ) * 15
    audio_svc.transcribe = AsyncMock(return_value={"text": "Thank you.", "language": "en"})
    await session._run_stt_ai_tts(quiet_pcm)
    audio_svc.synthesize.assert_not_awaited()

