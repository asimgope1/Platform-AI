"""
Phase 3 Verification Suite: FastAPI ai-platform <-> Windows Django Remote Business Tool Service.

Covers all 8 test scenarios:
  1. Authenticated remote tool catalog retrieval & schema validation
  2. Remote get_order_status execution (+ retry / duplicate execution prevention + PII masking)
  3. Second remote tool (get_customer_orders) execution
  4. Gemma selecting a remote tool dynamically via /v1/chat/completions ToolLoopRunner
  5. Two tenants with different remote database mappings & strict TenantContext isolation
  6. Unauthorized customer access & SQLi / prompt-injection rejection
  7. Unavailable Windows backend handling (non-fatal startup + bounded retries + error surface)
  8. Complete voice IVR conversation (Whisper -> Gemma -> remote tool -> TTS -> WebSocket PCM)
"""
from __future__ import annotations

import copy
import json
import math
import struct
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.security import sanitize_tool_argument
from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.contracts.requests import (
    ChatMessage,
    FunctionCall,
    NormalizedChatResponse,
    NormalizedChoice,
    ToolCall,
    UsageInfo,
)
from platform_core.errors.exceptions import PermissionDeniedError
from platform_core.security.api_keys import hash_api_key
from platform_core.security.tenant_context import TenantContext
from providers.chat.openai_compatible import OpenAICompatibleChatAdapter
from storage.models.api_key import ApiKey
from storage.models.tenant import Tenant
from telephony.ivr_menu import IVRState
from telephony.ivr_session import IVRCallSession
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry, tool_registry
from tools.remote.django_client import (
    DjangoRemoteToolClient,
    RemoteToolCatalogError,
    RemoteToolUnavailableError,
)
from tools.remote.registry_sync import sync_remote_django_tools

SAMPLE_CATALOG = {
    "count": 10,
    "tools": [
        {
            "name": "search_customer",
            "description": "Search for a customer record by phone number, email, or name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "get_customer_details",
            "description": "Retrieve details of a customer by customer_reference.",
            "parameters": {
                "type": "object",
                "properties": {"customer_reference": {"type": "string"}},
                "required": ["customer_reference"],
            },
        },
        {
            "name": "get_order_status",
            "description": "Fetch current status and summary for a specific order number.",
            "parameters": {
                "type": "object",
                "properties": {"order_number": {"type": "string"}},
                "required": ["order_number"],
            },
        },
        {
            "name": "get_customer_orders",
            "description": "Retrieve recent orders for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_reference": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["customer_reference"],
            },
        },
        {
            "name": "get_order_tracking",
            "description": "Retrieve live delivery tracking information for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_number": {"type": "string"}},
                "required": ["order_number"],
            },
        },
        {
            "name": "get_payment_status",
            "description": "Fetch payment transactions and refund status.",
            "parameters": {
                "type": "object",
                "properties": {"order_number": {"type": "string"}},
                "required": ["order_number"],
            },
        },
        {
            "name": "get_invoice_details",
            "description": "Retrieve invoice billing details for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_number": {"type": "string"}},
                "required": ["order_number"],
            },
        },
        {
            "name": "get_customer_order_summary",
            "description": "Aggregate total orders placed and lifetime spent for a customer.",
            "parameters": {
                "type": "object",
                "properties": {"customer_reference": {"type": "string"}},
                "required": ["customer_reference"],
            },
        },
        {
            "name": "search_database_records",
            "description": "General search across customer records, orders, and products.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
        {
            "name": "search_knowledge_documents",
            "description": "Search company knowledge documents and policy PDFs.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ],
}


@contextmanager
def _preserve_global_tool_registry():
    snap_tools = copy.deepcopy(tool_registry._tools)
    snap_handlers = dict(tool_registry._handlers)
    snap_sources = dict(tool_registry._sources)
    try:
        yield
    finally:
        tool_registry._tools = snap_tools
        tool_registry._handlers = snap_handlers
        tool_registry._sources = snap_sources


async def _create_tenant(db: AsyncSession, name: str, raw_key: str) -> tuple[Tenant, dict[str, str]]:
    tenant = Tenant(name=name, is_active=True)
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    key = ApiKey(tenant_id=tenant.id, key_hash=hash_api_key(raw_key), name=f"{name}-key", scopes="*")
    db.add(key)
    await db.commit()
    return tenant, {"Authorization": f"Bearer {raw_key}"}


@pytest.mark.asyncio
async def test_1_authenticated_remote_catalog_and_sql_guard():
    """Verify authenticated GET /api/v1/tools/ validation, auth enforcement, and SQL tool rejection."""
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        if request.headers.get("authorization") != "Bearer test-django-token":
            return httpx.Response(401, json={"detail": "Unauthorized"})
        return httpx.Response(200, json=SAMPLE_CATALOG)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        unauth_client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            require_auth=True,
            http_client=mock_http,
        )
        with pytest.raises(PermissionDeniedError):
            await unauth_client.fetch_and_validate_catalog()

        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            auth_token="test-django-token",
            api_key="django-api-key-01",
            require_auth=True,
            http_client=mock_http,
        )
        local_reg = ToolRegistry()
        synced = await sync_remote_django_tools(client, registry=local_reg, prefer_remote=True)
        assert len(synced) == 10
        assert "get_order_status" in synced
        assert "get_customer_orders" in synced
        assert seen_headers.get("authorization") == "Bearer test-django-token"
        assert seen_headers.get("x-api-key") == "django-api-key-01"
        assert local_reg.get_tool_source("get_order_status") == "remote_django"

    # Reject any remote tool catalog trying to expose raw SQL execution
    def unsafe_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "count": 1,
                "tools": [
                    {
                        "name": "execute_sql",
                        "description": "Run raw SQL",
                        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}},
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(unsafe_handler)) as unsafe_http:
        client = DjangoRemoteToolClient("http://192.168.1.141:8000", http_client=unsafe_http)
        with pytest.raises(RemoteToolCatalogError):
            await client.fetch_and_validate_catalog()


@pytest.mark.asyncio
async def test_2_and_3_remote_order_status_customer_orders_and_retry_dedup():
    """Verify remote get_order_status & get_customer_orders + transient retry + no duplicate execution."""
    execute_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/":
            return httpx.Response(200, json=SAMPLE_CATALOG)
        if request.url.path == "/api/v1/tools/execute/":
            body = json.loads(request.content.decode())
            execute_calls.append(
                {
                    "tool": body["tool_name"],
                    "tenant_id": body["tenant_id"],
                    "args": body["arguments"],
                    "idem": request.headers.get("idempotency-key"),
                }
            )
            # Simulate 1 transient 503 on first call, then succeed
            if body["tool_name"] == "get_order_status" and len(execute_calls) == 1:
                return httpx.Response(503, json={"detail": "Temporary overload"})
            if body["tool_name"] == "get_order_status":
                return httpx.Response(
                    200,
                    json={
                        "tool_name": "get_order_status",
                        "status": "success",
                        "execution_time_ms": 2,
                        "data": {
                            "order_number": body["arguments"]["order_number"],
                            "customer_name": "John Doe",
                            "phone": "+919876543210",
                            "email": "john@example.com",
                            "status": "shipped",
                            "total_amount": 149.99,
                            "currency": "USD",
                        },
                    },
                )
            if body["tool_name"] == "get_customer_orders":
                return httpx.Response(
                    200,
                    json={
                        "tool_name": "get_customer_orders",
                        "status": "success",
                        "execution_time_ms": 2,
                        "data": {
                            "customer_reference": body["arguments"]["customer_reference"],
                            "orders_count": 1,
                            "orders": [
                                {
                                    "order_number": "ORD-2005",
                                    "status": "processing",
                                    "total_amount": 499.0,
                                }
                            ],
                        },
                    },
                )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            auth_token="tok-123",
            max_retries=2,
            tenant_map={"fast-t1": "tenant_001"},
            http_client=mock_http,
        )
        with _preserve_global_tool_registry():
            await sync_remote_django_tools(client, registry=tool_registry, prefer_remote=True)
            executor = ToolExecutor(timeout_seconds=5.0)
            ctx = TenantContext(tenant_id="fast-t1", tenant_name="Fast T1", api_key_id="k1", scopes={"*"})

            # Test 2: get_order_status (retries once on 503 with SAME Idempotency-Key + PII masked)
            res_order = await executor.execute_tool(
                call_id="call_ord_1",
                name="get_order_status",
                arguments_json='{"order_number": "ORD-1001"}',
                context=ctx,
            )
            assert not res_order.is_error, res_order.error_message
            assert res_order.output["status"] == "shipped"
            assert res_order.output["order_number"] == "ORD-1001"
            assert res_order.output["phone"] == "********3210"
            assert res_order.output["email"].startswith("j***@")
            assert len(execute_calls) == 2
            assert execute_calls[0]["idem"] == execute_calls[1]["idem"]
            assert execute_calls[1]["tenant_id"] == "tenant_001"

            # Repeating the same call_id must NOT re-execute on Django (dedup cache hit)
            res_repeat = await executor.execute_tool(
                call_id="call_ord_1",
                name="get_order_status",
                arguments_json='{"order_number": "ORD-1001"}',
                context=ctx,
            )
            assert not res_repeat.is_error
            assert len(execute_calls) == 2  # No extra POST sent!

            # Test 3: get_customer_orders
            res_orders = await executor.execute_tool(
                call_id="call_cust_orders_1",
                name="get_customer_orders",
                arguments_json='{"customer_reference": "CUST_456", "limit": 5}',
                context=ctx,
            )
            assert not res_orders.is_error, res_orders.error_message
            assert res_orders.output["orders_count"] == 1
            assert res_orders.output["orders"][0]["order_number"] == "ORD-2005"


@pytest.mark.asyncio
async def test_4_and_5_gemma_dynamic_selection_and_two_tenants(
    async_client: httpx.AsyncClient,
    db_session: AsyncSession,
):
    """Verify Gemma dynamically invokes remote Django tools across 2 tenants via /v1/chat/completions."""
    tenant_a, headers_a = await _create_tenant(db_session, "Remote Tenant Alpha", "sk-rem-alpha")
    tenant_b, headers_b = await _create_tenant(db_session, "Remote Tenant Beta", "sk-rem-beta")

    tenant_dbs = {
        "tenant_alpha_db": {"ORD-1001": "Dispatched via FedEx (Test Data)"},
        "tenant_beta_db": {"ORD-1001": "Held at Customs - Chennai (Test Data)"},
    }
    captured_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/":
            return httpx.Response(200, json=SAMPLE_CATALOG)
        if request.url.path == "/api/v1/tools/execute/":
            body = json.loads(request.content.decode())
            captured_payloads.append(body)
            tid = body["tenant_id"]
            ord_no = body["arguments"]["order_number"]
            status_val = tenant_dbs.get(tid, {}).get(ord_no, "not_found")
            return httpx.Response(
                200,
                json={
                    "tool_name": "get_order_status",
                    "status": "success",
                    "execution_time_ms": 1,
                    "data": {"order_number": ord_no, "status": status_val},
                },
            )
        return httpx.Response(404)

    async def _fake_gemma(self, request, model_meta):
        last = request.messages[-1]
        if last.role == "tool":
            payload = json.loads(str(last.content))
            return NormalizedChatResponse(
                id="gemma-final",
                model=model_meta.alias,
                created=1,
                choices=[
                    NormalizedChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=f"Order {payload['order_number']} is currently: {payload['status']}.",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=15, completion_tokens=15, total_tokens=30),
            )
        # Even if an adversarial argument tried passing tenant_id, ToolExecutor strips it
        return NormalizedChatResponse(
            id="gemma-tc",
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
                                id=f"tc_{len(captured_payloads) + 1}",
                                function=FunctionCall(
                                    name="get_order_status",
                                    arguments='{"order_number": "ORD-1001", "tenant_id": "SPOOFED"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageInfo(prompt_tokens=15, completion_tokens=15, total_tokens=30),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            tenant_map={tenant_a.id: "tenant_alpha_db", tenant_b.id: "tenant_beta_db"},
            http_client=mock_http,
        )
        with _preserve_global_tool_registry():
            await sync_remote_django_tools(client, registry=tool_registry, prefer_remote=True)
            with patch.object(OpenAICompatibleChatAdapter, "chat_completion", new=_fake_gemma):
                chat_req = {
                    "model": "gemma4",
                    "messages": [{"role": "user", "content": "What is the status of order ORD-1001?"}],
                    "enable_tools": True,
                }
                resp_a = await async_client.post("/v1/chat/completions", json=chat_req, headers=headers_a)
                resp_b = await async_client.post("/v1/chat/completions", json=chat_req, headers=headers_b)

    assert resp_a.status_code == 200, resp_a.text
    assert resp_b.status_code == 200, resp_b.text
    assert "Dispatched via FedEx" in resp_a.json()["choices"][0]["message"]["content"]
    assert "Held at Customs - Chennai" in resp_b.json()["choices"][0]["message"]["content"]
    assert [p["tenant_id"] for p in captured_payloads] == ["tenant_alpha_db", "tenant_beta_db"]


@pytest.mark.asyncio
async def test_6_unauthorized_customer_and_injection_blocked():
    """Verify caller cannot query another customer's reference/records or inject SQL."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/execute/":
            body = json.loads(request.content.decode())
            if body["tool_name"] == "get_order_status":
                # Simulated order owned by a DIFFERENT customer CUST_999
                return httpx.Response(
                    200,
                    json={
                        "tool_name": "get_order_status",
                        "status": "success",
                        "data": {
                            "order_number": "ORD-9999",
                            "customer_reference": "CUST_999",
                            "status": "delivered",
                        },
                    },
                )
        return httpx.Response(200, json=SAMPLE_CATALOG)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            default_tenant_slug="tenant_001",
            http_client=mock_http,
        )
        with _preserve_global_tool_registry():
            await sync_remote_django_tools(client, registry=tool_registry, prefer_remote=True)
            executor = ToolExecutor(timeout_seconds=5.0)
            ctx = TenantContext(tenant_id="tenant_001", tenant_name="Tenant 001", api_key_id="k1", scopes={"*"})

            # Cross-customer get_customer_orders blocked BEFORE calling Django
            res1 = await executor.execute_tool(
                call_id="tc_unauth_1",
                name="get_customer_orders",
                arguments_json='{"customer_reference": "CUST_999"}',
                context=ctx,
                trusted_metadata={"authorized_customer_id": "CUST_456"},
            )
            assert res1.is_error is True
            assert "Unauthorized" in (res1.error_message or "")

            # Cross-customer order response blocked AFTER lookup
            res2 = await executor.execute_tool(
                call_id="tc_unauth_2",
                name="get_order_status",
                arguments_json='{"order_number": "ORD-9999"}',
                context=ctx,
                trusted_metadata={"authorized_customer_id": "CUST_456"},
            )
            assert res2.is_error is True
            assert "authorized" in (res2.error_message or "").lower()

            # SQLi attempt rejected in argument sanitization
            res3 = await executor.execute_tool(
                call_id="tc_sqli",
                name="get_order_status",
                arguments_json='{"order_number": "ORD-1001; DROP TABLE orders--"}',
                context=ctx,
            )
            assert res3.is_error is True
            assert "unsafe" in (res3.error_message or "").lower()


@pytest.mark.asyncio
async def test_7_unavailable_windows_backend_graceful_handling():
    """Verify offline Windows backend fails cleanly without infinite retries or hangs."""
    attempts = 0

    def offline_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("Connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(offline_handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            timeout_seconds=1.0,
            max_retries=1,
            http_client=mock_http,
        )
        with pytest.raises(RemoteToolUnavailableError):
            await client.fetch_and_validate_catalog()
        assert attempts == 2  # initial try + 1 bounded retry

        attempts = 0
        with pytest.raises(RemoteToolUnavailableError):
            await client.execute_tool(
                "get_order_status",
                {"order_number": "ORD-1001"},
                context=TenantContext(tenant_id="tenant_001", tenant_name="Tenant 001", api_key_id="k1", scopes={"*"}),
            )
        assert attempts == 2


@pytest.mark.asyncio
async def test_8_complete_voice_conversation_using_remote_tool():
    """Verify full Voice IVR turn: caller question -> Gemma selects remote Django tool -> spoken TTS PCM."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/":
            return httpx.Response(200, json=SAMPLE_CATALOG)
        if request.url.path == "/api/v1/tools/execute/":
            body = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={
                    "tool_name": body["tool_name"],
                    "status": "success",
                    "execution_time_ms": 1,
                    "data": {
                        "order_number": "ORD-2005",
                        "customer_name": "Sarah Connor",
                        "status": "processing",
                        "total_amount": 499.0,
                        "currency": "USD",
                    },
                },
            )
        return httpx.Response(404)

    async def _voice_gemma(self, request, model_meta):
        last = request.messages[-1]
        if last.role == "tool":
            payload = json.loads(str(last.content))
            return NormalizedChatResponse(
                id="gemma-voice-final",
                model=model_meta.alias,
                created=1,
                choices=[
                    NormalizedChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=f"Your order {payload['order_number']} is currently processing.",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=10, completion_tokens=12, total_tokens=22),
            )
        assert request.tools and any(t["function"]["name"] == "get_order_status" for t in request.tools)
        return NormalizedChatResponse(
            id="gemma-voice-tc",
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
                                id="voice_tc_1",
                                function=FunctionCall(
                                    name="get_order_status",
                                    arguments='{"order_number": "ORD-2005"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    from telephony.ivr_session import _raw_pcm_to_wav
    mock_ws = AsyncMock()
    mock_audio = AsyncMock()
    mock_audio.transcribe = AsyncMock(return_value="Where is my order ORD-2005?")
    # Valid 8kHz mono WAV bytes for _tts_output_to_pcm8k
    mock_audio.synthesize = AsyncMock(return_value=_raw_pcm_to_wav(b"\x01\x00" * 1600, sample_rate=8000))

    orchestrator = ChatOrchestrator()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            default_tenant_slug="tenant_001",
            http_client=mock_http,
        )
        with _preserve_global_tool_registry():
            await sync_remote_django_tools(client, registry=tool_registry, prefer_remote=True)
            session = IVRCallSession(ws=mock_ws, orchestrator=orchestrator, audio_service=mock_audio)
            session.menu_machine.state = IVRState.AI_CONVERSATION
            session.selected_language = "en"
            session.selected_service = "customer_support"
            session.response_generator.retrieval_service.search = AsyncMock(
                return_value=MagicMock(
                    chunks=[],
                    chunk_texts=[],
                    cache_hit=False,
                    evaluation=MagicMock(tier=MagicMock(value="none"), top_score=0.0),
                )
            )

            # Synthesize 400ms voiced PCM @ 8kHz so audio quality gate passes cleanly
            voiced_samples = [
                int(6000 * math.sin(2.0 * math.pi * 300.0 * i / 8000.0)) for i in range(3200)
            ]
            voiced_pcm = struct.pack("<" + "h" * len(voiced_samples), *voiced_samples)

            with patch.object(OpenAICompatibleChatAdapter, "chat_completion", new=_voice_gemma):
                with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock()):
                    await session._run_stt_ai_tts(voiced_pcm)

    mock_audio.synthesize.assert_awaited()
    spoken_text = mock_audio.synthesize.call_args.kwargs.get("text") or mock_audio.synthesize.call_args.args[0]
    assert "ORD-2005" in spoken_text
    assert "processing" in spoken_text.lower()
    assert mock_ws.send_bytes.await_count >= 1
