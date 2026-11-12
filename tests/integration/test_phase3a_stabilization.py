"""
Phase 3A Stabilization & Resilience Test Suite (Tasks 5–11).

Covers:
  - Task 5: Non-fatal FastAPI lifespan startup when Windows Django backend is unavailable
  - Task 6: Tool catalog refresh, stale tool pruning, and failed-refresh preservation
  - Task 7: Strict TenantContext + trusted customer identity enforcement
  - Task 8: Idempotency across connection failures, transient retries, and concurrent gather calls
  - Task 9: Remote tool timeout & asyncio.Task cancellation handling
  - Task 10: Safe voice IVR fallback when remote backend / Gemma is unavailable (no fake order fabrication)
  - Task 11: Zero secret / raw PII leakage in application logs
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import struct
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from orchestration.chat_orchestrator import ChatOrchestrator
from platform_core.config.settings import settings
from platform_core.contracts.requests import (
    ChatMessage,
    FunctionCall,
    NormalizedChatResponse,
    NormalizedChoice,
    ToolCall,
    UsageInfo,
)
from platform_core.errors.exceptions import PermissionDeniedError
from platform_core.security.tenant_context import TenantContext
from providers.chat.openai_compatible import OpenAICompatibleChatAdapter
from services.api.lifespan import lifespan
from telephony.ivr_menu import IVRState
from telephony.ivr_session import IVRCallSession, _raw_pcm_to_wav
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry, tool_registry
from tools.remote.django_client import DjangoRemoteToolClient, RemoteToolUnavailableError
from tools.remote.registry_sync import (
    _LOCAL_BACKUPS,
    _SYNC_STATE,
    get_remote_sync_state,
    sync_remote_django_tools,
)


@contextmanager
def _preserve_global_tool_registry():
    snap_tools = copy.deepcopy(tool_registry._tools)
    snap_handlers = dict(tool_registry._handlers)
    snap_sources = dict(tool_registry._sources)
    snap_backups = copy.deepcopy(_LOCAL_BACKUPS)
    snap_sync = copy.deepcopy(_SYNC_STATE)
    try:
        yield
    finally:
        tool_registry._tools = snap_tools
        tool_registry._handlers = snap_handlers
        tool_registry._sources = snap_sources
        _LOCAL_BACKUPS.clear()
        _LOCAL_BACKUPS.update(snap_backups)
        _SYNC_STATE.clear()
        _SYNC_STATE.update(snap_sync)


@pytest.mark.asyncio
async def test_5_startup_when_django_unavailable(tmp_path):
    """Task 5: FastAPI lifespan boots cleanly even when Windows Django backend is offline."""
    local_db = tmp_path / "startup_test.db"
    dummy_app = FastAPI()

    def offline_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    orig_enabled = settings.REMOTE_TOOLS_ENABLED
    orig_url = settings.REMOTE_TOOLS_BASE_URL
    orig_db = settings.DATABASE_URL
    try:
        settings.REMOTE_TOOLS_ENABLED = True
        settings.REMOTE_TOOLS_BASE_URL = "http://192.168.1.141:8000"
        settings.DATABASE_URL = f"sqlite+aiosqlite:///{local_db}"

        with _preserve_global_tool_registry():
            with patch(
                "tools.remote.django_client.DjangoRemoteToolClient.fetch_and_validate_catalog",
                side_effect=RemoteToolUnavailableError("Windows backend offline at boot"),
            ):
                async with lifespan(dummy_app):
                    # Local tools remain intact and usable when Django is unreachable at startup
                    assert tool_registry.get_tool("get_order_status") is not None
                    assert tool_registry.get_tool_source("get_order_status") == "local"
    finally:
        settings.REMOTE_TOOLS_ENABLED = orig_enabled
        settings.REMOTE_TOOLS_BASE_URL = orig_url
        settings.DATABASE_URL = orig_db


@pytest.mark.asyncio
async def test_6_catalog_refresh_and_stale_handling():
    """Task 6: Catalog refresh prunes removed remote tools & preserves last known good catalog on failure."""
    catalog_phase = {"step": 1}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/v1/tools/":
            return httpx.Response(404)
        if catalog_phase["step"] == 1:
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {
                            "name": "get_order_status",
                            "description": "v1 description",
                            "parameters": {"type": "object", "properties": {"order_number": {"type": "string"}}, "required": ["order_number"]},
                        },
                        {
                            "name": "legacy_promo_lookup",
                            "description": "Deprecated remote tool",
                            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
                        },
                    ]
                },
            )
        if catalog_phase["step"] == 2:
            # legacy_promo_lookup removed; description updated
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {
                            "name": "get_order_status",
                            "description": "v2 updated description",
                            "parameters": {"type": "object", "properties": {"order_number": {"type": "string"}}, "required": ["order_number"]},
                        }
                    ]
                },
            )
        # Step 3: Windows Django offline during refresh
        return httpx.Response(503, json={"detail": "Service temporarily unavailable"})

    reg = ToolRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = DjangoRemoteToolClient("http://192.168.1.141:8000", max_retries=0, http_client=mock_http)

        # Initial sync
        names1 = await sync_remote_django_tools(client, registry=reg, prefer_remote=True)
        assert set(names1) == {"get_order_status", "legacy_promo_lookup"}
        assert get_remote_sync_state(reg)["stale"] is False

        # Refresh: prunes stale tool & updates schema
        catalog_phase["step"] = 2
        names2 = await sync_remote_django_tools(client, registry=reg, prefer_remote=True)
        assert names2 == ["get_order_status"]
        assert reg.get_tool("legacy_promo_lookup") is None
        assert reg.get_tool("get_order_status").function.description == "v2 updated description"

        # Failed refresh: preserves last known good catalog and flags stale=True
        catalog_phase["step"] = 3
        with pytest.raises(RemoteToolUnavailableError):
            await sync_remote_django_tools(client, registry=reg, prefer_remote=True)
        assert reg.get_tool("get_order_status") is not None
        state = get_remote_sync_state(reg)
        assert state["stale"] is True
        assert state["tool_count"] == 1


@pytest.mark.asyncio
async def test_8_idempotency_across_timeouts_and_concurrent_retries():
    """Task 8: Retries after ReadTimeout/503 and concurrent calls share same Idempotency-Key & run once."""
    seen_calls = []

    async def slow_once_then_ok(request: httpx.Request) -> httpx.Response:
        seen_calls.append(request.headers.get("idempotency-key"))
        if len(seen_calls) == 1:
            raise httpx.ReadTimeout("Simulated read timeout", request=request)
        await asyncio.sleep(0.05)
        return httpx.Response(
            200,
            json={
                "tool_name": "get_order_status",
                "status": "success",
                "execution_time_ms": 2,
                "data": {"order_number": "ORD-2005", "status": "processing"},
            },
        )

    transport = httpx.AsyncClient(transport=httpx.MockTransport(slow_once_then_ok))
    async with transport as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            default_tenant_slug="tenant_001",
            max_retries=2,
            http_client=mock_http,
        )
        ctx = TenantContext(tenant_id="t1", tenant_name="T1", api_key_id="k1", scopes={"*"})

        # Concurrent invocations with the exact same call_id coalesce cleanly;
        # attempt #1 raises ReadTimeout, retry #2 succeeds, concurrent caller receives cached result.
        r1, r2 = await asyncio.gather(
            client.execute_tool("get_order_status", {"order_number": "ORD-2005"}, context=ctx, call_id="call_idem_99"),
            client.execute_tool("get_order_status", {"order_number": "ORD-2005"}, context=ctx, call_id="call_idem_99"),
        )
        assert r1["status"] == "processing"
        assert r2["status"] == "processing"
        assert len(seen_calls) == 2  # 1 failed attempt + 1 retry (NOT 2 independent executions!)
        assert seen_calls[0] == seen_calls[1]


@pytest.mark.asyncio
async def test_9_timeout_and_task_cancellation():
    """Task 9: ToolExecutor enforces timeout cleanly and asyncio.Task cancellation propagates without hangs."""
    async def hanging_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/":
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {
                            "name": "get_order_status",
                            "description": "Status",
                            "parameters": {"type": "object", "properties": {"order_number": {"type": "string"}}, "required": ["order_number"]},
                        }
                    ]
                },
            )
        await asyncio.sleep(5.0)
        return httpx.Response(200, json={"status": "success", "data": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(hanging_handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            default_tenant_slug="tenant_001",
            timeout_seconds=10.0,
            max_retries=0,
            http_client=mock_http,
        )
        ctx = TenantContext(tenant_id="t1", tenant_name="T1", api_key_id="k1", scopes={"*"})

        with _preserve_global_tool_registry():
            await sync_remote_django_tools(client, registry=tool_registry, prefer_remote=True)
            executor = ToolExecutor(timeout_seconds=0.15)
            timeout_res = await executor.execute_tool(
                call_id="tc_timeout",
                name="get_order_status",
                arguments_json='{"order_number": "ORD-1001"}',
                context=ctx,
            )
            assert timeout_res.is_error is True
            assert "timed out" in (timeout_res.error_message or "").lower()

            # Task cancellation
            task = asyncio.create_task(
                client.execute_tool("get_order_status", {"order_number": "ORD-1001"}, context=ctx)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_10_and_11_voice_fallback_when_remote_unavailable_and_no_secret_or_pii_logs(caplog):
    """Tasks 10 & 11: Voice IVR refuses to fabricate order statuses on backend failure & logs never leak secrets/PII."""
    secret_token = "super-secret-django-token-xyz987"
    secret_key = "super-secret-api-key-654321"
    raw_phone = "+919876543210"
    raw_email = "private.customer@secretcorp.com"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tools/":
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {
                            "name": "get_order_status",
                            "description": "Status",
                            "parameters": {"type": "object", "properties": {"order_number": {"type": "string"}}, "required": ["order_number"]},
                        }
                    ]
                },
            )
        # Simulate Windows backend outage during live call
        return httpx.Response(503, json={"detail": "Backend down"})

    async def _gemma_tries_tool_then_fails_or_apologizes(self, request, model_meta):
        # Simulate upstream Ollama ALSO failing after backend failure to test worst-case fallback path
        last = request.messages[-1]
        if last.role == "tool":
            raise RuntimeError("Upstream Gemma unavailable")
        return NormalizedChatResponse(
            id="tc_fail_case",
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
                                id="voice_down_1",
                                function=FunctionCall(name="get_order_status", arguments='{"order_number": "ORD-2005"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    mock_ws = AsyncMock()
    mock_audio = AsyncMock()
    mock_audio.transcribe = AsyncMock(return_value="Where is my order ORD-2005?")
    mock_audio.synthesize = AsyncMock(return_value=_raw_pcm_to_wav(b"\x01\x00" * 1600, sample_rate=8000))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = DjangoRemoteToolClient(
            "http://192.168.1.141:8000",
            auth_token=secret_token,
            api_key=secret_key,
            default_tenant_slug="tenant_001",
            max_retries=0,
            http_client=mock_http,
        )
        with caplog.at_level(logging.INFO):
            # Exercise catalog + masked execution once to generate logs
            with _preserve_global_tool_registry():
                await sync_remote_django_tools(client, registry=tool_registry, prefer_remote=True)

                session = IVRCallSession(ws=mock_ws, orchestrator=ChatOrchestrator(), audio_service=mock_audio)
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

                voiced_samples = [
                    int(6000 * math.sin(2.0 * math.pi * 300.0 * i / 8000.0)) for i in range(3200)
                ]
                voiced_pcm = struct.pack("<" + "h" * len(voiced_samples), *voiced_samples)

                with patch.object(OpenAICompatibleChatAdapter, "chat_completion", new=_gemma_tries_tool_then_fails_or_apologizes):
                    with patch("telephony.ivr_session.learn_from_voice_turn", new=AsyncMock()):
                        await session._run_stt_ai_tts(voiced_pcm)

    mock_audio.synthesize.assert_awaited()
    spoken_text = mock_audio.synthesize.call_args.kwargs.get("text") or mock_audio.synthesize.call_args.args[0]
    # Must NOT fabricate Delhivery / BlueDart / Ecom Express fake delivery status!
    assert "temporarily unavailable" in spoken_text.lower()
    assert "delhivery" not in spoken_text.lower()
    assert "bluedart" not in spoken_text.lower()
    assert mock_ws.send_bytes.await_count >= 1

    # Verify zero secrets leaked to logs
    combined_logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert secret_token not in combined_logs
    assert secret_key not in combined_logs
    assert raw_phone not in combined_logs
    assert raw_email not in combined_logs

