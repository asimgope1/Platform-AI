# Phase 3 Report: Connect Mac FastAPI `ai-platform` to Windows Django Business Tool Service

## 1. Architecture Verified

- **MacBook — FastAPI `ai-platform`**: Remains the **single primary AI orchestrator** (`ChatOrchestrator`, `ToolLoopRunner`, `ToolExecutor`, `ToolRegistry`, Hybrid RAG/pgvector, Whisper STT, Edge/Odia TTS, and `IVRCallSession`).
- **Windows (`192.168.1.141:8000`) — Django Backend**: Acts strictly as the **remote business tool execution service** via `GET /api/v1/tools/` and `POST /api/v1/tools/execute/` (no second LLM orchestrator or duplicate RAG created).
- **Windows (`192.168.1.141:11434/v1`) — Ollama**: Provides Gemma inference (`gemma4:latest` with native `"tools"` capability and `reasoning_effort: none` for responsive tool selection; `gemma2:2b` preserved under `gemma`).

---

## 2. Modified & Created Files

### Created
- `tools/remote/__init__.py`
- `tools/remote/django_client.py` (`DjangoRemoteToolClient`, `validate_remote_tool_spec`, `RemoteToolCatalogError`, `RemoteToolUnavailableError`, `RemoteToolExecutionError`)
- `tools/remote/registry_sync.py` (`sync_remote_django_tools` binding validated Django catalog tools into `ToolRegistry` while preserving Phase 1 per-tenant `TenantDatabaseConfig` routing when present and `prefer_remote=False`)
- `tests/integration/test_remote_django_tools.py` (8-scenario integration test suite)

### Modified
- `platform_core/config/settings.py`: added `GEMMA_MODEL_ALIAS` and `REMOTE_TOOLS_*` configuration settings (`REMOTE_TOOLS_ENABLED`, `REMOTE_TOOLS_BASE_URL`, `REMOTE_TOOLS_AUTH_TOKEN`, `REMOTE_TOOLS_API_KEY`, `REMOTE_TOOLS_REQUIRE_AUTH`, `REMOTE_TOOLS_TIMEOUT_SECONDS`, `REMOTE_TOOLS_MAX_RETRIES`, `REMOTE_TOOLS_DEFAULT_TENANT_SLUG`, `REMOTE_TOOLS_TENANT_MAP`, `REMOTE_TOOLS_PREFER_REMOTE`)
- `platform_core/config/validation.py`: added optional `reasoning_effort` on `ProviderConfig`
- `tools/registry.py`: added `source` tracking (`local` vs `remote_django`), `get_tool_source()`, `list_tools_with_sources()`, and `unregister()`
- `tools/executor.py`: normalized `order_id` <-> `order_number` alias across local/remote schemas and injected trusted `call_id` into supporting handlers alongside `tenant_context`, `db_session`, and `authorized_customer_id`
- `orchestration/tool_loop.py`: added per-turn deduplication cache (`turn_exec_cache`) keyed by `(tool_name, canonical_args)` to prevent duplicate execution on LLM/turn retries
- `providers/chat/openai_compatible.py`: added configurable `reasoning_effort`, raised connect timeout to `min(3.0, self.timeout)`, and logged exception type names on network errors
- `config/providers.yaml`: added `gemma4` (`gemma4:latest`, `tool_calling`) and `reasoning_effort: none` + `timeout_seconds: 60.0` on `windows_ollama`
- `services/api/lifespan.py`: non-fatal startup sync calling `sync_remote_django_tools()` when `REMOTE_TOOLS_ENABLED=true` and `REMOTE_TOOLS_BASE_URL` is set
- `services/conversation/response_generator.py`: optional `context`, `tools`, and `authorized_customer_id` forwarding to `ChatOrchestrator.execute`
- `telephony/ivr_session.py`: routes conversational turns with synced `remote_django` tools through `RAGFirstResponseGenerator` + `ChatOrchestrator` tool loop while preserving local `ivr_mini_brain` behavior when remote tools are not synced
- `.env`: configured `REMOTE_TOOLS_ENABLED=true`, `REMOTE_TOOLS_BASE_URL=http://192.168.1.141:8000`, `REMOTE_TOOLS_TIMEOUT_SECONDS=5.0`, `REMOTE_TOOLS_MAX_RETRIES=2`, `REMOTE_TOOLS_DEFAULT_TENANT_SLUG=tenant_001`, `REMOTE_TOOLS_TENANT_MAP={"879e9bda-8049-4824-9aa0-fc5e9f0012d9":"tenant_001"}`

---

## 3. Security & Reliability Controls

- **Catalog Validation & SQL Block**: `validate_remote_tool_spec()` rejects malformed schemas, non-identifier tool names, SQL-executing tool names (`execute_sql`, `raw_sql`, anything containing `sql`), and SQL parameters (`sql`, `raw_sql`, `ddl`, `statement`).
- **Tenant & Customer Authorization**:
  - `resolve_remote_tenant_id()` derives target Django `tenant_id` exclusively from trusted `TenantContext` (`REMOTE_TOOLS_TENANT_MAP` / `REMOTE_TOOLS_DEFAULT_TENANT_SLUG`), while `ToolExecutor._RESERVED_INTERNAL_ARGS` strips any model-supplied `tenant_id`.
  - When `authorized_customer_id` is set in `trusted_metadata`, cross-customer `customer_reference`/`customer_id` queries are rejected with `PermissionDeniedError` both pre-request and post-response, and returned records pass through `mask_record_pii()`.
- **Authentication, Timeouts, Bounded Retries & Duplicate Prevention**:
  - Supports `Authorization: Bearer <token>`, `X-API-Key`, `X-Tenant-ID`, `X-Customer-Reference`, `Idempotency-Key`, and strict `require_auth` rejection when unauthenticated.
  - Retries only transient transport/502/503/504 errors up to `REMOTE_TOOLS_MAX_RETRIES`, preserves the same `Idempotency-Key`, deduplicates repeated calls by `call_id`/canonical args in `DjangoRemoteToolClient` and within `ToolLoopRunner.run()`, and never retries `4xx` errors.

---

## 4. Actual Test Results & Data Provenance

### A. Live Cross-Machine LAN Checks (`MacBook -> Windows 192.168.1.141`)
> **Data Source Provenance:** All records returned by Windows `192.168.1.141:8000` below (`tenant_001` = `"Acme Global Solutions"`, `ORD-1001` / John Doe, `ORD-2005` / Sarah Connor / `CUST_456` / tracking `DHL8821901122`) are **seeded test fixture data** on the Windows Django backend, **not** a production customer database.

1. **Connectivity & Catalog**:
   - `nc -vz -w 3 192.168.1.141 8000` -> `Connection to 192.168.1.141 port 8000 [tcp/irdmi] succeeded!`
   - `GET http://192.168.1.141:8000/api/v1/tools/` -> `HTTP/1.1 200 OK`, `count: 10` tools discovered and validated (`search_customer`, `get_customer_details`, `get_order_status`, `get_customer_orders`, `get_order_tracking`, `get_payment_status`, `get_invoice_details`, `get_customer_order_summary`, `search_database_records`, `search_knowledge_documents`).
   - `GET http://192.168.1.141:8000/api/v1/tenants/tenants/` -> `HTTP/1.1 200 OK` (`slug: "tenant_001"`, `"Acme Global Solutions"`).
   - `GET http://192.168.1.141:11434/api/tags` -> `HTTP/1.1 200 OK` (`gemma2:2b` without `tools`; `gemma4:latest` with `["completion", "vision", "audio", "tools", "thinking"]`).
2. **Live Remote Tool Execution via FastAPI `ToolExecutor`**:
   - `get_order_status(order_number="ORD-1001")` -> `is_error=False`, `status="shipped"`, `customer_name="John Doe"`, `total_amount=149.99` (`source="remote_django"`, `execution_time_ms=2`) — **Seeded Windows Test Data**.
   - `get_customer_orders(customer_reference="CUST_456")` -> `is_error=False`, `orders_count=1`, `ORD-2005` (`status="processing"`, `total_amount=499.0`) (`source="remote_django"`, `execution_time_ms=2–3`) — **Seeded Windows Test Data**.
   - Nonexistent tenant `nonexistent_tenant_999` -> `HTTP/1.1 404 Not Found` (`"No Tenant matches the given query."`).
3. **Live Real Gemma 4 + Remote Django End-to-End Roundtrip**:
   - `ChatOrchestrator.execute()` with `model_alias="gemma4"` (`windows_ollama` -> `192.168.1.141:11434/v1` with `reasoning_effort="none"`) + 10 synced Django tools for `"Where is my order ORD-2005? Check the live order status using the available tool."`:
   - `LIVE_GEMMA4_TOOL_LOOP_MODEL: gemma4` (no fallback): `gemma4` dynamically invoked remote tool `get_order_tracking` against `192.168.1.141:8000/api/v1/tools/execute/` and responded:
     `"The live tracking for your order ORD-2005 shows the following: Current Status: Fulfillment center preparing dispatch, Estimated Delivery: Friday by 7:00 PM, Carrier: DHL Express, Tracking Number: DHL8821901122, Last Location: Warehouse Logistics Hub 3, Chicago, IL"` — **Seeded Windows Test Data**.

### B. Automated Pytest Suite (Post-Final-Edit Run)
> **Data Source Provenance:** Tests in `tests/integration/test_remote_django_tools.py` and `tests/integration/test_database_agnostic_architecture.py` use **mocked HTTP transports (`httpx.MockTransport`) / mocked LLM tool calls and temporary SQLite fixtures** (no production customer database touched).

Command:
`cd ai-platform && .venv/bin/pytest tests/integration/test_remote_django_tools.py tests/integration/test_database_agnostic_architecture.py tests/unit/test_tools.py tests/telephony/test_ivr_flow.py tests/telephony/test_hybrid_intelligence_e2e.py -v`

Result: **25 passed in 10.19s** (`0 failed`, `0 skipped`):
- `test_1_authenticated_remote_catalog_and_sql_guard` — PASSED
- `test_2_and_3_remote_order_status_customer_orders_and_retry_dedup` — PASSED
- `test_4_and_5_gemma_dynamic_selection_and_two_tenants` — PASSED
- `test_6_unauthorized_customer_and_injection_blocked` — PASSED
- `test_7_unavailable_windows_backend_graceful_handling` — PASSED
- `test_8_complete_voice_conversation_using_remote_tool` — PASSED
- All 4 Phase 1 tests (`test_database_agnostic_architecture.py`) — PASSED
- All 2 unit tool tests (`test_tools.py`) — PASSED
- All 13 Telephony / Hybrid IVR E2E tests (`test_ivr_flow.py`, `test_hybrid_intelligence_e2e.py`) — PASSED

---

## 5. Known Notes & Next Recommendations

- Windows Ollama's `gemma4:latest` has `"thinking"` enabled by default; setting `reasoning_effort: none` on `windows_ollama` prevents multi-turn tool timeouts and enables native tool selection.
- Note that `REMOTE_TOOLS_PREFER_REMOTE` defaults to `false`, so tenants with an active local Phase 1 `TenantDatabaseConfig` continue querying their configured DB while tenants without a local config route cleanly to Windows Django.
- Windows Django `/api/v1/tools/` and `/api/v1/tools/execute/` currently respond without enforcing incoming API token headers in dev mode; before connecting real customer data, enforce token/HMAC authentication on Django's side and set `REMOTE_TOOLS_AUTH_TOKEN` / `REMOTE_TOOLS_REQUIRE_AUTH=true` in FastAPI `.env`.

