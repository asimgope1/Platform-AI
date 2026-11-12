# Production Deployment & Operations Guide

## 1. Topology & Responsibility Split
1. **MacBook / Linux Primary AI Tier (`services/api/main.py`)**:
   - Hosts FastAPI (`uvicorn`), `ChatOrchestrator`, `ToolLoopRunner`, Hybrid RAG + Real-Time Sync worker/endpoints, `UniversalDataAccessService`, and Telephony WebSocket server (`/v1/...`, `/ws/ivr/...`).
   - Internal database: PostgreSQL with `pgvector` extension (`Vector(384)` and `Vector(768)` columns, Alembic revisions `001`–`005`).
2. **Remote Windows / Backend Business Tool Service**:
   - Hosts Django `/api/v1/tools/` and `/api/v1/tools/execute/`.
   - **Required Before Production Cutover**: Deploy [`tools/remote/django_server_auth.py`](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/tools/remote/django_server_auth.py) middleware into the Windows Django `settings.MIDDLEWARE` behind HTTPS/mTLS.
3. **Ollama / GPU Inference Node (`config/providers.yaml`)**:
   - Serves `gemma4:latest` (with `"reasoning_effort": "none"` for tool-calling turns) or hosted OpenAI-compatible GPU inference for lower latency.

---

## 2. Required Production Environment Variables

Configure these strictly via your secret manager (never commit `.env` values to source control):

```bash
ENVIRONMENT=production
MASTER_API_KEY=<32+ character cryptographically random secret>
DATA_ACCESS_MASTER_KEY=<32+ character Fernet key derivation secret>
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<pg-host>:5432/<ai_platform_db>

# Remote Tool Security (Fail-Closed in ENVIRONMENT=production)
REMOTE_TOOLS_ENABLED=true
REMOTE_TOOLS_BASE_URL=https://<django-internal-domain>
REMOTE_TOOLS_REQUIRE_AUTH=true
REMOTE_TOOLS_REQUIRE_HTTPS_IN_PROD=true
REMOTE_TOOLS_API_KEY=<shared service API key matching Django DJANGO_INTERNAL_API_KEYS>
REMOTE_TOOLS_HMAC_SECRET=<shared 32+ byte HMAC-SHA256 signing secret>
REMOTE_TOOLS_TIMEOUT_SECONDS=5.0
REMOTE_TOOLS_MAX_RETRIES=2
REMOTE_TOOLS_PER_TENANT_CONCURRENCY=8
REMOTE_TOOLS_RATE_LIMIT_PER_MIN=120
```

Startup validation (`validate_master_key` in `platform_core/config/settings.py`) automatically aborts boot if `ENVIRONMENT=production` lacks a strong `MASTER_API_KEY`, remote credentials, `REMOTE_TOOLS_REQUIRE_AUTH=true`, or `https://` URL (unless TLS tunnel override is explicitly set).

---

## 3. Database Migration & Tenant Onboarding Sequence
1. Ensure PostgreSQL `vector` extension is enabled (`CREATE EXTENSION IF NOT EXISTS vector;`).
2. Run internal AI platform Alembic migrations:
   ```bash
   cd ai-platform
   alembic upgrade head   # applies 001 -> 002 -> 003 -> 004 -> 005
   ```
3. Onboard each tenant's external read-only DB (only a dedicated **read-only DB role** should be registered):
   - `POST /v1/tenants/databases`
   - `POST /v1/tenants/databases/{config_id}/test`
   - `POST /v1/tenants/databases/{config_id}/discover`
   - `PUT /v1/tenants/databases/{config_id}/mapping`
   - `POST /v1/tenants/databases/{config_id}/validate`
   - `POST /v1/tenants/databases/{config_id}/activate`

---

## 4. Health & Sync Verification Checks
- `GET /health` -> verifies `services.database == "healthy"`, `services.remote_tools` sync & stale flags, and request metrics.
- `GET /v1/sync/status` (Bearer tenant key) -> inspects outbox event backlog, tombstones, and dead-letter counts (`sync_dead_letters`).

---

## 5. Rollback Plan
1. **Remote Tool Regression / Outage**: Set `REMOTE_TOOLS_ENABLED=false` (or switch per-tenant routing to local `UniversalDataAccessService`) and restart API workers. Non-tool FAQ flows remain functional via Hybrid RAG; tool-enabled voice flows safely emit `tool_unavailable_fallback` without hallucinating.
2. **Mapping Update Regression**: Previous `schema_mapping` edits increment `mapping_version`; re-activate the last validated `TenantDatabaseConfig` via `/activate` or invalidate pool caches cleanly.
3. **Database Schema Rollback**: `alembic downgrade 004_tenant_db_configs` removes `sync_*` tables cleanly without touching customer business databases.

