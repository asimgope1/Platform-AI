# Security Audit Report

## 1. MacBook FastAPI Security Controls Implemented & Verified

| Control Area | Implementation Location | Verification Result |
| :--- | :--- | :--- |
| **Fail-Closed Production Startup Guard** | `platform_core/config/settings.py` (`validate_master_key`) | **Verified** (`test_stage3_production_fail_closed_settings`): blocks weak `MASTER_API_KEY`, requires `REMOTE_TOOLS_REQUIRE_AUTH=True`, requires remote credentials, and requires `https://` unless explicitly overridden. |
| **Remote HMAC-SHA256 Signing & Replay Protection** | `tools/remote/django_client.py` (`compute_hmac_signature`, `_build_headers`), `tools/remote/django_server_auth.py` (`ReplayNonceStore`, `verify_service_request`) | **Verified** (`test_stage3_hmac_signing_replay_and_rate_limit`): canonical method+path+timestamp+nonce+SHA256(body) signature, clock-skew guard, duplicate nonce rejection, body tampering detection, tenant header mismatch check. |
| **Remote Fail-Closed Client & HTTPS Guard** | `tools/remote/django_client.py` | **Verified**: rejects plain HTTP when `require_https=True` and rejects missing credentials when `require_auth=True`. |
| **Remote Concurrency & Rate Limits** | `tools/remote/django_client.py` (`_enforce_tenant_rate_limit`, `_get_tenant_semaphore`), `api/middleware/rate_limit.py` | **Verified**: per-tenant sliding-window rate limit (HTTP 429) and bounded per-tenant `asyncio.Semaphore`. |
| **Customer Ownership Enforcement** | `data_access/security.py` (`enforce_customer_authorization`), `data_access/service.py`, `tools/remote/django_client.py`, `tools/executor.py` | **Verified**: strips reserved internal args (`tenant_id`, `tenant_context`, `authorized_customer_id`, etc.) from LLM args, enforces `trusted_metadata["authorized_customer_id"]` before querying AND on returned records. |
| **Recursive PII & Credential Masking** | `data_access/security.py` (`mask_record_pii`, `CredentialCipher`) | **Verified**: masks `phone`, `email`, `addresses`, `shipping_address`, `billing_address`, `card_number`, `cvv`, `payment_details`, `pan`, `aadhaar`, `password`, `token` across nested dicts/lists; encrypts DB credentials at rest with Fernet/MultiFernet rotation. |
| **SQL & Prompt Injection Defense** | `data_access/security.py` (`validate_identifier`, `validate_read_only_sql`, `sanitize_tool_argument`), `tools/remote/django_client.py` | **Verified**: blocks raw SQL tool names (`execute_sql`, `raw_sql`, etc.), forbidden SQL args, stacked queries (`;`, `--`, `/* */`), `UNION SELECT`, and prompt-injection patterns. |
| **Vector RAG PII Guard & Tenant Isolation** | `retrieval/parsers.py` (`assert_no_transactional_pii_dump`), `data_access/sync/service.py` | **Verified**: rejects CSV/document transactional card/CVV/password dumps and transactional entity sync (`order`, `payment`, `customer`, `shipment`); cross-tenant search & delete blocked cleanly. |

---

## 2. External Security Blocker — Windows Django Server-Side Auth (Stage 2)

> [!WARNING]
> **EXTERNAL BLOCKER**: The Windows Django Business Tool Service repository running at `http://192.168.1.141:8000` resides on a separate Windows host and is **not accessible** inside `/Users/asimgope/.gemini/antigravity-ide/scratch`. During earlier live LAN probing (`GET /api/v1/tools/`, `GET /api/v1/tenants/tenants/`, `POST /api/v1/tools/execute/`), the Windows endpoint responded without rejecting unauthenticated requests (development mode).

### Ready-to-Apply Windows Django Remediation Delivered
We authored and unit-tested a matching server-side verifier and middleware module at [`tools/remote/django_server_auth.py`](file:///Users/asimgope/.gemini/antigravity-ide/scratch/ai-platform/tools/remote/django_server_auth.py):
1. Copy `verify_service_request`, `ReplayNonceStore`, `compute_hmac_signature`, and `DjangoServiceAuthMiddleware` into the Windows Django project (e.g. `core/middleware/service_auth.py`).
2. Add `"core.middleware.service_auth.DjangoServiceAuthMiddleware"` to Django `settings.MIDDLEWARE` (before views).
3. Configure identical shared secrets in both Windows `.env` and MacBook `ai-platform/.env`:
   - `DJANGO_INTERNAL_API_KEYS` (Windows) == `REMOTE_TOOLS_API_KEY` (MacBook)
   - `REMOTE_TOOLS_HMAC_SECRET` (identical high-entropy 32+ byte secret on both hosts)
4. Also set `REMOTE_TOOLS_REQUIRE_AUTH=true` on MacBook `ai-platform/.env` immediately once the Windows middleware is enabled, and place Django behind TLS (`https://`) before production cutover.

---

## 3. Secrets Hygiene Verification
- Audited all test assertions (`caplog` checks in `test_phase3a_stabilization.py`), tool audit rows (`arguments_masked`), and markdown reports: zero passwords, `.env` keys, Fernet master keys, or raw customer PII are written to logs or documentation.

