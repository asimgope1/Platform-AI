# Database-Agnostic Multi-Tenant Architecture — Phase 1 Implementation

## 1. Architecture Overview

`ai-platform` separates its **internal AI platform database** from **configurable external customer databases** via a strict Universal Data Access Layer (`data_access/`):

```
  POST /v1/chat/completions (enable_tools=True, Bearer API key -> TenantContext)
         │
         ▼
  ChatOrchestrator / ToolLoopExecutor / ToolExecutor
    • Validates schema & required arguments
    • Strips reserved internal args & blocks SQLi / prompt injection
    • Injects trusted TenantContext, db_session, authorized_customer_id
         │
         ▼
  Tool: get_order_status (tools/business_data/orders.py)
         │
         ▼
  UniversalDataAccessService (data_access/service.py)
    • Resolves active TenantDatabaseConfig for tenant_id from internal DB
    • Decrypts external credentials (CredentialCipher / Fernet)
    • Translates canonical entity/field filters ("order", "order_number")
      using validated schema_mapping -> tenant physical tables/columns
    • Applies PII masking & customer-level authorization
    • Writes ToolExecutionAudit & ConnectionMetrics
         │
         ▼
  ConnectionManager -> BaseDatabaseAdapter
    ├── SQLAlchemyCoreAdapter (postgresql, mysql, mssql, oracle, sqlite)
    └── MongoDBAdapter (mongodb)
         │
         ▼
  External Customer Database (STRICT READ-ONLY: SELECT / find only, pooled, timed out)
```

### Preserved Components
All existing working subsystems remain intact without replacement:
- `telephony/` (Asterisk / IVR WebSocket, waiting/processing audio, barge-in, sentiment, Whisper STT, Edge/Odia TTS)
- `orchestration/` (`ChatOrchestrator`, `ToolLoopExecutor`), `retrieval/` (Hybrid RAG / pgvector / MiniLM embeddings), PDF ingestion
- Existing registered tools (`count_open_purchase_requests`, `get_customer_info`) alongside the refactored `get_order_status`
- Existing `/v1/chat/completions` API contract (`enable_tools: bool` and `authorized_customer_id: Optional[str]` added as backwards-compatible optional fields)

---

## 2. New Modules & Database Schema

### Modules Created
- `data_access/__init__.py`
- `data_access/security.py`: `CredentialCipher` (Fernet + `MultiFernet` rotation), `validate_identifier`, `validate_read_only_sql`, `sanitize_tool_argument`, `mask_phone`, `mask_email`, `mask_record_pii`, `enforce_customer_authorization`
- `data_access/models.py`: `TenantDatabaseConfig` (`tenant_database_configs`), `ToolExecutionAudit` (`tool_execution_audits`)
- `data_access/connectors/base.py`: `BaseDatabaseAdapter`, `ConnectionConfigSpec`, `DiscoveredSchema`, `TableMeta`, `ColumnMeta`, `ForeignKeyMeta`, `IndexMeta`, `RelationshipMeta`, `QueryFilter`, `QueryJoin`, `QuerySpec`, `QueryResult`
- `data_access/connectors/sql_adapter.py`: `build_sqlalchemy_url`, `SQLAlchemyCoreAdapter`
- `data_access/connectors/mongodb_adapter.py`: `MongoDBAdapter`
- `data_access/connectors/connection_manager.py`: `ConnectionMetrics`, `ConnectionManager`, `connection_manager`
- `data_access/schema_discovery.py`: `SchemaDiscoveryService`
- `data_access/entity_mapping.py`: `UNIVERSAL_ENTITIES`, `REQUIRED_ENTITY_FIELDS`, `EntityRelationSpec`, `EntityMappingSpec`, `EntityMappingConfig`, `MappingValidationResult`, `validate_mapping_against_schema`, `suggest_entity_mapping`
- `data_access/service.py`: `UniversalDataAccessService`
- `api/routes/tenant_databases.py`: Tenant onboarding, schema discovery, mapping validation, activation, credential rotation, and metrics REST API
- `tools/business_data/orders.py`: Refactored `get_order_status`
- `storage/migrations/versions/004_tenant_db_configs.py`: Alembic migration (internal platform DB only; **never run against customer DBs**)

### Updated Wiring
- `pyproject.toml`: added `"data_access*"` package discovery
- `storage/models/__init__.py`: exported `TenantDatabaseConfig`, `ToolExecutionAudit`
- `services/api/main.py`: mounted `tenant_databases.router`
- `services/api/lifespan.py`: disposes all external DB connection pools (`connection_manager.close_all()`) on shutdown
- `services/api/dependencies.py`: injects `session` into `ToolExecutor(timeout_seconds=10.0, db_session=session)`
- `tools/registry.py`, `tools/executor.py`, `orchestration/tool_loop.py`: registered `get_order_status`, enforced argument sanitization + reserved internal arg stripping + trusted metadata propagation
- `api/schemas/chat.py`, `api/routes/chat_completions.py`: added `enable_tools` and `authorized_customer_id`, injecting `tool_registry.get_all_tool_schemas()` when `enable_tools=True`
- `config/providers.yaml`, `providers/chat/openai_compatible.py`, `providers/chat/mock.py`: enabled `tool_calling` on `gemma`, serialized dict tool arguments for OpenAI-compatible responses, and handled `get_order_status` in `MockChatAdapter`

---

## 3. Database Adapter Interface & Engines

`BaseDatabaseAdapter` (`data_access/connectors/base.py`) defines four engine-agnostic async methods:
- `test_connection(timeout_seconds: float = 5.0) -> Tuple[bool, str]`
- `discover_schema(timeout_seconds: float = 10.0) -> DiscoveredSchema`
- `execute_read_only(query_spec: QuerySpec, timeout_seconds: float = 5.0, max_rows: int = 50) -> QueryResult`
- `close() -> None`

Supported engines:
- **Relational SQL** (`SQLAlchemyCoreAdapter` using pooled SQLAlchemy Core + `sqlalchemy.inspect()`): `postgresql` (`postgresql+psycopg2`), `mysql` (`mysql+pymysql`), `mssql` (`mssql+pyodbc`), `oracle` (`oracle+oracledb`), and `sqlite` (`sqlite:///...`).
  - Enforces session-level read-only & execution timeouts (`SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY` + `SET LOCAL statement_timeout` for PostgreSQL, `SET SESSION TRANSACTION READ ONLY` + `MAX_EXECUTION_TIME` for MySQL, `PRAGMA query_only = ON` for SQLite).
  - Builds parameterized `sa.select(...)` statements with validated identifiers (`validate_identifier`) and optional `LEFT OUTER JOIN` clauses.
- **Document Store** (`MongoDBAdapter`): `mongodb` — samples collection keys for schema discovery, runs parameterized `find(..., max_time_ms=...)` projections with strict read-only semantics.

*(Note: in this environment `.venv` includes `sqlalchemy`, `aiosqlite`, and `asyncpg`; external DB drivers `psycopg2`, `pymysql`, `pyodbc`, `oracledb`, `pymongo` use lazy driver imports and were verified via SQL builder + mock client unit tests, while end-to-end live DB execution was verified against separate multi-file SQLite databases.)*

---

## 4. Schema Discovery & Universal Entity Mapping

`SchemaDiscoveryService` inspects customer databases without modifying customer tables, producing `DiscoveredSchema` (`engine`, `version`, `schemas`, `tables` with columns/types/nullability/primary keys/foreign keys/indexes, and normalized `relationships`).

`UNIVERSAL_ENTITIES` (`data_access/entity_mapping.py`) maps arbitrary customer tables/columns onto six canonical entities:
1. `customer`: `customer_id`, `name`, `phone`, `email`, `status`, `tier` (required: `customer_id`)
2. `order`: `order_id`, `order_number`, `customer_id`, `order_status`, `tracking_number`, `carrier`, `expected_delivery`, `total_amount`, `currency`, `created_at` (required: `order_number`, `order_status`)
3. `payment`: `payment_id`, `order_id`, `customer_id`, `payment_status`, `payment_method`, `amount`, `paid_at` (required: `payment_id`, `payment_status`)
4. `invoice`: `invoice_id`, `invoice_number`, `order_id`, `customer_id`, `invoice_status`, `total_amount`, `issued_at`, `due_at` (required: `invoice_number`, `invoice_status`)
5. `product`: `product_id`, `sku`, `name`, `category`, `price`, `stock_status` (required: `product_id`, `name`)
6. `support_ticket`: `ticket_id`, `customer_id`, `order_id`, `subject`, `ticket_status`, `priority`, `updated_at` (required: `ticket_id`, `ticket_status`)

`validate_mapping_against_schema()` rejects unknown entities, invalid SQL identifiers, missing required entity fields, nonexistent physical tables/columns, and broken relation join targets before allowing database activation.

---

## 5. Tenant DB Config & Onboarding APIs

All endpoints under `/v1/tenants/databases` require `get_current_tenant` (`TenantContext`) and isolate configs by `tenant_id`:
- `POST /v1/tenants/databases`: register external database config (`read_only=True` required; credentials encrypted with `CredentialCipher` and never serialized in responses)
- `GET /v1/tenants/databases`: list tenant configs safely
- `POST /v1/tenants/databases/{config_id}/test`: verify connectivity & read-only probe
- `POST /v1/tenants/databases/{config_id}/discover`: run read-only schema discovery and generate heuristic `suggested_mapping`
- `GET /v1/tenants/databases/{config_id}/schema`: inspect stored discovered schema & mapping version
- `PUT /v1/tenants/databases/{config_id}/mapping`: save canonical entity mapping
- `POST /v1/tenants/databases/{config_id}/mapping/validate`: validate mapping against discovered schema
- `POST /v1/tenants/databases/{config_id}/activate` / `deactivate`: enforce validated mapping before serving tools
- `POST /v1/tenants/databases/{config_id}/rotate-credentials`: rotate password/URI or re-encrypt under active primary key
- `GET /v1/tenants/databases/metrics`: per-tenant query count, failures, rejected queries, average latency (ms), and recent `ToolExecutionAudit` rows

---

## 6. Refactored Tool (Phase 1 Scope): `get_order_status`

`tools/business_data/orders.py:get_order_status` accepts `order_id` and optional `customer_id`, sanitizes arguments via `sanitize_tool_argument`, and delegates to `UniversalDataAccessService.query_entity(entity="order", filters={"order_number": clean_order_id}, include_related_customer=True, mask_pii=True, ...)`.

Using the exact same tool definition, orchestrator, and `/v1/chat/completions` request (`"What is the status of order ORD-1001?"`):
- **Tenant A** (`customers` + `sales_orders.order_status`) returns `"Dispatched - In Transit"`
- **Tenant B** (`client_master` + `purchase_records.current_state`) returns `"Delivered to Hub"`

---

## 7. Security Controls Implemented

- **Credential Encryption & Key Rotation (`CredentialCipher`)**: External passwords and connection URIs encrypted at rest with Fernet / `MultiFernet` (`DATA_ACCESS_MASTER_KEY` / `MASTER_API_KEY`, `DATA_ACCESS_ROTATION_KEYS`). `_serialize_config()` never returns encrypted tokens or plaintext credentials.
- **Read-Only Enforcement**: Customer DB registrations and `SQLAlchemyCoreAdapter` reject `read_only=False`; sessions set engine-specific read-only pragmas/transactions; `validate_read_only_sql()` blocks non-`SELECT` verbs, stacked queries (`;`), comments (`--`, `/*`), and `UNION SELECT`.
- **SQL Injection & Prompt Injection Defense**: `sanitize_tool_argument()` rejects SQL injection and prompt-override patterns; `ToolExecutor` strips reserved internal parameters (`tenant_context`, `db_session`, `authorized_customer_id`) from model-supplied JSON arguments; SQL generation uses `validate_identifier()` + SQLAlchemy parameterized bind values (no LLM-generated SQL).
- **Multi-Tenant & Customer Authorization + PII Masking**: Tenant configs and audits are scoped by `TenantContext.tenant_id`; `enforce_customer_authorization()` verifies record ownership when `authorized_customer_id` is supplied (via request or caller context); `mask_record_pii()` masks sensitive fields such as phone numbers (`+919876543210` -> `********3210`) and emails before returning tool output to the LLM.
- **Audit Logging & Telemetry**: Every tool query records an internal `ToolExecutionAudit` entry (`tenant_id`, `tool_name`, `entity_name`, `database_engine`, `status`, `latency_ms`, `row_count`, `error_message`) without storing raw PII.

---

## 8. Observed Test Results

1. **Phase 1 Integration Suite** (`cd ai-platform && .venv/bin/pytest tests/integration/test_database_agnostic_architecture.py -v`):
   - **4 passed in 0.42s**:
     - `test_one_tool_two_databases_two_schemas_and_onboarding` — PASSED
     - `test_incorrect_mappings_sqli_and_security_controls` — PASSED
     - `test_gemma_openai_compatible_dynamic_tool_calling_on_two_schemas` — PASSED
     - `test_multi_engine_url_builders_and_mongodb_adapter` — PASSED

2. **Full `ai-platform` Test Suite** (`cd ai-platform && .venv/bin/pytest tests/ -q`):
   - **178 passed, 4 failed, 3 skipped, 3 xfailed, 1 error in 21.26s**.
   - Zero regressions from Phase 1 changes. The 4 pre-existing/environment failures & 1 error observed in this sandbox were:
     - `test_audio_endpoints`: offline sandbox blocked `speech.platform.bing.com:443` (Edge-TTS fell back to `audio/wav` vs expected `audio/mpeg`)
     - `test_full_api_level_rag_with_real_embeddings_and_weights`: sandbox blocked PostgreSQL socket `::1:5432`
     - `TestIVRRAGMiniModel::test_emotion_frustration_empathy`: pre-existing wording mismatch in expected substring
     - `TestIVRVoiceBargeIn::test_voice_barge_in_cancels_playback`: pre-existing IVR unit test failure
     - `ComoneDjangoTelephonyTests`: `DJANGO_SETTINGS_MODULE` unset when invoking standalone pytest

---

## 9. Remaining Limitations & Next Step (Pending Approval)

- **Remaining 9 Tools Not Yet Implemented (Strict Incremental Gate)**: `search_customer`, `get_customer_details`, `get_customer_orders`, `get_order_tracking`, `get_payment_status`, `get_invoice_details`, `get_customer_order_summary`, `search_database_records`, `search_knowledge_documents`.
- **Database Drivers**: Live network tests against external PostgreSQL/MySQL/MSSQL/Oracle/MongoDB instances require installing their optional DBAPI/driver packages (`psycopg2-binary`, `pymysql`, `pyodbc`, `oracledb`, `pymongo`) and running outside the network-restricted sandbox.
- **Continuous Sync / Knowledge Base Indexing**: Incremental sync workers to ingest read-only customer catalog/KB updates into pgvector can be layered cleanly on top of `UniversalDataAccessService` once all 10 tools are migrated.

