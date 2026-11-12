# Database Compatibility & Schema Mapping Report

## 1. Supported Database Engines & Security Controls

The Universal Data Access Layer (`data_access/`) connects to heterogeneous customer business databases strictly in **read-only mode** via per-tenant encrypted credentials (`CredentialCipher` using Fernet / `MultiFernet` key rotation).

| Engine | Adapter & Driver | SSL / TLS Options Supported | Timeout & Read-Only Enforcement | Pool Isolation | Verification Status in Workspace |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL** | `SQLAlchemyCoreAdapter` (`postgresql+psycopg2`) | `sslmode`, `sslrootcert`, `sslcert`, `sslkey` | `SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY`, `SET LOCAL statement_timeout = <ms>` | Per-`(tenant_id, config_id)` SQLAlchemy engine pool (`pool_size`, `max_overflow`, `pool_recycle`, `pool_pre_ping`) | Dialect SQL statements unit-tested (`test_stage4_sql_adapter_urls_ssl_and_read_only_dialects`); `psycopg2` driver not installed in `.venv`; local pgvector `::1:5432` blocked in sandbox |
| **MySQL / MariaDB** | `SQLAlchemyCoreAdapter` (`mysql+pymysql`) | `sslmode`, `connect_args={"ssl": {ca, cert, key}}` | `SET SESSION TRANSACTION READ ONLY`, `SET SESSION MAX_EXECUTION_TIME = <ms>` | Per-`(tenant_id, config_id)` engine pool | Dialect SQL & URL builder unit-tested; `pymysql` driver & live MySQL server not present in workspace |
| **Microsoft SQL Server** | `SQLAlchemyCoreAdapter` (`mssql+pyodbc`) | `Encrypt`, `TrustServerCertificate`, configurable `odbc_driver` | `SET LOCK_TIMEOUT <ms>` + SQLAlchemy Core `SELECT`-only AST + `read_only=True` policy | Per-`(tenant_id, config_id)` engine pool | Dialect SQL & URL builder unit-tested; `pyodbc` / SQL Server not present in workspace |
| **Oracle** | `SQLAlchemyCoreAdapter` (`oracle+oracledb`) | URL & connection options forwarding | `SET TRANSACTION READ ONLY` + dual-table ping (`SELECT 1 FROM DUAL`) | Per-`(tenant_id, config_id)` engine pool | Dialect SQL & URL builder unit-tested; `oracledb` & live Oracle DB not present in workspace |
| **MongoDB** | `MongoDBAdapter` (`pymongo`) | `tls`, `tlsCAFile` / `ssl_ca`, `readPreference=secondaryPreferred` | No `$where` / arbitrary JS, escaped regex literals, `.max_time_ms()`, `.limit(<= 50)`, read-only `find()` projection | `maxPoolSize` per client instance | **Unit-tested** with in-memory Mongo client stub (schema sampling, regex escaping, joins, projection); `pymongo` not installed in `.venv` |
| **SQLite** | `SQLAlchemyCoreAdapter` (`sqlite`) | Local file / URI | `PRAGMA query_only = ON`, `PRAGMA busy_timeout = <ms>` | `check_same_thread=False` isolated engine | **TESTED — DEVELOPMENT** (live E2E queries, schema discovery, joins, multi-schema isolation verified across multiple test suites) |

---

## 2. Schema Constraints & Query Safeguards
1. **Strict Identifier Validation (`validate_identifier`)**: All schema, table, column, and alias names must match `^[A-Za-z_][A-Za-z0-9_]{0,127}$`.
2. **Parameterized Queries Only**: Filter predicates bind parameters via SQLAlchemy Core `sa.table` / `sa.column` expressions (or structured PyMongo dict filters with escaped regex literals), eliminating string-interpolated SQL.
3. **Hard Row & Time Limits**: Every `QuerySpec` clamps `limit` to `max_rows <= 50` and wraps execution inside `asyncio.wait_for(..., timeout=timeout_seconds)`.
4. **Activation Gate**: A tenant DB config (`TenantDatabaseConfig`) cannot be activated until both `test_config_connection` succeeds AND `validate_mapping` verifies all mapped tables, columns, primary keys, required fields, and relations against the discovered schema snapshot.
5. **Credential Isolation**: Credentials are stored encrypted in `tenant_database_configs.encrypted_credentials` (internal AI DB only — never migrated to customer DBs and never returned in API responses or logs).

---

## 3. Universal Business Entity Mapping (All 8 Canonical Entities)

`UNIVERSAL_ENTITIES` and `REQUIRED_ENTITY_FIELDS` in `data_access/entity_mapping.py`:

| Canonical Entity | Required Mapped Fields | Supported Canonical Fields | Heuristic Auto-Suggestion |
| :--- | :--- | :--- | :---: |
| `customer` | `customer_id` | `customer_id`, `name`, `phone`, `email`, `status`, `tier` | Yes |
| `order` | `order_number`, `order_status` | `order_id`, `order_number`, `customer_id`, `order_status`, `tracking_number`, `carrier`, `expected_delivery`, `total_amount`, `currency`, `created_at` | Yes |
| `payment` | `payment_id`, `payment_status` | `payment_id`, `order_id`, `customer_id`, `payment_status`, `payment_method`, `amount`, `paid_at` | Yes |
| `invoice` | `invoice_number`, `invoice_status` | `invoice_id`, `invoice_number`, `order_id`, `customer_id`, `invoice_status`, `total_amount`, `issued_at`, `due_at` | Yes |
| `product` | `product_id`, `name` | `product_id`, `sku`, `name`, `category`, `price`, `stock_status` | Yes |
| `support_ticket` | `ticket_id`, `ticket_status` | `ticket_id`, `customer_id`, `order_id`, `subject`, `ticket_status`, `priority`, `updated_at` | Yes |
| `inventory` | `sku`, `quantity_available` | `inventory_id`, `product_id`, `sku`, `warehouse`, `quantity_available`, `reserved_quantity`, `reorder_level`, `stock_status`, `updated_at` | Yes |
| `shipment` | `tracking_number`, `shipment_status` | `shipment_id`, `order_id`, `order_number`, `customer_id`, `tracking_number`, `carrier`, `shipment_status`, `shipped_at`, `expected_delivery`, `delivered_at`, `origin`, `destination` | Yes |

