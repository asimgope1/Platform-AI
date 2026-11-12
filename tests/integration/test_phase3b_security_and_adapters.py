import json
import sqlite3
import tempfile
import uuid
from unittest.mock import MagicMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from data_access.connectors.base import ColumnMeta, ConnectionConfigSpec, DiscoveredSchema, QuerySpec, TableMeta
from data_access.connectors.mongodb_adapter import MongoDBAdapter
from data_access.connectors.sql_adapter import SQLAlchemyCoreAdapter, build_sqlalchemy_url
from data_access.entity_mapping import (
    REQUIRED_ENTITY_FIELDS,
    UNIVERSAL_ENTITIES,
    EntityMappingConfig,
    EntityMappingSpec,
    suggest_entity_mapping,
    validate_mapping_against_schema,
)
from data_access.security import mask_record_pii
from data_access.service import UniversalDataAccessService
from platform_core.config.settings import Settings, validate_master_key
from platform_core.errors.exceptions import PermissionDeniedError, PlatformError, ValidationError
from platform_core.security.tenant_context import TenantContext
from storage.database import Base
from storage.models.tenant import Tenant
from tools.remote.django_client import DjangoRemoteToolClient, compute_hmac_signature
from tools.remote.django_server_auth import ReplayNonceStore, verify_service_request


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: FastAPI Remote Auth, HMAC, Rate Limiting, Production Fail-Closed, PII
# ──────────────────────────────────────────────────────────────────────────────


def test_stage3_production_fail_closed_settings():
    with pytest.raises(ValueError, match="MASTER_API_KEY"):
        validate_master_key(Settings(ENVIRONMENT="production", MASTER_API_KEY="short"))

    long_key = "sk-prod-very-strong-master-api-key-0123456789abcdef"
    with pytest.raises(ValueError, match="REMOTE_TOOLS_REQUIRE_AUTH"):
        validate_master_key(
            Settings(
                ENVIRONMENT="production",
                MASTER_API_KEY=long_key,
                REMOTE_TOOLS_ENABLED=True,
                REMOTE_TOOLS_REQUIRE_AUTH=False,
            )
        )

    with pytest.raises(ValueError, match="credential"):
        validate_master_key(
            Settings(
                ENVIRONMENT="production",
                MASTER_API_KEY=long_key,
                REMOTE_TOOLS_ENABLED=True,
                REMOTE_TOOLS_REQUIRE_AUTH=True,
                REMOTE_TOOLS_BASE_URL="https://django.internal.local",
            )
        )

    with pytest.raises(ValueError, match="https://"):
        validate_master_key(
            Settings(
                ENVIRONMENT="production",
                MASTER_API_KEY=long_key,
                REMOTE_TOOLS_ENABLED=True,
                REMOTE_TOOLS_REQUIRE_AUTH=True,
                REMOTE_TOOLS_API_KEY="svc-key-1234567890",
                REMOTE_TOOLS_BASE_URL="http://192.168.1.141:8000",
            )
        )

    # Valid production config passes cleanly
    validate_master_key(
        Settings(
            ENVIRONMENT="production",
            MASTER_API_KEY=long_key,
            REMOTE_TOOLS_ENABLED=True,
            REMOTE_TOOLS_REQUIRE_AUTH=True,
            REMOTE_TOOLS_API_KEY="svc-key-1234567890",
            REMOTE_TOOLS_HMAC_SECRET="hmac-super-secret-0001",
            REMOTE_TOOLS_BASE_URL="https://tools.internal.company.com",
        )
    )


@pytest.mark.asyncio
async def test_stage3_hmac_signing_replay_and_rate_limit():
    captured_headers = {}
    captured_body = b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers, captured_body
        captured_headers = dict(request.headers)
        captured_body = request.content
        if request.url.path == "/api/v1/tools/execute/":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "tool_name": "get_order_status",
                    "data": {
                        "order_number": "ORD-1001",
                        "customer_id": "CUST-1",
                        "phone": "+919876543210",
                        "email": "alice@example.com",
                        "addresses": [{"line1": "221B Baker St", "zip": "10001"}],
                        "payment_details": {"card_number": "4111111111111111", "cvv": "123"},
                    },
                },
            )
        return httpx.Response(404)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = DjangoRemoteToolClient(
        "https://django.example.internal",
        api_key="key-abc",
        hmac_secret="super-secret-hmac-key",
        require_auth=True,
        require_https=True,
        rate_limit_per_minute=2,
        http_client=async_client,
    )
    ctx = TenantContext(tenant_id="t1", tenant_name="T1", api_key_id="k1", scopes={"*"})

    res = await client.execute_tool("get_order_status", {"order_id": "ORD-1001"}, context=ctx)
    # Recursive PII masking applied
    assert res["data"]["phone"] == "********3210"
    assert res["data"]["email"] == "a***@example.com"
    assert res["data"]["addresses"] == ["***"]
    assert res["data"]["payment_details"]["card_number"] == "***"
    assert res["data"]["payment_details"]["cvv"] == "***"

    # Server-side HMAC verification matches signed headers
    nonce_store = ReplayNonceStore()
    ok, code, _ = verify_service_request(
        method="POST",
        path="/api/v1/tools/execute/",
        headers=captured_headers,
        body_bytes=captured_body,
        expected_api_keys={"key-abc"},
        hmac_secret="super-secret-hmac-key",
        nonce_store=nonce_store,
    )
    assert (ok, code) == (True, 200)

    # Duplicate nonce replay is rejected
    ok2, code2, _ = verify_service_request(
        method="POST",
        path="/api/v1/tools/execute/",
        headers=captured_headers,
        body_bytes=captured_body,
        expected_api_keys={"key-abc"},
        hmac_secret="super-secret-hmac-key",
        nonce_store=nonce_store,
    )
    assert (ok2, code2) == (False, 401)

    # Tampered body fails signature check
    bad_headers = dict(captured_headers)
    bad_headers["x-nonce"] = "fresh-nonce-99999"
    bad_sig = compute_hmac_signature(
        secret="other-secret",
        method="POST",
        path="/api/v1/tools/execute/",
        timestamp=bad_headers["x-timestamp"],
        nonce=bad_headers["x-nonce"],
        body_bytes=captured_body,
    )
    bad_headers["x-signature"] = bad_sig
    ok3, code3, _ = verify_service_request(
        method="POST",
        path="/api/v1/tools/execute/",
        headers=bad_headers,
        body_bytes=captured_body,
        expected_api_keys={"key-abc"},
        hmac_secret="super-secret-hmac-key",
        nonce_store=nonce_store,
    )
    assert (ok3, code3) == (False, 401)

    # Rate limiter triggers on 3rd call within 1 minute
    await client.execute_tool("get_order_status", {"order_id": "ORD-1002"}, context=ctx)
    with pytest.raises(PlatformError) as exc_info:
        await client.execute_tool("get_order_status", {"order_id": "ORD-1003"}, context=ctx)
    assert exc_info.value.status_code == 429

    # Fail-closed auth & HTTPS guards
    with pytest.raises(PermissionDeniedError, match="HTTPS"):
        DjangoRemoteToolClient("http://insecure.local", require_https=True)
    unauth_client = DjangoRemoteToolClient("https://django.example.internal", require_auth=True, http_client=async_client)
    with pytest.raises(PermissionDeniedError, match="authentication required"):
        await unauth_client.fetch_and_validate_catalog()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Multi-Engine Adapters (PostgreSQL, MySQL, MSSQL, Oracle, Mongo, SQLite)
# ──────────────────────────────────────────────────────────────────────────────


def test_stage4_sql_adapter_urls_ssl_and_read_only_dialects():
    pg_url = build_sqlalchemy_url(
        ConnectionConfigSpec(
            config_id="c-pg",
            tenant_id="t1",
            engine="postgresql",
            host="pg.db.internal",
            database_name="sales",
            username="ro_user",
            password="pw",
            options={"sslmode": "verify-full", "sslrootcert": "/etc/ssl/ca.pem"},
        )
    )
    assert pg_url.drivername == "postgresql+psycopg2"
    assert pg_url.query["sslmode"] == "verify-full"
    assert pg_url.query["sslrootcert"] == "/etc/ssl/ca.pem"

    mssql_url = build_sqlalchemy_url(
        ConnectionConfigSpec(
            config_id="c-mssql",
            tenant_id="t1",
            engine="mssql",
            host="sql.internal",
            database_name="erp",
            options={"encrypt": "yes", "trust_server_certificate": "no"},
        )
    )
    assert mssql_url.drivername == "mssql+pyodbc"
    assert mssql_url.query["Encrypt"] == "yes"

    ora_url = build_sqlalchemy_url(
        ConnectionConfigSpec(config_id="c-ora", tenant_id="t1", engine="oracle", host="ora.internal", database_name="ORCL")
    )
    assert ora_url.drivername == "oracle+oracledb"

    my_url = build_sqlalchemy_url(
        ConnectionConfigSpec(config_id="c-my", tenant_id="t1", engine="mysql", host="my.internal", database_name="shop")
    )
    assert my_url.drivername == "mysql+pymysql"

    # Verify read-only/timeout SQL execution statements emitted for each SQL dialect
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        adapter = SQLAlchemyCoreAdapter(ConnectionConfigSpec(config_id="c-sq", tenant_id="t1", engine="sqlite", database_name=tf.name))
        for dialect_name, expected_frag in (
            ("postgresql", "READ ONLY"),
            ("mysql", "MAX_EXECUTION_TIME"),
            ("mssql", "LOCK_TIMEOUT"),
            ("oracle", "SET TRANSACTION READ ONLY"),
            ("sqlite", "PRAGMA query_only = ON"),
        ):
            mock_conn = MagicMock()
            mock_conn.dialect.name = dialect_name
            adapter._apply_read_only_and_timeout(mock_conn, 2500)
            executed_sql = " ".join(str(call.args[0]) for call in mock_conn.execute.call_args_list)
            assert expected_frag in executed_sql


@pytest.mark.asyncio
async def test_stage4_mongodb_adapter_read_only_schema_and_query():
    class _FakeCursor:
        def __init__(self, docs):
            self._docs = list(docs)

        def max_time_ms(self, _ms):
            return self

        def limit(self, n):
            self._docs = self._docs[:n]
            return self

        def sort(self, _key, _dir):
            return self

        def __iter__(self):
            return iter(self._docs)

    class _FakeCollection:
        def __init__(self, docs):
            self._docs = docs

        def find(self, flt, projection=None):
            import re as _re

            res = []
            for d in self._docs:
                ok = True
                for k, v in flt.items():
                    if isinstance(v, dict) and "$regex" in v:
                        flags = _re.IGNORECASE if "i" in str(v.get("$options", "")) else 0
                        ok = bool(_re.search(v["$regex"], str(d.get(k, "")), flags))
                    elif d.get(k) != v:
                        ok = False
                if ok:
                    res.append(d)
            return _FakeCursor(res)

        def find_one(self, flt):
            for d in self._docs:
                if all(d.get(k) == v for k, v in flt.items()):
                    return d
            return None

    class _FakeDB:
        def __init__(self):
            self._colls = {
                "orders_col": _FakeCollection([{"_id": "1", "ord_no": "ORD-99", "cust_id": "C1", "status": "shipped"}]),
                "customers_col": _FakeCollection([{"_id": "c1", "id": "C1", "full_name": "Meera"}]),
            }

        def command(self, _cmd):
            return {"ok": 1}

        def list_collection_names(self):
            return list(self._colls.keys())

        def __getitem__(self, name):
            return self._colls[name]

    class _FakeClient:
        def __init__(self):
            self._db = _FakeDB()

        def server_info(self):
            return {"version": "7.0.5"}

        def __getitem__(self, _name):
            return self._db

    spec = ConnectionConfigSpec(config_id="c-mongo", tenant_id="t1", engine="mongodb", database_name="bizdb", read_only=True)
    adapter = MongoDBAdapter(spec, client=_FakeClient())
    ok, _ = await adapter.test_connection()
    assert ok is True

    discovered = await adapter.discover_schema()
    assert discovered.engine == "mongodb"
    assert {t.name for t in discovered.tables} == {"orders_col", "customers_col"}

    from data_access.connectors.base import QueryFilter, QueryJoin

    qres = await adapter.execute_read_only(
        QuerySpec(
            table="orders_col",
            select_columns={"order_number": "ord_no", "order_status": "status", "customer_id": "cust_id"},
            filters=[QueryFilter(column="ord_no", operator="eq", value="ord-99")],
            joins=[
                QueryJoin(
                    right_table="customers_col",
                    left_column="cust_id",
                    right_column="id",
                    select_columns={"customer_name": "full_name"},
                )
            ],
            limit=5,
        )
    )
    assert qres.row_count == 1
    assert qres.records[0]["order_number"] == "ORD-99"
    assert qres.records[0]["customer_name"] == "Meera"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: Complete 8-Entity Universal Mapping Including inventory & shipment
# ──────────────────────────────────────────────────────────────────────────────


def test_stage5_universal_entities_include_inventory_and_shipment():
    assert {"customer", "order", "payment", "invoice", "product", "support_ticket", "inventory", "shipment"} <= set(
        UNIVERSAL_ENTITIES.keys()
    )
    assert "inventory" in REQUIRED_ENTITY_FIELDS
    assert "shipment" in REQUIRED_ENTITY_FIELDS

    discovered = DiscoveredSchema(
        engine="sqlite",
        version="3.45",
        schemas=["main"],
        tables=[
            TableMeta(
                name="warehouse_stock",
                columns=[
                    ColumnMeta(name="sku", data_type="TEXT"),
                    ColumnMeta(name="available_qty", data_type="INTEGER"),
                    ColumnMeta(name="warehouse_code", data_type="TEXT"),
                ],
            ),
            TableMeta(
                name="shipments",
                columns=[
                    ColumnMeta(name="tracking_no", data_type="TEXT"),
                    ColumnMeta(name="delivery_status", data_type="TEXT"),
                    ColumnMeta(name="courier", data_type="TEXT"),
                ],
            ),
        ],
    )
    suggested = suggest_entity_mapping(discovered)
    assert "inventory" in suggested.entities
    assert "shipment" in suggested.entities
    assert suggested.entities["inventory"].fields["sku"] == "sku"
    assert suggested.entities["shipment"].fields["tracking_number"] == "tracking_no"

    res = validate_mapping_against_schema(suggested, discovered)
    assert res.valid is True, res.errors

    # Missing required field rejected
    bad_cfg = EntityMappingConfig(
        entities={
            "inventory": EntityMappingSpec(table="warehouse_stock", fields={"sku": "sku"})
        }
    )
    bad_res = validate_mapping_against_schema(bad_cfg, discovered)
    assert bad_res.valid is False
    assert any("quantity_available" in err for err in bad_res.errors)
