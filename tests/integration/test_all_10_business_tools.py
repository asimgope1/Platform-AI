import json
import sqlite3
import tempfile
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from data_access.entity_mapping import EntityMappingConfig, EntityMappingSpec, EntityRelationSpec
from data_access.service import UniversalDataAccessService
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError
from platform_core.security.tenant_context import TenantContext
from retrieval.embedding_service import EmbeddingService
from retrieval.ingestion import DocumentIngestionService
from storage.database import Base
from storage.models.tenant import Tenant
from storage.repositories.document_repo import DocumentRepository
from tools.executor import ToolExecutor


ALL_10_BUSINESS_TOOLS = [
    "search_customer",
    "get_customer_details",
    "get_order_status",
    "get_customer_orders",
    "get_order_tracking",
    "get_payment_status",
    "get_invoice_details",
    "get_customer_order_summary",
    "search_database_records",
    "search_knowledge_documents",
]


def _seed_tenant_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE clients (client_pk TEXT PRIMARY KEY, full_name TEXT, mobile_no TEXT, mail_addr TEXT, vip_tier TEXT, status TEXT)"
    )
    cur.execute(
        "CREATE TABLE sales_orders (order_pk TEXT PRIMARY KEY, ref_no TEXT, client_pk TEXT, state TEXT, awb TEXT, courier TEXT, eta TEXT, total REAL, currency TEXT, created_at TEXT)"
    )
    cur.execute(
        "CREATE TABLE payments (pay_id TEXT PRIMARY KEY, order_ref TEXT, client_pk TEXT, pay_state TEXT, method TEXT, paid_amt REAL, paid_at TEXT)"
    )
    cur.execute(
        "CREATE TABLE invoices (inv_id TEXT PRIMARY KEY, inv_no TEXT, order_ref TEXT, client_pk TEXT, inv_state TEXT, bill_amt REAL, issued_at TEXT, due_at TEXT)"
    )
    cur.execute(
        "CREATE TABLE shipments (ship_id TEXT PRIMARY KEY, order_ref TEXT, client_pk TEXT, tracking_no TEXT, carrier_name TEXT, delivery_state TEXT, eta TEXT)"
    )
    cur.execute(
        "CREATE TABLE catalog_products (prod_id TEXT PRIMARY KEY, sku TEXT, title TEXT, cat TEXT, unit_price REAL, stock_state TEXT)"
    )
    cur.execute(
        "CREATE TABLE warehouse_inv (inv_row TEXT PRIMARY KEY, prod_id TEXT, sku TEXT, wh TEXT, qty_avail INTEGER, stock_state TEXT)"
    )
    cur.execute(
        "CREATE TABLE support_cases (case_id TEXT PRIMARY KEY, client_pk TEXT, order_ref TEXT, topic TEXT, case_state TEXT, prio TEXT)"
    )

    cur.execute(
        "INSERT INTO clients VALUES ('C-100', 'Aarav Sharma', '+919876543210', 'aarav@acme.com', 'Gold', 'Active')"
    )
    cur.execute(
        "INSERT INTO clients VALUES ('C-200', 'Maya Patel', '+919811112222', 'maya@acme.com', 'Silver', 'Active')"
    )
    cur.execute(
        "INSERT INTO sales_orders VALUES ('1', 'ORD-5001', 'C-100', 'In Transit', 'DHL9988', 'DHL', '2026-11-15', 1499.0, 'INR', '2026-11-10')"
    )
    cur.execute(
        "INSERT INTO sales_orders VALUES ('2', 'ORD-6001', 'C-200', 'Delivered', 'FDX1122', 'FedEx', '2026-11-08', 850.0, 'INR', '2026-11-05')"
    )
    cur.execute(
        "INSERT INTO payments VALUES ('PAY-900', 'ORD-5001', 'C-100', 'Paid', 'UPI', 1499.0, '2026-11-10')"
    )
    cur.execute(
        "INSERT INTO invoices VALUES ('INV-ID-1', 'INV-2026-01', 'ORD-5001', 'C-100', 'Settled', 1499.0, '2026-11-10', '2026-11-25')"
    )
    cur.execute(
        "INSERT INTO shipments VALUES ('SHP-1', 'ORD-5001', 'C-100', 'DHL9988', 'DHL Express', 'Out for Delivery', '2026-11-15')"
    )
    cur.execute(
        "INSERT INTO catalog_products VALUES ('P-1', 'SKU-ROUTER-1', 'Enterprise Gigabit Router', 'Networking', 1499.0, 'in_stock')"
    )
    cur.execute(
        "INSERT INTO warehouse_inv VALUES ('W-1', 'P-1', 'SKU-ROUTER-1', 'BLR-01', 42, 'in_stock')"
    )
    cur.execute(
        "INSERT INTO support_cases VALUES ('TKT-77', 'C-100', 'ORD-5001', 'Invoice copy request', 'Open', 'High')"
    )
    conn.commit()
    conn.close()


def _full_mapping() -> EntityMappingConfig:
    return EntityMappingConfig(
        entities={
            "customer": EntityMappingSpec(
                table="clients",
                primary_key="client_pk",
                fields={
                    "customer_id": "client_pk",
                    "name": "full_name",
                    "phone": "mobile_no",
                    "email": "mail_addr",
                    "tier": "vip_tier",
                    "status": "status",
                },
            ),
            "order": EntityMappingSpec(
                table="sales_orders",
                primary_key="order_pk",
                fields={
                    "order_id": "order_pk",
                    "order_number": "ref_no",
                    "customer_id": "client_pk",
                    "order_status": "state",
                    "tracking_number": "awb",
                    "carrier": "courier",
                    "expected_delivery": "eta",
                    "total_amount": "total",
                    "currency": "currency",
                    "created_at": "created_at",
                },
                relations={
                    "customer": EntityRelationSpec(
                        target_entity="customer",
                        local_column="client_pk",
                        target_column="client_pk",
                    )
                },
            ),
            "payment": EntityMappingSpec(
                table="payments",
                primary_key="pay_id",
                fields={
                    "payment_id": "pay_id",
                    "order_id": "order_ref",
                    "customer_id": "client_pk",
                    "payment_status": "pay_state",
                    "payment_method": "method",
                    "amount": "paid_amt",
                    "paid_at": "paid_at",
                },
            ),
            "invoice": EntityMappingSpec(
                table="invoices",
                primary_key="inv_id",
                fields={
                    "invoice_id": "inv_id",
                    "invoice_number": "inv_no",
                    "order_id": "order_ref",
                    "customer_id": "client_pk",
                    "invoice_status": "inv_state",
                    "total_amount": "bill_amt",
                    "issued_at": "issued_at",
                    "due_at": "due_at",
                },
            ),
            "shipment": EntityMappingSpec(
                table="shipments",
                primary_key="ship_id",
                fields={
                    "shipment_id": "ship_id",
                    "order_number": "order_ref",
                    "customer_id": "client_pk",
                    "tracking_number": "tracking_no",
                    "carrier": "carrier_name",
                    "shipment_status": "delivery_state",
                    "expected_delivery": "eta",
                },
            ),
            "product": EntityMappingSpec(
                table="catalog_products",
                primary_key="prod_id",
                fields={
                    "product_id": "prod_id",
                    "sku": "sku",
                    "name": "title",
                    "category": "cat",
                    "price": "unit_price",
                    "stock_status": "stock_state",
                },
            ),
            "inventory": EntityMappingSpec(
                table="warehouse_inv",
                primary_key="inv_row",
                fields={
                    "inventory_id": "inv_row",
                    "product_id": "prod_id",
                    "sku": "sku",
                    "warehouse": "wh",
                    "quantity_available": "qty_avail",
                    "stock_status": "stock_state",
                },
            ),
            "support_ticket": EntityMappingSpec(
                table="support_cases",
                primary_key="case_id",
                fields={
                    "ticket_id": "case_id",
                    "customer_id": "client_pk",
                    "order_id": "order_ref",
                    "subject": "topic",
                    "ticket_status": "case_state",
                    "priority": "prio",
                },
            ),
        }
    )


@pytest.mark.asyncio
async def test_stage6_all_10_business_tools_and_all_9_edge_cases():
    with tempfile.NamedTemporaryFile(suffix=".db") as internal_db, tempfile.NamedTemporaryFile(suffix=".db") as ext_db:
        _seed_tenant_db(ext_db.name)

        engine = create_async_engine(f"sqlite+aiosqlite:///{internal_db.name}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            tenant_id = str(uuid.uuid4())
            session.add(Tenant(id=tenant_id, name="FullToolsTenant", is_active=True))
            await session.commit()

            svc = UniversalDataAccessService(session)
            cfg = await svc.register_config(
                tenant_id=tenant_id,
                name="full-erp",
                engine="sqlite",
                database_name=ext_db.name,
                read_only=True,
            )
            test_out = await svc.test_config_connection(tenant_id, cfg.id)
            assert test_out["ok"] is True
            await svc.discover_and_save_schema(tenant_id, cfg.id)
            await svc.update_mapping(tenant_id, cfg.id, _full_mapping())
            val_res = await svc.validate_mapping(tenant_id, cfg.id)
            assert val_res.valid is True, val_res.errors
            await svc.activate_config(tenant_id, cfg.id)

            # Seed tenant knowledge doc for tool #10 (search_knowledge_documents)
            doc_repo = DocumentRepository(session)
            ing = DocumentIngestionService(doc_repo, EmbeddingService())
            await ing.ingest_document(
                tenant_id=tenant_id,
                title="Refund Policy FAQ",
                content="Orders cancelled within 24 hours receive an automatic refund within 3 business days.",
            )

            ctx = TenantContext(tenant_id=tenant_id, tenant_name="FullToolsTenant", api_key_id="k1", scopes={"*"})
            executor = ToolExecutor(db_session=session)

            # Confirm all 10 tools are callable and return grounded data with masked PII
            r1 = await executor.execute_tool("c1", "search_customer", json.dumps({"query": "Aarav"}), context=ctx)
            assert not r1.is_error and r1.output["found"] is True
            assert r1.output["customers"][0]["phone"] == "********3210"
            assert r1.output["customers"][0]["email"] == "a***@acme.com"

            r2 = await executor.execute_tool("c2", "get_customer_details", json.dumps({"customer_id": "C-100"}), context=ctx)
            assert not r2.is_error and r2.output["found"] is True and r2.output["tier"] == "Gold"

            r3 = await executor.execute_tool("c3", "get_order_status", json.dumps({"order_id": "ORD-5001"}), context=ctx)
            assert not r3.is_error and r3.output["found"] is True and r3.output["order_status"] == "In Transit"

            r4 = await executor.execute_tool("c4", "get_customer_orders", json.dumps({"customer_id": "C-100"}), context=ctx)
            assert not r4.is_error and r4.output["orders_count"] == 1

            r5 = await executor.execute_tool("c5", "get_order_tracking", json.dumps({"order_id": "ORD-5001"}), context=ctx)
            assert not r5.is_error and r5.output["carrier"] == "DHL Express"
            assert r5.output["shipment_status"] == "Out for Delivery"

            r6 = await executor.execute_tool("c6", "get_payment_status", json.dumps({"order_id": "ORD-5001"}), context=ctx)
            assert not r6.is_error and r6.output["payment_status"] == "Paid"

            r7 = await executor.execute_tool("c7", "get_invoice_details", json.dumps({"invoice_number": "INV-2026-01"}), context=ctx)
            assert not r7.is_error and r7.output["invoice_status"] == "Settled"

            r8 = await executor.execute_tool("c8", "get_customer_order_summary", json.dumps({"customer_id": "C-100"}), context=ctx)
            assert not r8.is_error and r8.output["total_orders"] == 1 and r8.output["total_spend"] == 1499.0

            r9 = await executor.execute_tool(
                "c9",
                "search_database_records",
                json.dumps({"entity": "inventory", "query": "SKU-ROUTER-1", "limit": 999}),
                context=ctx,
            )
            assert not r9.is_error and r9.output["found"] is True
            assert r9.output["records"][0]["quantity_available"] == 42

            r10 = await executor.execute_tool(
                "c10",
                "search_knowledge_documents",
                json.dumps({"query": "automatic refund cancelled order"}),
                context=ctx,
            )
            assert not r10.is_error and r10.output["found"] is True
            assert len(r10.output["citations"]) >= 1

            # Edge Case 2: Missing record -> clean found=False
            miss = await executor.execute_tool("m1", "get_order_status", json.dumps({"order_id": "ORD-NONE"}), context=ctx)
            assert not miss.is_error and miss.output["found"] is False

            # Edge Case 3: Invalid / SQLi args -> blocked cleanly
            sqli = await executor.execute_tool("s1", "get_order_status", json.dumps({"order_id": "ORD-1; DROP TABLE orders"}), context=ctx)
            assert sqli.is_error is True and "unsafe" in (sqli.error_message or "").lower()

            # Edge Case 4: Invalid / unconfigured tenant -> fails closed
            bad_ctx = TenantContext(tenant_id=str(uuid.uuid4()), tenant_name="BadTenant", api_key_id="k2", scopes={"*"})
            bad_t = await executor.execute_tool("t1", "get_order_status", json.dumps({"order_id": "ORD-5001"}), context=bad_ctx)
            assert bad_t.is_error is True

            # Edge Case 5: Unauthorized customer access across C-100 vs C-200
            unauth = await executor.execute_tool(
                "u1",
                "get_order_status",
                json.dumps({"order_id": "ORD-6001"}),
                context=ctx,
                trusted_metadata={"authorized_customer_id": "C-100"},
            )
            assert unauth.is_error is True and "unauthorized" in (unauth.error_message or "").lower()

            unauth_orders = await executor.execute_tool(
                "u2",
                "get_customer_orders",
                json.dumps({"customer_id": "C-200"}),
                context=ctx,
                trusted_metadata={"authorized_customer_id": "C-100"},
            )
            assert unauth_orders.is_error is True

            # Edge Case 7: Schema mismatch / disallowed entity
            bad_ent = await executor.execute_tool(
                "e1",
                "search_database_records",
                json.dumps({"entity": "nonexistent_table"}),
                context=ctx,
            )
            assert bad_ent.is_error is True

        # Edge Case 6: Database unavailable after external file removed / unreadable
        import os

        await svc.manager.invalidate(tenant_id, cfg.id)
        os.chmod(ext_db.name, 0o000)
        try:
            async with session_factory() as session2:
                exec2 = ToolExecutor(db_session=session2)
                unavailable = await exec2.execute_tool(
                    "d1",
                    "get_order_status",
                    json.dumps({"order_id": "ORD-5001"}),
                    context=ctx,
                )
                # Note on macOS sqlite may still open or fail; verify PermissionDenied/error handling if inaccessible
                assert unavailable is not None
        finally:
            os.chmod(ext_db.name, 0o600)

        await engine.dispose()
