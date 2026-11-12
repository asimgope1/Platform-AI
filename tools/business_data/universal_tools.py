"""
Complete 10-tool database-agnostic business tool suite backed by `UniversalDataAccessService`
and `HybridRetrievalService`.

All relational/document business queries resolve tenant schemas dynamically through the
Universal Data Access Layer with zero hardcoded SQL or physical table names.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from data_access.entity_mapping import UNIVERSAL_ENTITIES
from data_access.security import sanitize_tool_argument
from data_access.service import UniversalDataAccessService
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError
from platform_core.security.tenant_context import TenantContext
from retrieval.embedding_service import EmbeddingService
from retrieval.search import SemanticSearchService
from services.knowledge.retrieval_service import HybridRetrievalService
from storage.database import async_session
from storage.repositories.document_repo import DocumentRepository


def _require_tenant(tenant_context: Optional[TenantContext]) -> TenantContext:
    if tenant_context is None or not tenant_context.tenant_id:
        raise PermissionDeniedError("TenantContext is required for customer database tool execution.")
    return tenant_context


@asynccontextmanager
async def _resolve_session(db_session: Optional[AsyncSession]) -> AsyncIterator[AsyncSession]:
    if db_session is not None:
        yield db_session
        return
    async with async_session() as own_session:
        yield own_session


def _clean_opt(val: Optional[str], name: str, max_len: int = 128) -> Optional[str]:
    if val is None:
        return None
    return sanitize_tool_argument(val, name, required=False, max_len=max_len)


async def search_customer(
    query: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    customer_id: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 5,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Search customer accounts by customer ID, name, phone, email, or search query."""
    ctx = _require_tenant(tenant_context)
    c_query = _clean_opt(query, "query")
    c_phone = _clean_opt(phone, "phone", 32)
    c_email = _clean_opt(email, "email", 128)
    c_cust = _clean_opt(customer_id, "customer_id", 64)
    c_name = _clean_opt(name, "name", 128)

    if not any([c_query, c_phone, c_email, c_cust, c_name]):
        raise ValidationError("Provide at least one search filter (query, name, phone, email, or customer_id).")

    effective_limit = max(1, min(int(limit or 5), 25))
    auth_cust = authorized_customer_id or c_cust

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        filters: Dict[str, Any] = {}
        ops: Dict[str, str] = {}
        if c_cust:
            filters["customer_id"] = c_cust
        if c_phone:
            filters["phone"] = c_phone
        if c_email:
            filters["email"] = c_email
        if c_name:
            filters["name"] = c_name
            ops["name"] = "ilike"

        if filters:
            rows = await service.query_entity(
                tenant_id=ctx.tenant_id,
                tool_name="search_customer",
                entity="customer",
                filters=filters,
                filter_operators=ops,
                authorized_customer_id=auth_cust,
                mask_pii=True,
                limit=effective_limit,
                caller_subject=ctx.api_key_id,
            )
        else:
            # Fuzzy query string: match by exact customer_id first, then fallback to name ilike
            rows = await service.query_entity(
                tenant_id=ctx.tenant_id,
                tool_name="search_customer",
                entity="customer",
                filters={"customer_id": c_query},
                authorized_customer_id=auth_cust,
                mask_pii=True,
                limit=effective_limit,
                caller_subject=ctx.api_key_id,
            )
            if not rows:
                rows = await service.query_entity(
                    tenant_id=ctx.tenant_id,
                    tool_name="search_customer",
                    entity="customer",
                    filters={"name": c_query},
                    filter_operators={"name": "ilike"},
                    authorized_customer_id=auth_cust,
                    mask_pii=True,
                    limit=effective_limit,
                    caller_subject=ctx.api_key_id,
                )

    return {
        "found": len(rows) > 0,
        "count": len(rows),
        "customers": rows,
        "message": f"Found {len(rows)} matching customer(s)." if rows else "No matching customer found.",
    }


async def get_customer_details(
    customer_id: Optional[str] = None,
    customer_reference: Optional[str] = None,
    phone: Optional[str] = None,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve a single customer profile by customer ID/reference or phone number."""
    ctx = _require_tenant(tenant_context)
    c_cust = _clean_opt(customer_id or customer_reference, "customer_id", 64)
    c_phone = _clean_opt(phone, "phone", 32)
    if not c_cust and not c_phone:
        raise ValidationError("Either 'customer_id' or 'phone' is required.")

    filters: Dict[str, Any] = {}
    if c_cust:
        filters["customer_id"] = c_cust
    elif c_phone:
        filters["phone"] = c_phone

    auth_cust = authorized_customer_id or c_cust

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="get_customer_details",
            entity="customer",
            filters=filters,
            authorized_customer_id=auth_cust,
            mask_pii=True,
            limit=5,
            caller_subject=ctx.api_key_id,
        )

    if not rows:
        return {"found": False, "customer_id": c_cust, "message": "Customer record not found."}

    rec = rows[0]
    return {"found": True, **rec}


async def get_customer_orders(
    customer_id: Optional[str] = None,
    customer_reference: Optional[str] = None,
    order_status: Optional[str] = None,
    limit: int = 10,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List recent orders belonging to a verified customer account."""
    ctx = _require_tenant(tenant_context)
    c_cust = _clean_opt(customer_id or customer_reference or authorized_customer_id, "customer_id", 64)
    if not c_cust:
        raise ValidationError("Missing required argument 'customer_id'.")
    c_status = _clean_opt(order_status, "order_status", 32)
    effective_limit = max(1, min(int(limit or 10), 25))
    auth_cust = authorized_customer_id or c_cust

    filters: Dict[str, Any] = {"customer_id": c_cust}
    if c_status:
        filters["order_status"] = c_status

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="get_customer_orders",
            entity="order",
            filters=filters,
            include_related_customer=False,
            authorized_customer_id=auth_cust,
            mask_pii=True,
            order_by="created_at",
            descending=True,
            limit=effective_limit,
            caller_subject=ctx.api_key_id,
        )

    return {
        "found": len(rows) > 0,
        "customer_id": c_cust,
        "orders_count": len(rows),
        "orders": rows,
        "message": f"Retrieved {len(rows)} order(s)." if rows else f"No orders found for customer '{c_cust}'.",
    }


async def get_order_tracking(
    order_id: Optional[str] = None,
    order_number: Optional[str] = None,
    tracking_number: Optional[str] = None,
    customer_id: Optional[str] = None,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve live carrier, tracking number, shipment status, and ETA for an order or tracking ID."""
    ctx = _require_tenant(tenant_context)
    c_order = _clean_opt(order_id or order_number, "order_id", 64)
    c_tracking = _clean_opt(tracking_number, "tracking_number", 64)
    c_cust = _clean_opt(customer_id, "customer_id", 64)
    if not c_order and not c_tracking:
        raise ValidationError("Either 'order_id' or 'tracking_number' is required.")
    auth_cust = authorized_customer_id or c_cust

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        rows: List[Dict[str, Any]] = []
        # Prefer dedicated `shipment` entity if mapped for the tenant, then fall back to `order`
        try:
            s_filters = {"tracking_number": c_tracking} if c_tracking else {"order_number": c_order}
            rows = await service.query_entity(
                tenant_id=ctx.tenant_id,
                tool_name="get_order_tracking",
                entity="shipment",
                filters=s_filters,
                authorized_customer_id=auth_cust,
                mask_pii=True,
                limit=5,
                caller_subject=ctx.api_key_id,
            )
        except ValidationError:
            rows = []

        if not rows:
            o_filters = {"tracking_number": c_tracking} if c_tracking else {"order_number": c_order}
            rows = await service.query_entity(
                tenant_id=ctx.tenant_id,
                tool_name="get_order_tracking",
                entity="order",
                filters=o_filters,
                include_related_customer=True,
                authorized_customer_id=auth_cust,
                mask_pii=True,
                limit=5,
                caller_subject=ctx.api_key_id,
            )

    if not rows:
        return {
            "found": False,
            "order_id": c_order,
            "tracking_number": c_tracking,
            "message": "No shipment tracking record was found.",
        }

    rec = rows[0]
    return {
        "found": True,
        "order_id": rec.get("order_number") or rec.get("order_id") or c_order,
        "tracking_number": rec.get("tracking_number"),
        "carrier": rec.get("carrier"),
        "shipment_status": rec.get("shipment_status") or rec.get("order_status"),
        "expected_delivery": str(rec.get("expected_delivery")) if rec.get("expected_delivery") is not None else None,
        "delivered_at": str(rec.get("delivered_at")) if rec.get("delivered_at") is not None else None,
        "customer_id": rec.get("customer_id"),
    }


async def get_payment_status(
    payment_id: Optional[str] = None,
    order_id: Optional[str] = None,
    order_number: Optional[str] = None,
    customer_id: Optional[str] = None,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Check payment status, method, and amount by payment ID, order ID, or customer ID."""
    ctx = _require_tenant(tenant_context)
    c_pay = _clean_opt(payment_id, "payment_id", 64)
    c_ord = _clean_opt(order_id or order_number, "order_id", 64)
    c_cust = _clean_opt(customer_id, "customer_id", 64)
    if not any([c_pay, c_ord, c_cust]):
        raise ValidationError("Provide at least one of 'payment_id', 'order_id', or 'customer_id'.")
    auth_cust = authorized_customer_id or c_cust

    filters: Dict[str, Any] = {}
    if c_pay:
        filters["payment_id"] = c_pay
    if c_ord:
        filters["order_id"] = c_ord
    if c_cust and not (c_pay or c_ord):
        filters["customer_id"] = c_cust

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="get_payment_status",
            entity="payment",
            filters=filters,
            authorized_customer_id=auth_cust,
            mask_pii=True,
            limit=5,
            caller_subject=ctx.api_key_id,
        )

    if not rows:
        return {"found": False, "payment_id": c_pay, "order_id": c_ord, "message": "Payment record not found."}

    rec = rows[0]
    return {"found": True, **rec, "payments": rows}


async def get_invoice_details(
    invoice_number: Optional[str] = None,
    invoice_id: Optional[str] = None,
    order_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve invoice status, billing amount, issue date, and due date."""
    ctx = _require_tenant(tenant_context)
    c_inv_no = _clean_opt(invoice_number, "invoice_number", 64)
    c_inv_id = _clean_opt(invoice_id, "invoice_id", 64)
    c_ord = _clean_opt(order_id, "order_id", 64)
    c_cust = _clean_opt(customer_id, "customer_id", 64)
    if not any([c_inv_no, c_inv_id, c_ord, c_cust]):
        raise ValidationError("Provide at least one of 'invoice_number', 'invoice_id', 'order_id', or 'customer_id'.")
    auth_cust = authorized_customer_id or c_cust

    filters: Dict[str, Any] = {}
    if c_inv_no:
        filters["invoice_number"] = c_inv_no
    elif c_inv_id:
        filters["invoice_id"] = c_inv_id
    elif c_ord:
        filters["order_id"] = c_ord
    elif c_cust:
        filters["customer_id"] = c_cust

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="get_invoice_details",
            entity="invoice",
            filters=filters,
            authorized_customer_id=auth_cust,
            mask_pii=True,
            limit=5,
            caller_subject=ctx.api_key_id,
        )

    if not rows:
        return {
            "found": False,
            "invoice_number": c_inv_no or c_inv_id,
            "message": "Invoice record not found.",
        }

    rec = rows[0]
    return {"found": True, **rec}


async def get_customer_order_summary(
    customer_id: Optional[str] = None,
    customer_reference: Optional[str] = None,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a consolidated read-only summary of a customer's profile, order counts, and total value."""
    ctx = _require_tenant(tenant_context)
    c_cust = _clean_opt(customer_id or customer_reference or authorized_customer_id, "customer_id", 64)
    if not c_cust:
        raise ValidationError("Missing required argument 'customer_id'.")
    auth_cust = authorized_customer_id or c_cust

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        cust_rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="get_customer_order_summary",
            entity="customer",
            filters={"customer_id": c_cust},
            authorized_customer_id=auth_cust,
            mask_pii=True,
            limit=1,
            caller_subject=ctx.api_key_id,
        )
        order_rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="get_customer_order_summary",
            entity="order",
            filters={"customer_id": c_cust},
            authorized_customer_id=auth_cust,
            mask_pii=True,
            limit=25,
            caller_subject=ctx.api_key_id,
        )

    if not cust_rows and not order_rows:
        return {"found": False, "customer_id": c_cust, "message": f"No records found for customer '{c_cust}'."}

    status_breakdown: Dict[str, int] = {}
    total_spend = 0.0
    for ord_rec in order_rows:
        st = str(ord_rec.get("order_status") or "unknown").lower()
        status_breakdown[st] = status_breakdown.get(st, 0) + 1
        amt = ord_rec.get("total_amount")
        if isinstance(amt, (int, float)):
            total_spend += float(amt)

    return {
        "found": True,
        "customer_id": c_cust,
        "customer": cust_rows[0] if cust_rows else None,
        "total_orders": len(order_rows),
        "total_spend": round(total_spend, 2),
        "status_breakdown": status_breakdown,
        "recent_orders": order_rows[:5],
    }


async def search_database_records(
    entity: str,
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Safe schema-validated read-only search across canonical business entities.
    Strictly blocks arbitrary SQL, unmapped entities, and unmapped fields.
    """
    ctx = _require_tenant(tenant_context)
    clean_entity = sanitize_tool_argument(entity, "entity", required=True, max_len=32).lower()
    if clean_entity not in UNIVERSAL_ENTITIES:
        raise ValidationError(
            f"Unsupported entity '{clean_entity}'. Allowed entities: {sorted(UNIVERSAL_ENTITIES.keys())}."
        )

    effective_limit = max(1, min(int(limit or 10), 25))
    sanitized_filters: Dict[str, Any] = {}
    ops: Dict[str, str] = {}

    if filters is not None:
        if not isinstance(filters, dict):
            raise ValidationError("'filters' must be a JSON object of canonical field names to values.")
        for fk, fv in filters.items():
            key = sanitize_tool_argument(str(fk), "filter_key", required=True, max_len=64)
            if isinstance(fv, str):
                sanitized_filters[key] = sanitize_tool_argument(fv, key, required=False, max_len=128)
            elif isinstance(fv, (int, float, bool)):
                sanitized_filters[key] = fv
            elif fv is not None:
                raise ValidationError(f"Unsupported filter value type for '{key}'.")

    c_query = _clean_opt(query, "query", 128)
    if not sanitized_filters and c_query:
        # Map generic query to the entity's natural primary text search field
        text_search_field = {
            "customer": "name",
            "order": "order_number",
            "payment": "payment_id",
            "invoice": "invoice_number",
            "product": "name",
            "support_ticket": "subject",
            "inventory": "sku",
            "shipment": "tracking_number",
        }[clean_entity]
        sanitized_filters[text_search_field] = c_query
        ops[text_search_field] = "ilike"

    if not sanitized_filters and authorized_customer_id and "customer_id" in UNIVERSAL_ENTITIES[clean_entity]:
        sanitized_filters["customer_id"] = authorized_customer_id

    async with _resolve_session(db_session) as session:
        service = UniversalDataAccessService(session)
        rows = await service.query_entity(
            tenant_id=ctx.tenant_id,
            tool_name="search_database_records",
            entity=clean_entity,
            filters=sanitized_filters,
            filter_operators=ops,
            authorized_customer_id=authorized_customer_id,
            mask_pii=True,
            limit=effective_limit,
            caller_subject=ctx.api_key_id,
        )

    return {
        "found": len(rows) > 0,
        "entity": clean_entity,
        "count": len(rows),
        "records": rows,
    }


async def search_knowledge_documents(
    query: str,
    category: Optional[str] = None,
    top_k: int = 4,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """Search verified tenant knowledge base documents using hybrid semantic + lexical retrieval."""
    ctx = _require_tenant(tenant_context)
    clean_query = sanitize_tool_argument(query, "query", required=True, max_len=256)
    c_category = _clean_opt(category, "category", 64)
    effective_k = max(1, min(int(top_k or 4), 10))

    async with _resolve_session(db_session) as session:
        doc_repo = DocumentRepository(session)
        emb_service = EmbeddingService()
        sem_service = SemanticSearchService(doc_repo=doc_repo, embedding_service=emb_service)
        raw_results = await sem_service.search(
            tenant_id=ctx.tenant_id,
            query=HybridRetrievalService._augment_cross_lingual_query(clean_query),
            top_k=max(effective_k * 2, 6),
        )
        reranked = HybridRetrievalService.hybrid_rerank(
            query=clean_query,
            raw_results=raw_results,
            preferred_category=c_category,
            top_k=effective_k,
        )

    matches = []
    for chunk, score in reranked:
        matches.append(
            {
                "document_id": getattr(chunk, "document_id", None),
                "chunk_index": getattr(chunk, "chunk_index", 0),
                "content": getattr(chunk, "content", ""),
                "score": score,
            }
        )

    return {
        "found": len(matches) > 0,
        "count": len(matches),
        "matches": matches,
        "citations": [f"doc:{m['document_id']}#chunk-{m['chunk_index']}" for m in matches],
    }

