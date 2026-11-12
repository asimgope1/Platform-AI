"""
Database-agnostic order status tool backed by the Universal Data Access Layer.
Resolves the caller's tenant database configuration dynamically without hardcoding table names or SQL.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.security import sanitize_tool_argument
from data_access.service import UniversalDataAccessService
from platform_core.errors.exceptions import PermissionDeniedError
from platform_core.security.tenant_context import TenantContext
from storage.database import async_session


async def get_order_status(
    order_id: Optional[str] = None,
    order_number: Optional[str] = None,
    customer_id: Optional[str] = None,
    *,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Look up live status and delivery details for an order ID from the active tenant's
    configured business database using its validated universal entity mapping.
    """
    if tenant_context is None:
        raise PermissionDeniedError("TenantContext is required for customer database tool execution.")

    clean_order_id = sanitize_tool_argument(order_id or order_number, "order_id", required=True, max_len=64)
    clean_customer_id = sanitize_tool_argument(customer_id, "customer_id", required=False, max_len=64)
    auth_customer = authorized_customer_id or clean_customer_id

    async def _run_with_session(session: AsyncSession) -> Dict[str, Any]:
        service = UniversalDataAccessService(session)
        rows = await service.query_entity(
            tenant_id=tenant_context.tenant_id,
            tool_name="get_order_status",
            entity="order",
            filters={"order_number": clean_order_id},
            include_related_customer=True,
            authorized_customer_id=auth_customer,
            mask_pii=True,
            limit=5,
            caller_subject=tenant_context.api_key_id,
        )
        if not rows:
            return {
                "found": False,
                "order_id": clean_order_id,
                "message": f"No order matching '{clean_order_id}' was found.",
            }

        rec = rows[0]
        return {
            "found": True,
            "order_id": rec.get("order_number") or rec.get("order_id") or clean_order_id,
            "order_status": rec.get("order_status"),
            "customer_id": rec.get("customer_id"),
            "customer_name": rec.get("customer_name"),
            "phone": rec.get("phone"),
            "tracking_number": rec.get("tracking_number"),
            "carrier": rec.get("carrier"),
            "expected_delivery": str(rec.get("expected_delivery")) if rec.get("expected_delivery") is not None else None,
            "total_amount": rec.get("total_amount"),
            "currency": rec.get("currency"),
        }

    if db_session is not None:
        return await _run_with_session(db_session)

    async with async_session() as own_session:
        return await _run_with_session(own_session)

