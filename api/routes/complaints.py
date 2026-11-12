from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.complaints import ComplaintCreateSchema, ComplaintUpdateSchema
from platform_core.errors.exceptions import PlatformError, ValidationError
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from storage.database import get_db_session
from storage.models.complaint import Complaint
from tools.business_data.complaints import _complaint_to_dict, register_complaint

router = APIRouter(prefix="/v1/complaints", tags=["Complaints"])

_ALLOWED_STATUSES = {"open", "in_progress", "escalated", "resolved", "closed"}
_ALLOWED_PRIORITIES = {"low", "normal", "high", "urgent"}


@router.post("", status_code=201)
async def create_complaint(
    payload: ComplaintCreateSchema,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Register a customer complaint via REST and dispatch to configured external API if enabled."""
    return await register_complaint(
        description=payload.description,
        category=payload.category,
        subject=payload.subject,
        order_id=payload.order_id,
        customer_id=payload.customer_id,
        caller_phone=payload.caller_phone,
        priority=payload.priority,
        tenant_context=context,
        db_session=session,
        call_uuid=payload.call_uuid,
        source_channel="rest_api",
    )


@router.get("")
async def list_complaints(
    status: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    customer_id: Optional[str] = Query(default=None),
    order_id: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """List tenant-scoped customer complaints."""
    stmt = select(Complaint).where(Complaint.tenant_id == context.tenant_id)
    if status:
        stmt = stmt.where(Complaint.status == status.strip().lower())
    if category:
        stmt = stmt.where(Complaint.category == category.strip().lower())
    if customer_id:
        stmt = stmt.where(Complaint.customer_id == customer_id.strip())
    if order_id:
        stmt = stmt.where(Complaint.order_id == order_id.strip())
    stmt = stmt.order_by(desc(Complaint.created_at)).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return {"count": len(rows), "complaints": [_complaint_to_dict(r) for r in rows]}


@router.get("/{ticket_number}")
async def get_complaint(
    ticket_number: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Retrieve a single complaint ticket strictly scoped to the authenticated tenant."""
    stmt = select(Complaint).where(
        Complaint.tenant_id == context.tenant_id,
        Complaint.ticket_number == ticket_number.strip().upper(),
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise PlatformError(
            message=f"Complaint ticket '{ticket_number}' not found.",
            code="complaint_not_found",
            status_code=404,
        )
    return _complaint_to_dict(row)


@router.patch("/{ticket_number}")
async def update_complaint(
    ticket_number: str,
    payload: ComplaintUpdateSchema,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Update complaint status, priority, or resolution notes within the authenticated tenant."""
    stmt = select(Complaint).where(
        Complaint.tenant_id == context.tenant_id,
        Complaint.ticket_number == ticket_number.strip().upper(),
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise PlatformError(
            message=f"Complaint ticket '{ticket_number}' not found.",
            code="complaint_not_found",
            status_code=404,
        )

    if payload.status is not None:
        norm_status = payload.status.strip().lower()
        if norm_status not in _ALLOWED_STATUSES:
            raise ValidationError(f"Invalid complaint status '{payload.status}'.")
        row.status = norm_status

    if payload.priority is not None:
        norm_prio = payload.priority.strip().lower()
        if norm_prio not in _ALLOWED_PRIORITIES:
            raise ValidationError(f"Invalid complaint priority '{payload.priority}'.")
        row.priority = norm_prio

    if payload.resolution_notes is not None:
        row.resolution_notes = payload.resolution_notes.strip()

    await session.commit()
    await session.refresh(row)
    return _complaint_to_dict(row)
