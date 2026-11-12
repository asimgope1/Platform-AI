"""Real-time knowledge & catalog synchronization API routes (tenant-isolated)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.sync.service import RealTimeSyncService
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from storage.database import get_db_session

router = APIRouter(prefix="/v1/sync", tags=["Synchronization"])


class SyncEventRequest(BaseModel):
    entity_type: str
    record_id: str
    event_type: str = "updated"
    version: int = 1
    title: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    process_immediately: bool = True


class ReconcileRecordItem(BaseModel):
    record_id: str
    version: int = 1
    title: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReconcileRequest(BaseModel):
    entity_type: str = "product_catalog"
    records: List[ReconcileRecordItem] = Field(default_factory=list)


@router.post("/events")
async def ingest_sync_event(
    body: SyncEventRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Ingest a transactional outbox / CDC knowledge or catalog sync event for the caller's tenant."""
    svc = RealTimeSyncService(session)
    ev, created = await svc.enqueue_event(
        tenant_id=context.tenant_id,
        entity_type=body.entity_type,
        record_id=body.record_id,
        event_type=body.event_type,
        version=body.version,
        title=body.title,
        payload=body.payload,
        idempotency_key=body.idempotency_key,
    )
    outcome = ev.status
    if body.process_immediately and ev.status == "pending":
        outcome = await svc.process_event(ev)
    return {
        "event_id": ev.id,
        "idempotency_key": ev.idempotency_key,
        "created_new": created,
        "status": outcome,
    }


@router.post("/process")
async def process_sync_queue(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Run incremental outbox worker for the caller's tenant."""
    svc = RealTimeSyncService(session)
    return await svc.process_pending_events(context.tenant_id)


@router.post("/reconcile")
async def reconcile_sync_snapshot(
    body: ReconcileRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Reconcile indexed knowledge state against an authoritative source snapshot."""
    svc = RealTimeSyncService(session)
    return await svc.reconcile_snapshot(
        context.tenant_id,
        entity_type=body.entity_type,
        records=[r.model_dump() for r in body.records],
    )


@router.get("/status")
async def sync_status(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Return tenant-scoped sync progress, active synced records, tombstones, and dead letters."""
    svc = RealTimeSyncService(session)
    return await svc.get_status(context.tenant_id)

