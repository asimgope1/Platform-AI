"""
Real-Time Knowledge & Catalog Synchronization Service.

Provides:
  • Transactional outbox / webhook event ingestion (`enqueue_event`)
  • Idempotent, version-ordered worker processing (`process_pending_events`)
  • Soft-delete / tombstone cleanup (`event_type == "deleted"`)
  • Exponential backoff retries + dead-letter queue (`SyncDeadLetter`)
  • Incremental polling for non-PII catalog data (`poll_incremental_product_catalog`)
  • Snapshot reconciliation (`reconcile_snapshot`)

Transactional queries (orders, payments, shipments, customer PII) NEVER sync into vectors;
they always execute live via UniversalDataAccessService or remote business tools.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.service import UniversalDataAccessService
from data_access.sync.models import SyncDeadLetter, SyncEvent, SyncRecordState
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError
from retrieval.embedding_service import EmbeddingService
from retrieval.ingestion import DocumentIngestionService
from retrieval.parsers import assert_no_transactional_pii_dump
from storage.repositories.document_repo import DocumentRepository

ALLOWED_SYNC_ENTITY_TYPES = frozenset(
    {
        "product_catalog",
        "product",
        "faq_policy",
        "support_article",
        "metadata",
    }
)
FORBIDDEN_TRANSACTIONAL_SYNC_TYPES = frozenset(
    {"order", "orders", "payment", "payments", "invoice", "customer", "pii", "shipment", "card"}
)


def _format_sync_document_text(entity_type: str, record_id: str, title: str, payload: Dict[str, Any]) -> str:
    content_body = str(payload.get("content") or payload.get("description") or "").strip()
    if content_body:
        return f"[{entity_type}:{record_id}] {title}\n{content_body}"
    kv = ", ".join(f"{k}: {v}" for k, v in sorted(payload.items()) if v is not None)
    return f"[{entity_type}:{record_id}] {title}\n{kv}".strip()


class RealTimeSyncService:
    """Tenant-isolated incremental sync & outbox processor for knowledge/catalog updates."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        embedding_service: Optional[EmbeddingService] = None,
        base_retry_seconds: float = 2.0,
    ):
        self.session = session
        self.embedding_service = embedding_service or EmbeddingService()
        self.base_retry_seconds = max(0.01, float(base_retry_seconds))

    async def enqueue_event(
        self,
        *,
        tenant_id: str,
        entity_type: str,
        record_id: str,
        event_type: str,
        version: int = 1,
        title: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        max_retries: int = 3,
    ) -> tuple[SyncEvent, bool]:
        """
        Enqueue a knowledge/catalog sync event idempotently.
        Returns `(SyncEvent, created_new: bool)`.
        """
        etype = (entity_type or "").strip().lower()
        if etype in FORBIDDEN_TRANSACTIONAL_SYNC_TYPES or etype not in ALLOWED_SYNC_ENTITY_TYPES:
            raise PermissionDeniedError(
                f"Entity type '{entity_type}' is not eligible for RAG sync (allowed: {sorted(ALLOWED_SYNC_ENTITY_TYPES)}). "
                "Transactional entities must be queried live via business tools."
            )

        ev_action = (event_type or "").strip().lower()
        if ev_action not in ("created", "updated", "deleted"):
            raise ValidationError(f"Unsupported sync event_type '{event_type}'. Must be created|updated|deleted.")

        if int(version) < 1:
            raise ValidationError("Sync event version must be >= 1.")

        body_payload = dict(payload or {})
        if ev_action != "deleted":
            preview_text = _format_sync_document_text(etype, record_id, title or record_id, body_payload)
            assert_no_transactional_pii_dump(preview_text, filename=f"{etype}:{record_id}")

        idem = (idempotency_key or f"{etype}:{record_id}:v{int(version)}:{ev_action}").strip()
        existing_res = await self.session.execute(
            select(SyncEvent).where(
                SyncEvent.tenant_id == tenant_id,
                SyncEvent.idempotency_key == idem,
            )
        )
        existing = existing_res.scalar_one_or_none()
        if existing is not None:
            return existing, False

        ev = SyncEvent(
            tenant_id=tenant_id,
            idempotency_key=idem,
            entity_type=etype,
            record_id=str(record_id).strip(),
            event_type=ev_action,
            version=int(version),
            title=(title or f"{etype}:{record_id}").strip(),
            payload=body_payload,
            status="pending",
            max_retries=max(1, int(max_retries)),
        )
        self.session.add(ev)
        await self.session.commit()
        await self.session.refresh(ev)
        return ev, True

    async def process_event(
        self,
        event: SyncEvent,
        *,
        ingestion_override: Optional[Callable[[SyncEvent], Any]] = None,
    ) -> str:
        """Process one pending SyncEvent with version ordering, tombstone deletes, and DLQ handling."""
        if event.status in ("completed", "ignored_stale", "dead_letter"):
            return event.status

        now = datetime.now(timezone.utc)
        state_res = await self.session.execute(
            select(SyncRecordState).where(
                SyncRecordState.tenant_id == event.tenant_id,
                SyncRecordState.entity_type == event.entity_type,
                SyncRecordState.record_id == event.record_id,
            )
        )
        state = state_res.scalar_one_or_none()

        # Strict version ordering: reject out-of-order stale events
        if state is not None and event.version <= state.version:
            event.status = "ignored_stale"
            event.processed_at = now
            event.last_error = f"Ignored stale version {event.version} <= current {state.version}"
            await self.session.commit()
            return "ignored_stale"

        try:
            doc_repo = DocumentRepository(self.session)
            ingestion = DocumentIngestionService(doc_repo, self.embedding_service)

            if ingestion_override is not None:
                await ingestion_override(event)

            if event.event_type == "deleted":
                if state and state.document_id:
                    await doc_repo.delete_document(state.document_id, event.tenant_id)
                if state is None:
                    state = SyncRecordState(
                        tenant_id=event.tenant_id,
                        entity_type=event.entity_type,
                        record_id=event.record_id,
                        version=event.version,
                        document_id=None,
                        is_deleted=True,
                    )
                    self.session.add(state)
                else:
                    state.version = event.version
                    state.document_id = None
                    state.is_deleted = True
                    state.content_hash = None
            else:
                text = _format_sync_document_text(
                    event.entity_type,
                    event.record_id,
                    event.title or event.record_id,
                    event.payload or {},
                )
                assert_no_transactional_pii_dump(text, filename=f"{event.entity_type}:{event.record_id}")
                c_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

                doc_id: Optional[str] = state.document_id if state else None
                if doc_id:
                    updated_doc = await ingestion.reindex_document(
                        tenant_id=event.tenant_id,
                        document_id=doc_id,
                        content=text,
                        title=event.title or event.record_id,
                        source_type=event.entity_type,
                        version=str(event.version),
                    )
                    if updated_doc is None:
                        doc_id = None
                if not doc_id:
                    new_doc = await ingestion.ingest_document(
                        tenant_id=event.tenant_id,
                        title=event.title or f"{event.entity_type}:{event.record_id}",
                        content=text,
                        source_type=event.entity_type,
                        version=str(event.version),
                    )
                    doc_id = new_doc.id

                if state is None:
                    state = SyncRecordState(
                        tenant_id=event.tenant_id,
                        entity_type=event.entity_type,
                        record_id=event.record_id,
                        version=event.version,
                        document_id=doc_id,
                        content_hash=c_hash,
                        is_deleted=False,
                    )
                    self.session.add(state)
                else:
                    state.version = event.version
                    state.document_id = doc_id
                    state.content_hash = c_hash
                    state.is_deleted = False

            event.status = "completed"
            event.processed_at = now
            event.last_error = None
            await self.session.commit()
            return "completed"

        except Exception as exc:
            event.retry_count += 1
            event.last_error = f"{type(exc).__name__}: {exc}"
            if event.retry_count >= event.max_retries:
                event.status = "dead_letter"
                event.processed_at = now
                dlq = SyncDeadLetter(
                    sync_event_id=event.id,
                    tenant_id=event.tenant_id,
                    entity_type=event.entity_type,
                    record_id=event.record_id,
                    event_type=event.event_type,
                    version=event.version,
                    payload=event.payload or {},
                    error_message=event.last_error,
                    retry_count=event.retry_count,
                )
                self.session.add(dlq)
                await self.session.commit()
                return "dead_letter"

            delay = min(300.0, self.base_retry_seconds * (2 ** (event.retry_count - 1)))
            event.next_retry_at = now + timedelta(seconds=delay)
            await self.session.commit()
            return "retry_scheduled"

    async def process_pending_events(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        ignore_backoff: bool = False,
        ingestion_override: Optional[Callable[[SyncEvent], Any]] = None,
    ) -> Dict[str, int]:
        """Process ready outbox events for a single tenant cleanly."""
        now = datetime.now(timezone.utc)
        stmt = select(SyncEvent).where(
            SyncEvent.tenant_id == tenant_id,
            SyncEvent.status == "pending",
        )
        if not ignore_backoff:
            stmt = stmt.where(or_(SyncEvent.next_retry_at.is_(None), SyncEvent.next_retry_at <= now))
        stmt = stmt.order_by(SyncEvent.created_at.asc(), SyncEvent.version.asc()).limit(max(1, min(int(limit), 200)))

        res = await self.session.execute(stmt)
        events = list(res.scalars().all())
        counts = {"completed": 0, "ignored_stale": 0, "retry_scheduled": 0, "dead_letter": 0}
        for ev in events:
            outcome = await self.process_event(ev, ingestion_override=ingestion_override)
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    async def poll_incremental_product_catalog(
        self,
        tenant_id: str,
        *,
        limit: int = 25,
    ) -> Dict[str, Any]:
        """Poll read-only tenant business DB `product` entity and sync non-PII catalog items."""
        uda = UniversalDataAccessService(self.session)
        rows = await uda.query_entity(
            tenant_id=tenant_id,
            tool_name="sync_product_catalog",
            entity="product",
            filters={},
            mask_pii=True,
            limit=limit,
            caller_subject="sync_worker",
        )
        enqueued = 0
        for row in rows:
            pid = str(row.get("product_id") or row.get("sku") or row.get("name") or "")
            if not pid:
                continue
            state_res = await self.session.execute(
                select(SyncRecordState).where(
                    SyncRecordState.tenant_id == tenant_id,
                    SyncRecordState.entity_type == "product_catalog",
                    SyncRecordState.record_id == pid,
                )
            )
            state = state_res.scalar_one_or_none()
            next_ver = (state.version + 1) if state else 1
            title = str(row.get("name") or pid)
            preview = _format_sync_document_text("product_catalog", pid, title, row)
            c_hash = hashlib.sha256(preview.encode("utf-8")).hexdigest()
            if state and not state.is_deleted and state.content_hash == c_hash:
                continue
            _, created = await self.enqueue_event(
                tenant_id=tenant_id,
                entity_type="product_catalog",
                record_id=pid,
                event_type="updated" if state else "created",
                version=next_ver,
                title=title,
                payload=row,
                idempotency_key=f"poll:product_catalog:{pid}:{c_hash[:12]}",
            )
            if created:
                enqueued += 1

        processed = await self.process_pending_events(tenant_id)
        return {"polled_rows": len(rows), "enqueued": enqueued, "processed": processed}

    async def reconcile_snapshot(
        self,
        tenant_id: str,
        *,
        entity_type: str,
        records: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Compare an authoritative non-PII source snapshot against synced state and repair drift
        (enqueue upserts for missing/newer versions and tombstones for records absent from source).
        """
        etype = entity_type.strip().lower()
        existing_res = await self.session.execute(
            select(SyncRecordState).where(
                SyncRecordState.tenant_id == tenant_id,
                SyncRecordState.entity_type == etype,
                SyncRecordState.is_deleted == False,  # noqa: E712
            )
        )
        existing_by_id = {s.record_id: s for s in existing_res.scalars().all()}
        source_ids = set()
        upserts = 0
        deletes = 0

        for item in records:
            rid = str(item.get("record_id") or "")
            if not rid:
                continue
            source_ids.add(rid)
            ver = int(item.get("version") or 1)
            cur = existing_by_id.get(rid)
            if cur is None or ver > cur.version:
                _, created = await self.enqueue_event(
                    tenant_id=tenant_id,
                    entity_type=etype,
                    record_id=rid,
                    event_type="updated" if cur else "created",
                    version=ver,
                    title=str(item.get("title") or rid),
                    payload=dict(item.get("payload") or {}),
                    idempotency_key=f"reconcile:{etype}:{rid}:v{ver}",
                )
                if created:
                    upserts += 1

        for rid, cur in existing_by_id.items():
            if rid not in source_ids:
                ver = cur.version + 1
                _, created = await self.enqueue_event(
                    tenant_id=tenant_id,
                    entity_type=etype,
                    record_id=rid,
                    event_type="deleted",
                    version=ver,
                    title=rid,
                    payload={},
                    idempotency_key=f"reconcile_del:{etype}:{rid}:v{ver}",
                )
                if created:
                    deletes += 1

        proc = await self.process_pending_events(tenant_id)
        return {"upserts_enqueued": upserts, "deletes_enqueued": deletes, **proc}

    async def get_status(self, tenant_id: str) -> Dict[str, Any]:
        """Return tenant-scoped sync status metrics and recent DLQ entries."""
        status_rows = (
            await self.session.execute(
                select(SyncEvent.status, func.count(SyncEvent.id))
                .where(SyncEvent.tenant_id == tenant_id)
                .group_by(SyncEvent.status)
            )
        ).all()
        counts = {st: int(cnt) for st, cnt in status_rows}

        active_records = int(
            (
                await self.session.execute(
                    select(func.count(SyncRecordState.id)).where(
                        SyncRecordState.tenant_id == tenant_id,
                        SyncRecordState.is_deleted == False,  # noqa: E712
                    )
                )
            ).scalar()
            or 0
        )
        tombstones = int(
            (
                await self.session.execute(
                    select(func.count(SyncRecordState.id)).where(
                        SyncRecordState.tenant_id == tenant_id,
                        SyncRecordState.is_deleted == True,  # noqa: E712
                    )
                )
            ).scalar()
            or 0
        )
        dlq_res = await self.session.execute(
            select(SyncDeadLetter)
            .where(SyncDeadLetter.tenant_id == tenant_id)
            .order_by(SyncDeadLetter.failed_at.desc())
            .limit(20)
        )
        dlq_items = [
            {
                "id": d.id,
                "entity_type": d.entity_type,
                "record_id": d.record_id,
                "version": d.version,
                "error_message": d.error_message,
                "retry_count": d.retry_count,
                "failed_at": d.failed_at.isoformat() if d.failed_at else None,
            }
            for d in dlq_res.scalars().all()
        ]
        return {
            "tenant_id": tenant_id,
            "events": counts,
            "active_synced_records": active_records,
            "tombstones": tombstones,
            "dead_letters": dlq_items,
        }

