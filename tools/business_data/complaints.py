"""
Complaint Registration & Allowlisted External API Tooling
=======================================================
Provides production-grade conversational tools callable during voice IVR calls
or chat sessions:
  • register_complaint: Registers a persistent customer complaint with unique ticket ID
    and optionally dispatches to an external CRM / Django complaint API.
  • get_complaint_status: Looks up complaint ticket status scoped strictly to tenant.
  • call_external_api: Executes schema-validated, allowlist-guarded HTTP API calls
    against configured backend services (e.g., Windows Django or customer CRM API).
"""
from __future__ import annotations

import hashlib
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from conversation_intelligence.transcript_confidence import mask_sensitive_text
from data_access.security import mask_record_pii
from data_access.service import UniversalDataAccessService
from platform_core.config.settings import settings
from platform_core.errors.exceptions import ToolExecutionError, ValidationError
from platform_core.observability.logging import logger
from platform_core.security.tenant_context import TenantContext
from storage.database import async_session
from storage.models.complaint import Complaint

_VALID_CATEGORIES = {
    "order_delivery",
    "damaged_item",
    "payment_upi",
    "refund_billing",
    "card_account",
    "service_quality",
    "general",
}
_VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
_ALLOWED_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH"}


@asynccontextmanager
async def _session_scope(db_session: Optional[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Reuse injected AsyncSession or open an isolated internal DB session."""
    if db_session is not None:
        yield db_session
        return
    async with async_session() as session:
        yield session


def _resolve_tenant_id(context: Optional[TenantContext]) -> str:
    if not context or not getattr(context, "tenant_id", None):
        raise ToolExecutionError("A verified TenantContext is required to access complaint records.")
    return str(context.tenant_id)


def _infer_category(description: str, requested_category: Optional[str]) -> str:
    if requested_category:
        norm = requested_category.strip().lower().replace(" ", "_").replace("-", "_")
        if norm in _VALID_CATEGORIES:
            return norm
    lower = description.lower()
    if any(k in lower for k in ("damaged", "broken", "defective", "wrong item", "missing item")):
        return "damaged_item"
    if any(k in lower for k in ("upi", "debited", "payment failed", "transaction", "netbanking")):
        return "payment_upi"
    if any(k in lower for k in ("refund", "invoice", "charged twice", "overcharged", "billing")):
        return "refund_billing"
    if any(k in lower for k in ("delay", "late", "not delivered", "order", "shipment", "tracking", "courier")):
        return "order_delivery"
    if any(k in lower for k in ("card", "atm", "pin", "kyc", "account", "block")):
        return "card_account"
    if any(k in lower for k in ("rude", "agent", "service", "support", "wait")):
        return "service_quality"
    return "general"


def _infer_priority(description: str, requested_priority: Optional[str]) -> str:
    if requested_priority:
        norm = requested_priority.strip().lower()
        if norm in _VALID_PRIORITIES:
            return norm
    lower = description.lower()
    if any(k in lower for k in ("fraud", "unauthorized", "scam", "stolen", "urgent", "emergency")):
        return "urgent"
    if any(k in lower for k in ("debited", "charged twice", "damaged", "missing", "failed")):
        return "high"
    return "normal"


def _mask_phone_for_storage(phone: Optional[str]) -> Optional[str]:
    if not phone or phone == "unknown":
        return None
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) <= 4:
        return digits or None
    return f"***{digits[-4:]}"


def _build_ticket_number() -> str:
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))
    return f"CMP-{date_part}-{suffix}"


def _normalize_base_url(raw: Optional[str]) -> Optional[str]:
    if not raw or not raw.strip():
        return None
    parsed = urlparse(raw.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return raw.strip().rstrip("/")


def _collect_allowed_api_bases() -> Dict[str, str]:
    """Collect configured, explicitly allowlisted base URLs mapped by service alias."""
    bases: Dict[str, str] = {}
    remote_base = _normalize_base_url(settings.REMOTE_TOOLS_BASE_URL)
    complaint_url = _normalize_base_url(settings.COMPLAINT_API_URL)
    if remote_base:
        bases["remote_django"] = remote_base
    if complaint_url:
        bases["complaint_api"] = complaint_url
    for idx, entry in enumerate(str(settings.EXTERNAL_API_ALLOWED_BASES or "").split(","), start=1):
        norm = _normalize_base_url(entry)
        if norm:
            bases[f"configured_{idx}"] = norm
    return bases


def _resolve_target_url(service: str, path_or_url: str, allowed_bases: Dict[str, str]) -> tuple[str, str]:
    """
    Validate and resolve an API path or URL strictly against configured allowlisted base URLs.
    Prevents arbitrary SSRF calls to unconfigured domains or cloud metadata IPs.
    """
    if not allowed_bases:
        raise ValidationError(
            "No external API base URLs are configured (set REMOTE_TOOLS_BASE_URL, COMPLAINT_API_URL, or EXTERNAL_API_ALLOWED_BASES)."
        )

    candidate = (path_or_url or "").strip()
    if not candidate:
        raise ValidationError("API path is required.")

    # Full URL must match an allowlisted scheme + netloc + base path prefix
    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in ("http", "https"):
            raise ValidationError("Only http and https URLs are permitted.")
        for svc_key, base in allowed_bases.items():
            bp = urlparse(base)
            if (
                parsed.scheme == bp.scheme
                and parsed.netloc == bp.netloc
                and parsed.path.startswith(bp.path or "/")
            ):
                return svc_key, candidate
        raise ValidationError("Target URL host/prefix is not in the configured external API allowlist.")

    # Relative path resolved against chosen service alias
    if not candidate.startswith("/"):
        candidate = "/" + candidate
    svc = (service or "").strip().lower()
    if svc in allowed_bases:
        chosen_key = svc
    elif "remote_django" in allowed_bases and svc in ("", "django", "configured"):
        chosen_key = "remote_django"
    else:
        chosen_key = next(iter(allowed_bases.keys()))

    base_url = allowed_bases[chosen_key]
    resolved = urljoin(base_url + "/", candidate.lstrip("/"))
    return chosen_key, resolved


def _complaint_to_dict(row: Complaint) -> Dict[str, Any]:
    return {
        "ticket_number": row.ticket_number,
        "status": row.status,
        "category": row.category,
        "priority": row.priority,
        "subject": row.subject,
        "description": row.description,
        "order_id": row.order_id,
        "customer_id": row.customer_id,
        "source_channel": row.source_channel,
        "external_api_status": row.external_api_status,
        "external_reference": row.external_reference,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def _dispatch_external_complaint(
    *,
    complaint: Complaint,
    context: TenantContext,
    http_client: Optional[httpx.AsyncClient] = None,
) -> tuple[str, Optional[str], Dict[str, Any]]:
    """
    Dispatch a registered complaint to an external Complaint API or Remote Django service
    when configured, degrading non-fatally if external API is offline or unconfigured.
    """
    payload = {
        "ticket_number": complaint.ticket_number,
        "tenant_id": complaint.tenant_id,
        "call_uuid": complaint.call_uuid,
        "customer_id": complaint.customer_id,
        "order_id": complaint.order_id,
        "category": complaint.category,
        "priority": complaint.priority,
        "subject": complaint.subject,
        "description": complaint.description,
        "source_channel": complaint.source_channel,
        "created_at": complaint.created_at.isoformat() if complaint.created_at else None,
    }

    complaint_url = _normalize_base_url(settings.COMPLAINT_API_URL)
    if complaint_url:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-Tenant-Id": str(context.tenant_id),
            "Idempotency-Key": f"complaint:{complaint.ticket_number}",
        }
        if settings.COMPLAINT_API_TOKEN:
            headers["Authorization"] = f"Bearer {settings.COMPLAINT_API_TOKEN}"
        try:
            if http_client is not None:
                resp = await http_client.post(complaint_url, json=payload, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=settings.REMOTE_TOOLS_TIMEOUT_SECONDS) as client:
                    resp = await client.post(complaint_url, json=payload, headers=headers)
            if 200 <= resp.status_code < 300:
                body = resp.json() if resp.content else {}
                ext_ref = None
                if isinstance(body, dict):
                    ext_ref = str(
                        body.get("ticket_id")
                        or body.get("reference")
                        or body.get("id")
                        or complaint.ticket_number
                    )
                sanitized = mask_record_pii(body) if isinstance(body, dict) else {"status_code": resp.status_code}
                return "dispatched", ext_ref, {"status_code": resp.status_code, "response": sanitized}
            logger.warning(f"[ComplaintAPI] External complaint endpoint returned HTTP {resp.status_code}.")
            return "remote_error", None, {"status_code": resp.status_code}
        except Exception as exc:
            logger.warning(f"[ComplaintAPI] Failed dispatching complaint to COMPLAINT_API_URL: {exc}")
            return "remote_error", None, {"error": type(exc).__name__}

    if settings.REMOTE_TOOLS_ENABLED and _normalize_base_url(settings.REMOTE_TOOLS_BASE_URL):
        try:
            from tools.remote.django_client import DjangoRemoteToolClient

            client = DjangoRemoteToolClient.from_settings(settings)
            if http_client is not None:
                client._external_client = http_client
            # If the remote Django service exposes a complaint/ticket tool, execute it cleanly
            from tools.registry import tool_registry

            for remote_name in ("create_complaint", "create_support_ticket"):
                defn = tool_registry.get_tool(remote_name)
                if defn and getattr(defn, "source", "") == "remote_django":
                    res = await client.execute_tool(
                        remote_name,
                        payload,
                        context=context,
                        authorized_customer_id=complaint.customer_id,
                        call_id=complaint.call_uuid,
                    )
                    ext_ref = str(
                        res.get("ticket_id") or res.get("id") or res.get("reference") or complaint.ticket_number
                    )
                    return "remote_tool", ext_ref, mask_record_pii(res)
            # Otherwise attempt standard Django complaints endpoint without failing local DB save
            url = _normalize_base_url(settings.REMOTE_TOOLS_BASE_URL) + "/api/v1/complaints/"
            headers, body_bytes = client._build_headers(  # noqa: SLF001
                method="POST",
                path="/api/v1/complaints/",
                json_body=payload,
                tenant_id=client.resolve_remote_tenant_id(context),
                idempotency_key=f"complaint:{complaint.ticket_number}",
                authorized_customer_id=complaint.customer_id,
            )
            if http_client is not None:
                resp = await http_client.post(url, content=body_bytes, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=settings.REMOTE_TOOLS_TIMEOUT_SECONDS) as async_cli:
                    resp = await async_cli.post(url, content=body_bytes, headers=headers)
            if 200 <= resp.status_code < 300:
                body = resp.json() if resp.content else {}
                ext_ref = None
                if isinstance(body, dict):
                    ext_ref = str(body.get("ticket_id") or body.get("id") or complaint.ticket_number)
                return "dispatched", ext_ref, {"status_code": resp.status_code}
            if resp.status_code == 404:
                return "local_only", None, {"note": "Remote /api/v1/complaints/ not exposed on Django host."}
            return "remote_error", None, {"status_code": resp.status_code}
        except Exception as exc:
            logger.warning(f"[ComplaintAPI] Remote Django complaint sync non-fatal error: {exc}")
            return "remote_error", None, {"error": type(exc).__name__}

    return "local_only", None, {}


async def register_complaint(
    *,
    description: str,
    category: Optional[str] = None,
    subject: Optional[str] = None,
    order_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    caller_phone: Optional[str] = None,
    priority: Optional[str] = None,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
    call_uuid: Optional[str] = None,
    caller_number: Optional[str] = None,
    service_category: Optional[str] = None,
    source_channel: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """
    Register a customer complaint from a live call or chat conversation and dispatch to external API if configured.
    """
    tenant_id = _resolve_tenant_id(tenant_context)
    clean_desc = (description or "").strip()
    if len(clean_desc) < 5:
        raise ValidationError("Please provide a brief description of the complaint (at least 5 characters).")

    effective_customer = authorized_customer_id or (customer_id.strip() if customer_id else None)
    effective_category = _infer_category(clean_desc, category or service_category)
    effective_priority = _infer_priority(clean_desc, priority)
    clean_subject = (subject or "").strip()
    if not clean_subject:
        words = clean_desc.split()
        clean_subject = " ".join(words[:10])[:240]
        if len(words) > 10:
            clean_subject += "..."

    masked_caller = _mask_phone_for_storage(caller_phone or caller_number)
    effective_call_uuid = (call_uuid or "").strip() or None
    desc_fingerprint = hashlib.sha256(clean_desc.lower().encode("utf-8")).hexdigest()[:12]

    async with _session_scope(db_session) as session:
        # Idempotency check inside the same call session so retries do not file duplicate tickets
        if effective_call_uuid and effective_call_uuid != "unknown":
            existing_res = await session.execute(
                select(Complaint)
                .where(
                    Complaint.tenant_id == tenant_id,
                    Complaint.call_uuid == effective_call_uuid,
                    Complaint.status == "open",
                )
                .order_by(desc(Complaint.created_at))
                .limit(5)
            )
            for prev in existing_res.scalars().all():
                prev_fp = hashlib.sha256((prev.description or "").strip().lower().encode("utf-8")).hexdigest()[:12]
                if prev_fp == desc_fingerprint:
                    return {
                        "registration_status": "already_registered",
                        "message": f"Complaint is already registered under ticket {prev.ticket_number}.",
                        **_complaint_to_dict(prev),
                    }

        ticket_number = _build_ticket_number()
        row = Complaint(
            ticket_number=ticket_number,
            tenant_id=tenant_id,
            call_uuid=effective_call_uuid,
            caller_number=masked_caller,
            customer_id=effective_customer,
            order_id=order_id.strip() if order_id else None,
            category=effective_category,
            subject=clean_subject,
            description=clean_desc,
            priority=effective_priority,
            status="open",
            source_channel=(source_channel or ("voice_ivr" if effective_call_uuid else "chat_api")),
        )
        session.add(row)
        await session.flush()

        ext_status, ext_ref, ext_payload = await _dispatch_external_complaint(
            complaint=row,
            context=tenant_context,  # type: ignore[arg-type]
            http_client=http_client,
        )
        row.external_api_status = ext_status
        row.external_reference = ext_ref
        row.external_response_json = ext_payload

        await session.commit()
        await session.refresh(row)

    logger.info(
        f"[Complaint] Registered ticket={row.ticket_number} tenant={tenant_id} "
        f"category={row.category} priority={row.priority} ext_status={row.external_api_status} "
        f"summary='{mask_sensitive_text(row.subject)}'"
    )
    return {
        "registration_status": "registered",
        "message": f"Complaint registered successfully with ticket number {row.ticket_number}.",
        **_complaint_to_dict(row),
    }


async def get_complaint_status(
    *,
    ticket_number: Optional[str] = None,
    order_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    limit: int = 5,
    tenant_context: Optional[TenantContext] = None,
    db_session: Optional[AsyncSession] = None,
    authorized_customer_id: Optional[str] = None,
    call_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve status of complaints filed over calls/chat or mapped customer support tickets."""
    tenant_id = _resolve_tenant_id(tenant_context)
    safe_limit = max(1, min(int(limit or 5), 20))

    target_customer = customer_id.strip() if customer_id else None
    if authorized_customer_id and target_customer and str(target_customer) != str(authorized_customer_id):
        raise PermissionError("Access denied: cannot inspect complaints belonging to another customer.")
    effective_customer = authorized_customer_id or target_customer

    async with _session_scope(db_session) as session:
        stmt = select(Complaint).where(Complaint.tenant_id == tenant_id)
        if ticket_number and ticket_number.strip():
            stmt = stmt.where(Complaint.ticket_number == ticket_number.strip().upper())
        elif order_id and order_id.strip():
            stmt = stmt.where(Complaint.order_id == order_id.strip())
        elif effective_customer:
            stmt = stmt.where(Complaint.customer_id == effective_customer)
        elif call_uuid and call_uuid != "unknown":
            stmt = stmt.where(Complaint.call_uuid == call_uuid)
        else:
            raise ValidationError("Provide ticket_number, order_id, or customer_id to look up complaint status.")

        if authorized_customer_id and ticket_number:
            # Allow ticket lookup within tenant if matching customer or unlinked ticket
            pass

        stmt = stmt.order_by(desc(Complaint.created_at)).limit(safe_limit)
        res = await session.execute(stmt)
        rows = res.scalars().all()
        if rows:
            if authorized_customer_id and ticket_number:
                for r in rows:
                    if r.customer_id and str(r.customer_id) != str(authorized_customer_id):
                        raise PermissionError("Access denied for this complaint ticket.")
            return {
                "found": True,
                "count": len(rows),
                "complaints": [_complaint_to_dict(r) for r in rows],
            }

        # Fallback: check mapped external support_ticket entity if tenant configured an external DB
        try:
            filters: Dict[str, Any] = {}
            if ticket_number:
                filters["ticket_id"] = ticket_number.strip()
            elif effective_customer:
                filters["customer_id"] = effective_customer
            if filters:
                uda = UniversalDataAccessService(session)
                uda_rows = await uda.query_entity(
                    tenant_id=tenant_id,
                    tool_name="get_complaint_status",
                    entity="support_ticket",
                    filters=filters,
                    include_related_customer=True,
                    order_by="updated_at",
                    descending=True,
                    limit=safe_limit,
                    authorized_customer_id=authorized_customer_id,
                    mask_pii=True,
                    caller_subject=getattr(tenant_context, "api_key_id", "api"),
                )
                formatted = []
                for rec in uda_rows:
                    updated_val = rec.get("updated_at") or rec.get("created_at")
                    formatted.append(
                        {
                            "ticket_number": rec.get("ticket_id"),
                            "status": rec.get("status_name") or rec.get("phase_type") or str(rec.get("ticket_status") or ""),
                            "status_id": str(rec.get("ticket_status")) if rec.get("ticket_status") is not None else None,
                            "description": rec.get("subject"),
                            "updated_at": updated_val.isoformat() if hasattr(updated_val, "isoformat") else updated_val,
                            "customer_id": str(rec.get("customer_id")) if rec.get("customer_id") is not None else None,
                            "customer_name": rec.get("customer_name"),
                            "phone": rec.get("phone"),
                        }
                    )
                if formatted:
                    return {
                        "found": True,
                        "count": len(formatted),
                        "complaints": formatted,
                    }
        except PermissionError:
            raise
        except Exception as exc:
            logger.debug(f"[get_complaint_status] UDA fallback skipped/failed: {exc}")

    return {
        "found": False,
        "count": 0,
        "complaints": [],
        "message": "No matching complaint ticket found.",
    }


async def call_external_api(
    *,
    path: str,
    method: str = "GET",
    service: str = "remote_django",
    payload: Optional[Dict[str, Any]] = None,
    query_params: Optional[Dict[str, Any]] = None,
    tenant_context: Optional[TenantContext] = None,
    call_uuid: Optional[str] = None,
    authorized_customer_id: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """
    Execute an allowlisted HTTP API request against a configured backend service during conversation.
    Strictly validates method and target URL against REMOTE_TOOLS_BASE_URL, COMPLAINT_API_URL,
    or EXTERNAL_API_ALLOWED_BASES to prevent SSRF.
    """
    tenant_id = _resolve_tenant_id(tenant_context)
    upper_method = (method or "GET").strip().upper()
    if upper_method not in _ALLOWED_HTTP_METHODS:
        raise ValidationError(f"Unsupported HTTP method '{method}'. Allowed: {sorted(_ALLOWED_HTTP_METHODS)}")

    allowed_bases = _collect_allowed_api_bases()
    svc_key, target_url = _resolve_target_url(service, path, allowed_bases)
    parsed = urlparse(target_url)

    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Tenant-Id": tenant_id,
    }
    if call_uuid and call_uuid != "unknown":
        headers["X-Call-UUID"] = str(call_uuid)

    body_payload = dict(payload or {}) if isinstance(payload, dict) else {}
    if authorized_customer_id and "customer_id" not in body_payload and upper_method in ("POST", "PUT", "PATCH"):
        body_payload["customer_id"] = authorized_customer_id
    json_body_arg = body_payload if upper_method in ("POST", "PUT", "PATCH") else None
    signed_body_bytes: Optional[bytes] = None

    if svc_key == "remote_django":
        from tools.remote.django_client import DjangoRemoteToolClient

        client = DjangoRemoteToolClient.from_settings(settings)
        django_headers, signed_body_bytes = client._build_headers(  # noqa: SLF001
            method=upper_method,
            path=parsed.path,
            json_body=json_body_arg,
            tenant_id=client.resolve_remote_tenant_id(tenant_context),
            idempotency_key=f"extapi:{tenant_id}:{call_uuid or 'chat'}:{parsed.path}",
            authorized_customer_id=authorized_customer_id,
        )
        headers.update(django_headers)
    elif svc_key == "complaint_api" and settings.COMPLAINT_API_TOKEN:
        headers["Authorization"] = f"Bearer {settings.COMPLAINT_API_TOKEN}"
    elif settings.REMOTE_TOOLS_API_KEY:
        headers["X-API-Key"] = settings.REMOTE_TOOLS_API_KEY

    try:
        req_kwargs: Dict[str, Any] = {
            "params": query_params if isinstance(query_params, dict) else None,
            "headers": headers,
        }
        if signed_body_bytes is not None:
            req_kwargs["content"] = signed_body_bytes
        elif json_body_arg is not None:
            req_kwargs["json"] = json_body_arg

        if http_client is not None:
            resp = await http_client.request(upper_method, target_url, **req_kwargs)
        else:
            async with httpx.AsyncClient(timeout=settings.REMOTE_TOOLS_TIMEOUT_SECONDS) as async_cli:
                resp = await async_cli.request(upper_method, target_url, **req_kwargs)

        try:
            data = resp.json()
        except Exception:
            data = {"text": (resp.text or "")[:500]}

        sanitized_data: Any = data
        if isinstance(data, dict):
            sanitized_data = mask_record_pii(data)
        elif isinstance(data, list):
            sanitized_data = [mask_record_pii(item) if isinstance(item, dict) else item for item in data[:25]]

        logger.info(
            f"[ExternalAPI] {upper_method} service={svc_key} path={parsed.path} status={resp.status_code} tenant={tenant_id}"
        )
        return {
            "ok": 200 <= resp.status_code < 300,
            "status_code": resp.status_code,
            "service": svc_key,
            "path": parsed.path,
            "data": sanitized_data,
        }
    except Exception as exc:
        logger.warning(f"[ExternalAPI] Request failed for service={svc_key} path={parsed.path}: {exc}")
        return {
            "ok": False,
            "status_code": 0,
            "service": svc_key,
            "path": parsed.path,
            "error": f"External API request failed ({type(exc).__name__}).",
        }
