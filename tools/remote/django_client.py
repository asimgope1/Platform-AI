"""
Authenticated HTTP client for executing business tools hosted on the Windows Django service.

Architecture role:
  • FastAPI ai-platform remains the primary AI orchestrator (Gemma + ToolLoopRunner + RAG + IVR).
  • Django exposes the business tool catalog (/api/v1/tools/) and executes read-only
    business queries (/api/v1/tools/execute/) against customer databases.
  • Enforces tenant isolation, customer-level authorization, PII masking, argument
    sanitization, timeouts, bounded retries on transient transport errors only,
    idempotency deduplication, and SQL-execution prohibition.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from data_access.security import (
    enforce_customer_authorization,
    mask_record_pii,
    sanitize_tool_argument,
)
from platform_core.config.settings import Settings, settings as default_settings
from platform_core.errors.exceptions import (
    PermissionDeniedError,
    PlatformError,
    ValidationError,
)
from platform_core.observability.logging import logger
from platform_core.security.tenant_context import TenantContext
from tools.schemas import ToolDefinition, ToolFunctionSchema

_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_FORBIDDEN_TOOL_NAMES = frozenset(
    {
        "execute_sql",
        "run_sql",
        "raw_sql",
        "sql_query",
        "exec_sql",
        "shell",
        "eval",
    }
)
_FORBIDDEN_SQL_PARAM_NAMES = frozenset({"sql", "raw_sql", "sql_statement", "ddl", "statement"})
_TRANSIENT_STATUS_CODES = frozenset({502, 503, 504})


class RemoteToolCatalogError(PlatformError):
    """Raised when the remote tool catalog fails schema/security validation."""

    def __init__(self, message: str):
        super().__init__(message=message, code="invalid_remote_catalog", status_code=502)


class RemoteToolUnavailableError(PlatformError):
    """Raised when the remote Django tool backend cannot be reached or times out."""

    def __init__(self, message: str):
        super().__init__(message=message, code="remote_backend_unavailable", status_code=503)


class RemoteToolExecutionError(PlatformError):
    """Raised when remote Django tool execution returns an execution failure."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message=message, code="remote_tool_failed", status_code=status_code)


def _validate_base_url(base_url: str, *, require_https: bool = False) -> str:
    raw = (base_url or "").strip().rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValidationError(f"Invalid REMOTE_TOOLS_BASE_URL: '{base_url}'.")
    if require_https and parsed.scheme != "https":
        raise PermissionDeniedError(
            f"HTTPS is required for remote business tool service URL in this environment (got '{parsed.scheme}')."
        )
    return raw


def compute_hmac_signature(
    *,
    secret: str,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body_bytes: bytes,
) -> str:
    """Compute canonical HMAC-SHA256 request signature for zero-trust Django verification."""
    body_sha256 = hashlib.sha256(body_bytes or b"").hexdigest()
    canonical = f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_sha256}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()


def validate_remote_tool_spec(item: Dict[str, Any]) -> ToolDefinition:
    """Validate and convert a raw Django tool catalog entry into a local ToolDefinition."""
    if not isinstance(item, dict):
        raise RemoteToolCatalogError("Catalog entry must be a JSON object.")

    name = str(item.get("name") or "").strip()
    if not name or not _TOOL_NAME_RE.match(name):
        raise RemoteToolCatalogError(f"Invalid remote tool name: '{name}'.")
    if name.lower() in _FORBIDDEN_TOOL_NAMES or "sql" in name.lower():
        raise RemoteToolCatalogError(f"Forbidden remote tool name '{name}' (arbitrary SQL tools prohibited).")

    description = str(item.get("description") or "").strip()
    if not description:
        raise RemoteToolCatalogError(f"Remote tool '{name}' is missing a non-empty description.")

    parameters = item.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise RemoteToolCatalogError(f"Remote tool '{name}' must define a JSON object parameter schema.")

    props = parameters.get("properties")
    if props is None:
        parameters = {**parameters, "properties": {}}
        props = {}
    if not isinstance(props, dict):
        raise RemoteToolCatalogError(f"Remote tool '{name}' has invalid 'properties' schema.")

    for prop_name in props.keys():
        if str(prop_name).strip().lower() in _FORBIDDEN_SQL_PARAM_NAMES:
            raise RemoteToolCatalogError(
                f"Remote tool '{name}' exposes forbidden SQL parameter '{prop_name}'."
            )

    required = parameters.get("required") or []
    if not isinstance(required, list):
        raise RemoteToolCatalogError(f"Remote tool '{name}' has invalid 'required' list.")

    return ToolDefinition(
        type="function",
        function=ToolFunctionSchema(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": dict(props),
                "required": [str(r) for r in required],
            },
        ),
        required_scope="tools:execute",
    )


_COMPLAINT_SCOPED_TOOLS = frozenset(
    {
        "create_complaint",
        "search_existing_complaints",
        "get_complaint_status",
        "get_customer_complaints",
        "escalate_complaint",
        "add_complaint_note",
        "update_complaint",
    }
)


def _canonical_args_hash(arguments: Dict[str, Any]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _mask_and_authorize_payload(
    data: Any,
    *,
    authorized_customer_id: Optional[str],
    mask_pii: bool = True,
    inherited_customer_id: Optional[str] = None,
) -> Any:
    """Recursively enforce customer ownership & PII masking on remote tool payloads."""
    if isinstance(data, dict):
        rec_owner = (
            data.get("customer_id")
            or data.get("customer_reference")
            or inherited_customer_id
        )
        if authorized_customer_id:
            # Allow empty collection envelopes (count=0, empty lists) or explicit error responses without false 403
            is_empty_or_error_envelope = (
                isinstance(data.get("error"), str)
                or (
                    data.get("count") == 0
                    and all(isinstance(v, list) and len(v) == 0 for v in data.values() if isinstance(v, list))
                )
            )
            if not is_empty_or_error_envelope or rec_owner is not None:
                auth_rec = dict(data)
                if rec_owner is not None and not auth_rec.get("customer_id"):
                    auth_rec["customer_id"] = rec_owner
                enforce_customer_authorization([auth_rec], authorized_customer_id=authorized_customer_id)

        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, list):
                cleaned[k] = [
                    _mask_and_authorize_payload(
                        elem,
                        authorized_customer_id=authorized_customer_id,
                        mask_pii=mask_pii,
                        inherited_customer_id=str(rec_owner) if rec_owner is not None else None,
                    )
                    for elem in v
                ]
            elif isinstance(v, dict):
                cleaned[k] = _mask_and_authorize_payload(
                    v,
                    authorized_customer_id=authorized_customer_id,
                    mask_pii=mask_pii,
                    inherited_customer_id=str(rec_owner) if rec_owner is not None else None,
                )
            else:
                cleaned[k] = v
        return mask_record_pii(cleaned) if mask_pii else cleaned
    if isinstance(data, list):
        return [
            _mask_and_authorize_payload(
                elem,
                authorized_customer_id=authorized_customer_id,
                mask_pii=mask_pii,
                inherited_customer_id=inherited_customer_id,
            )
            for elem in data
        ]
    return data


@dataclass
class _DedupEntry:
    expires_at: float
    result: Dict[str, Any]


class DjangoRemoteToolClient:
    """Async HTTP client bridging FastAPI tool orchestration to the Django tool service."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        hmac_secret: Optional[str] = None,
        require_auth: bool = False,
        require_https: bool = False,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
        per_tenant_concurrency: int = 8,
        rate_limit_per_minute: int = 120,
        tenant_map: Optional[Dict[str, str]] = None,
        default_tenant_slug: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        dedup_ttl_seconds: float = 60.0,
    ):
        self.require_https = bool(require_https)
        self.base_url = _validate_base_url(base_url, require_https=self.require_https)
        self.auth_token = (auth_token or "").strip() or None
        self.api_key = (api_key or "").strip() or None
        self.hmac_secret = (hmac_secret or "").strip() or None
        self.require_auth = bool(require_auth)
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self.per_tenant_concurrency = max(1, int(per_tenant_concurrency))
        self.rate_limit_per_minute = max(1, int(rate_limit_per_minute))
        self.tenant_map: Dict[str, str] = dict(tenant_map or {})
        self.default_tenant_slug = (default_tenant_slug or "").strip() or None
        self._injected_client = http_client
        self._dedup_ttl_seconds = max(1.0, float(dedup_ttl_seconds))
        self._dedup_cache: Dict[str, _DedupEntry] = {}
        self._inflight_locks: Dict[str, asyncio.Lock] = {}
        self._tenant_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._tenant_rate_windows: Dict[str, List[float]] = defaultdict(list)

    @classmethod
    def from_settings(cls, cfg: Settings = default_settings) -> "DjangoRemoteToolClient":
        if not cfg.REMOTE_TOOLS_BASE_URL:
            raise ValidationError("REMOTE_TOOLS_BASE_URL is not configured.")
        raw_map: Dict[str, str] = {}
        if cfg.REMOTE_TOOLS_TENANT_MAP:
            try:
                parsed = json.loads(cfg.REMOTE_TOOLS_TENANT_MAP)
                if isinstance(parsed, dict):
                    raw_map = {str(k): str(v) for k, v in parsed.items()}
            except ValueError as exc:
                raise ValidationError(f"Invalid REMOTE_TOOLS_TENANT_MAP JSON: {exc}") from exc
        is_prod = str(cfg.ENVIRONMENT or "").strip().lower() == "production"
        return cls(
            base_url=cfg.REMOTE_TOOLS_BASE_URL,
            auth_token=cfg.REMOTE_TOOLS_AUTH_TOKEN,
            api_key=cfg.REMOTE_TOOLS_API_KEY,
            hmac_secret=cfg.REMOTE_TOOLS_HMAC_SECRET,
            require_auth=bool(cfg.REMOTE_TOOLS_REQUIRE_AUTH or is_prod),
            require_https=bool(is_prod and cfg.REMOTE_TOOLS_REQUIRE_HTTPS_IN_PROD),
            timeout_seconds=cfg.REMOTE_TOOLS_TIMEOUT_SECONDS,
            max_retries=cfg.REMOTE_TOOLS_MAX_RETRIES,
            per_tenant_concurrency=cfg.REMOTE_TOOLS_PER_TENANT_CONCURRENCY,
            rate_limit_per_minute=cfg.REMOTE_TOOLS_RATE_LIMIT_PER_MIN,
            tenant_map=raw_map,
            default_tenant_slug=cfg.REMOTE_TOOLS_DEFAULT_TENANT_SLUG,
        )

    def _build_headers(
        self,
        *,
        method: str = "GET",
        path: str = "/api/v1/tools/",
        json_body: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        authorized_customer_id: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Optional[bytes]]:
        if self.require_auth and not (self.auth_token or self.api_key or self.hmac_secret):
            raise PermissionDeniedError(
                "Remote Django tool authentication required: configure REMOTE_TOOLS_AUTH_TOKEN, "
                "REMOTE_TOOLS_API_KEY, or REMOTE_TOOLS_HMAC_SECRET."
            )
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        if authorized_customer_id:
            headers["X-Customer-Reference"] = authorized_customer_id
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
            headers["X-Idempotency-Key"] = idempotency_key

        body_bytes: Optional[bytes] = None
        if json_body is not None:
            body_bytes = json.dumps(json_body, separators=(",", ":"), sort_keys=True).encode("utf-8")

        if self.hmac_secret:
            ts = str(int(time.time()))
            nonce = secrets.token_urlsafe(12)
            sig = compute_hmac_signature(
                secret=self.hmac_secret,
                method=method,
                path=path,
                timestamp=ts,
                nonce=nonce,
                body_bytes=body_bytes or b"",
            )
            headers["X-Timestamp"] = ts
            headers["X-Nonce"] = nonce
            headers["X-Signature"] = sig

        return headers, body_bytes

    def _enforce_tenant_rate_limit(self, tenant_id: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        timestamps = [t for t in self._tenant_rate_windows[tenant_id] if t > cutoff]
        if len(timestamps) >= self.rate_limit_per_minute:
            raise PlatformError(
                message=f"Remote tool rate limit exceeded for tenant '{tenant_id}'.",
                code="remote_rate_limit_exceeded",
                status_code=429,
            )
        timestamps.append(now)
        self._tenant_rate_windows[tenant_id] = timestamps

    def _get_tenant_semaphore(self, tenant_id: str) -> asyncio.Semaphore:
        sem = self._tenant_semaphores.get(tenant_id)
        if sem is None:
            sem = asyncio.Semaphore(self.per_tenant_concurrency)
            self._tenant_semaphores[tenant_id] = sem
        return sem

    def resolve_remote_tenant_id(self, context: Optional[TenantContext]) -> str:
        """Resolve trusted TenantContext into the target Django tenant slug/ID (never from LLM args)."""
        if context is None or not context.tenant_id:
            raise PermissionDeniedError("Authenticated TenantContext is required for remote business tool calls.")
        tid = context.tenant_id.strip()
        if tid in self.tenant_map:
            return self.tenant_map[tid]
        if "*" in self.tenant_map:
            return self.tenant_map["*"]
        if self.default_tenant_slug:
            return self.default_tenant_slug
        return tid

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        headers: Dict[str, str],
        body_bytes: Optional[bytes] = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        last_response: Optional[httpx.Response] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self._injected_client is not None:
                    resp = await self._injected_client.request(
                        method, url, headers=headers, content=body_bytes, timeout=self.timeout_seconds
                    )
                else:
                    timeout_cfg = httpx.Timeout(self.timeout_seconds, connect=min(3.0, self.timeout_seconds))
                    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                        resp = await client.request(method, url, headers=headers, content=body_bytes)

                if resp.status_code in _TRANSIENT_STATUS_CODES and attempt < self.max_retries:
                    last_response = resp
                    await asyncio.sleep(0.15 * (2**attempt))
                    continue
                return resp

            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.15 * (2**attempt))
                    continue
                raise RemoteToolUnavailableError(
                    f"Windows Django tool backend unavailable at {self.base_url}: {exc}"
                ) from exc

        if last_response is not None:
            raise RemoteToolUnavailableError(
                f"Windows Django tool backend returned transient HTTP {last_response.status_code}."
            )
        raise RemoteToolUnavailableError(f"Windows Django tool backend request failed: {last_exc}")

    def _evict_expired_dedup(self) -> None:
        now = time.monotonic()
        expired = [k for k, entry in self._dedup_cache.items() if entry.expires_at <= now]
        for k in expired:
            self._dedup_cache.pop(k, None)

    async def fetch_and_validate_catalog(self) -> List[ToolDefinition]:
        """Fetch /api/v1/tools/ from Windows Django and validate every tool schema."""
        headers, body_bytes = self._build_headers(method="GET", path="/api/v1/tools/")
        resp = await self._request_with_retry("GET", "/api/v1/tools/", headers=headers, body_bytes=body_bytes)

        if resp.status_code in (401, 403):
            raise PermissionDeniedError(
                f"Remote Django tool catalog rejected credentials (HTTP {resp.status_code})."
            )
        if resp.status_code >= 500:
            raise RemoteToolUnavailableError(
                f"Remote Django tool catalog unavailable (HTTP {resp.status_code})."
            )
        if resp.status_code >= 400:
            raise RemoteToolCatalogError(
                f"Remote Django tool catalog error (HTTP {resp.status_code}): {resp.text[:300]}"
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise RemoteToolCatalogError("Remote Django tool catalog returned invalid JSON.") from exc

        raw_tools = body.get("tools") if isinstance(body, dict) else body
        if not isinstance(raw_tools, list):
            raise RemoteToolCatalogError("Remote Django tool catalog missing 'tools' list.")

        validated: List[ToolDefinition] = []
        seen_names = set()
        for entry in raw_tools:
            defn = validate_remote_tool_spec(entry)
            if defn.function.name in seen_names:
                raise RemoteToolCatalogError(f"Duplicate remote tool name '{defn.function.name}'.")
            seen_names.add(defn.function.name)
            validated.append(defn)

        return validated

    @staticmethod
    def _sanitize_arguments(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        long_text_keys = {
            "description",
            "subject",
            "note",
            "notes",
            "reason",
            "desired_resolution",
            "product_reference",
            "query",
            "keyword",
            "summary",
        }
        clean_args: Dict[str, Any] = {}
        for k, v in (arguments or {}).items():
            key = str(k).strip()
            if key.lower() in _FORBIDDEN_SQL_PARAM_NAMES:
                raise PermissionDeniedError(f"Forbidden SQL parameter '{key}' rejected.")
            if v is None:
                continue
            if isinstance(v, str):
                max_allowed = 1000 if key in long_text_keys else 128
                clean_args[key] = sanitize_tool_argument(v, key, required=False, max_len=max_allowed)
            elif isinstance(v, (int, float, bool)):
                clean_args[key] = v
            else:
                raise ValidationError(f"Unsupported argument type for '{key}' on tool '{tool_name}'.", param=key)

        # Normalize order_id -> order_number for legacy Django order tools unless complaint order_reference is used
        if "order_id" in clean_args and "order_number" not in clean_args and "order_reference" not in clean_args:
            if tool_name in ("create_complaint", "search_existing_complaints"):
                clean_args["order_reference"] = clean_args.pop("order_id")
            else:
                clean_args["order_number"] = clean_args.pop("order_id")
        return clean_args

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        context: Optional[TenantContext],
        authorized_customer_id: Optional[str] = None,
        call_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        mask_pii: bool = True,
    ) -> Dict[str, Any]:
        """Execute a validated business tool remotely via Django /api/v1/tools/execute/."""
        if not _TOOL_NAME_RE.match(tool_name) or "sql" in tool_name.lower():
            raise PermissionDeniedError(f"Tool '{tool_name}' is not permitted for remote execution.")

        remote_tenant_id = self.resolve_remote_tenant_id(context)
        clean_args = self._sanitize_arguments(tool_name, arguments)

        # Enforce caller customer ownership BEFORE hitting remote DB when customer_reference/customer_id is passed
        if authorized_customer_id:
            for ref_key in ("customer_reference", "customer_id"):
                if ref_key in clean_args and str(clean_args[ref_key]).strip() != str(authorized_customer_id).strip():
                    raise PermissionDeniedError(
                        f"Unauthorized customer access: caller '{authorized_customer_id}' cannot query "
                        f"'{clean_args[ref_key]}'."
                    )
            if tool_name in _COMPLAINT_SCOPED_TOOLS:
                clean_args.setdefault("customer_reference", str(authorized_customer_id).strip())

        # Idempotency / duplicate prevention on retries (coalesces concurrent calls with same key)
        args_sig = _canonical_args_hash(clean_args)
        stable_key = idempotency_key or (f"{remote_tenant_id}:{call_id}:{args_sig}" if call_id else None)
        if tool_name == "create_complaint" and stable_key:
            clean_args.setdefault("idempotency_key", stable_key)

        async def _dispatch_once() -> Dict[str, Any]:
            if stable_key:
                self._evict_expired_dedup()
                cached = self._dedup_cache.get(stable_key)
                if cached is not None:
                    logger.info(
                        f"[RemoteTool] Deduplicated repeated execution for '{tool_name}' (key={stable_key})."
                    )
                    return dict(cached.result)

            request_idem_header = stable_key or f"{remote_tenant_id}:{tool_name}:{args_sig}:{uuid.uuid4().hex[:12]}"
            payload = {
                "tenant_id": remote_tenant_id,
                "tool_name": tool_name,
                "arguments": clean_args,
            }
            headers, body_bytes = self._build_headers(
                method="POST",
                path="/api/v1/tools/execute/",
                json_body=payload,
                tenant_id=remote_tenant_id,
                idempotency_key=request_idem_header,
                authorized_customer_id=authorized_customer_id,
            )

            self._enforce_tenant_rate_limit(remote_tenant_id)
            async with self._get_tenant_semaphore(remote_tenant_id):
                resp = await self._request_with_retry(
                    "POST",
                    "/api/v1/tools/execute/",
                    headers=headers,
                    body_bytes=body_bytes,
                )

            if resp.status_code in (401, 403):
                raise PermissionDeniedError(
                    f"Remote tool service denied access for tenant '{remote_tenant_id}' (HTTP {resp.status_code})."
                )
            if resp.status_code == 404:
                detail = ""
                try:
                    detail = str(resp.json().get("detail") or resp.text)
                except Exception:
                    detail = resp.text[:200]
                raise ValidationError(f"Remote tenant or resource not found: {detail}")
            if resp.status_code == 400 or resp.status_code == 422:
                raise ValidationError(f"Remote tool rejected arguments (HTTP {resp.status_code}): {resp.text[:300]}")
            if resp.status_code >= 500:
                raise RemoteToolUnavailableError(
                    f"Remote Django tool service failed (HTTP {resp.status_code})."
                )

            try:
                body = resp.json()
            except ValueError as exc:
                raise RemoteToolExecutionError("Remote tool response was not valid JSON.") from exc

            if not isinstance(body, dict):
                raise RemoteToolExecutionError("Remote tool response must be a JSON object.")

            status_str = str(body.get("status") or "").lower()
            if status_str == "error":
                err_msg = str(body.get("error") or body.get("detail") or "Remote tool reported error.")
                raise RemoteToolExecutionError(err_msg, status_code=400)

            raw_data = body.get("data", body)
            scoped_owner = (
                str(authorized_customer_id).strip()
                if (authorized_customer_id and tool_name in _COMPLAINT_SCOPED_TOOLS and clean_args.get("customer_reference") == str(authorized_customer_id).strip())
                else None
            )
            protected_data = _mask_and_authorize_payload(
                raw_data,
                authorized_customer_id=authorized_customer_id,
                mask_pii=mask_pii,
                inherited_customer_id=scoped_owner,
            )

            result: Dict[str, Any] = {
                "tool_name": body.get("tool_name", tool_name),
                "execution_status": status_str or "success",
                "tenant_id": remote_tenant_id,
                "source": "remote_django",
                "execution_time_ms": body.get("execution_time_ms"),
                "data": protected_data,
            }
            if isinstance(protected_data, dict):
                for k, v in protected_data.items():
                    if k == "status" or k not in result:
                        result[k] = v

            if stable_key:
                self._dedup_cache[stable_key] = _DedupEntry(
                    expires_at=time.monotonic() + self._dedup_ttl_seconds,
                    result=dict(result),
                )
            return result

        if stable_key:
            lock = self._inflight_locks.setdefault(stable_key, asyncio.Lock())
            async with lock:
                return await _dispatch_once()
        return await _dispatch_once()
