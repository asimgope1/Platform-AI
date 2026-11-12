"""
Reference Server-Side Authentication & Zero-Trust Verification Middleware for Windows Django.

NOTE: The Windows Django Business Service repository (http://192.168.1.141:8000) resides on a
separate Windows host outside the MacBook workspace. This module provides the exact drop-in
authentication, tenant binding, timestamp/nonce replay protection, and HMAC-SHA256 verification
contract matching MacBook `DjangoRemoteToolClient` so it can be applied directly on Windows.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Dict, Optional, Set, Tuple

from tools.remote.django_client import compute_hmac_signature


class ReplayNonceStore:
    """Bounded in-memory replay protection store (replace with Redis SETNX in multi-worker prod)."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 50000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._seen: Dict[str, float] = {}

    def check_and_record(self, nonce: str, now_ts: Optional[float] = None) -> bool:
        if not nonce or len(nonce) < 8:
            return False
        now = now_ts if now_ts is not None else time.time()
        if len(self._seen) >= self.max_entries:
            cutoff = now - self.ttl_seconds
            expired = [k for k, ts in self._seen.items() if ts <= cutoff]
            for k in expired:
                self._seen.pop(k, None)
        existing = self._seen.get(nonce)
        if existing is not None and (now - existing) <= self.ttl_seconds:
            return False
        self._seen[nonce] = now
        return True


def verify_service_request(
    *,
    method: str,
    path: str,
    headers: Dict[str, str],
    body_bytes: bytes,
    expected_api_keys: Set[str],
    hmac_secret: Optional[str],
    nonce_store: Optional[ReplayNonceStore] = None,
    max_clock_skew_seconds: int = 300,
    claimed_body_tenant_id: Optional[str] = None,
) -> Tuple[bool, int, str]:
    """
    Fail-closed verifier matching Django `/api/v1/tools/`, `/api/v1/tools/execute/`, and `/api/v1/tenants/`.

    Returns `(ok, http_status, reason)`.
    """
    norm_headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    auth_header = norm_headers.get("authorization", "")
    bearer_key = auth_header[len("Bearer ") :].strip() if auth_header.startswith("Bearer ") else ""
    api_key = norm_headers.get("x-api-key", "").strip() or bearer_key

    if not expected_api_keys and not hmac_secret:
        return False, 503, "Service authentication keys are not configured (fail-closed)."

    if expected_api_keys:
        if not api_key or not any(hmac.compare_digest(api_key, valid) for valid in expected_api_keys):
            return False, 401, "Missing or invalid service API key / Bearer token."

    if hmac_secret:
        ts_raw = norm_headers.get("x-timestamp", "").strip()
        nonce = norm_headers.get("x-nonce", "").strip()
        sig = norm_headers.get("x-signature", "").strip()
        if not ts_raw or not nonce or not sig:
            return False, 401, "Missing HMAC request signature headers (X-Timestamp, X-Nonce, X-Signature)."
        try:
            ts_int = int(ts_raw)
        except ValueError:
            return False, 401, "Invalid X-Timestamp header."
        now_ts = time.time()
        if abs(now_ts - ts_int) > max_clock_skew_seconds:
            return False, 401, "Stale or future X-Timestamp outside allowed replay window."
        if nonce_store is not None and not nonce_store.check_and_record(nonce, now_ts=now_ts):
            return False, 401, "Duplicate X-Nonce detected (replay blocked)."

        expected_sig = compute_hmac_signature(
            secret=hmac_secret,
            method=method,
            path=path,
            timestamp=ts_raw,
            nonce=nonce,
            body_bytes=body_bytes,
        )
        if not hmac.compare_digest(sig, expected_sig):
            return False, 401, "Invalid HMAC request signature."

    header_tenant = norm_headers.get("x-tenant-id", "").strip()
    if claimed_body_tenant_id and header_tenant and claimed_body_tenant_id.strip() != header_tenant:
        return False, 403, "Tenant header mismatch against payload tenant_id."

    return True, 200, "ok"


class DjangoServiceAuthMiddleware:
    """
    Drop-in Django WSGI/ASGI middleware protecting `/api/v1/tools/*` and `/api/v1/tenants/*`.
    Copy this file into the Windows Django project and add it to `settings.MIDDLEWARE`.
    """

    PROTECTED_PREFIXES = ("/api/v1/tools/", "/api/v1/tenants/", "/api/v1/knowledge/", "/api/v1/ai/")

    def __init__(self, get_response):
        self.get_response = get_response
        self.nonce_store = ReplayNonceStore()
        raw_keys = os.getenv("DJANGO_INTERNAL_API_KEYS", os.getenv("REMOTE_TOOLS_API_KEY", ""))
        self.expected_api_keys = {k.strip() for k in raw_keys.split(",") if k.strip()}
        self.hmac_secret = os.getenv("REMOTE_TOOLS_HMAC_SECRET", "").strip() or None

    def __call__(self, request):
        path = getattr(request, "path", "/")
        if any(path.startswith(p) for p in self.PROTECTED_PREFIXES):
            from django.http import JsonResponse

            ok, status_code, reason = verify_service_request(
                method=getattr(request, "method", "GET"),
                path=path,
                headers=dict(getattr(request, "headers", {})),
                body_bytes=getattr(request, "body", b"") or b"",
                expected_api_keys=self.expected_api_keys,
                hmac_secret=self.hmac_secret,
                nonce_store=self.nonce_store,
            )
            if not ok:
                return JsonResponse({"error": reason, "code": "authentication_failed"}, status=status_code)
        return self.get_response(request)

