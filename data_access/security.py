"""
Security primitives for the Universal Data Access Layer:
  • Authenticated symmetric encryption (Fernet) + key rotation for DB credentials
  • Strict SQL identifier validation and read-only SQL enforcement
  • Tool argument SQL-injection / prompt-injection validation
  • PII masking & customer-level authorization checks
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from platform_core.config.settings import settings
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|grant|revoke|exec|execute|call|merge|attach|detach|pragma|copy)\b",
    re.IGNORECASE,
)
_SQL_COMMENT_OR_STACK_RE = re.compile(r"(;|--|/\*|\*/|\bxp_cmdshell\b|\bunion\s+select\b)", re.IGNORECASE)
_INJECTION_ARG_RE = re.compile(
    r"(;|--|/\*|\*/|'\s*or\s+['\d]|\"\s*or\s+[\"\d]|\bunion\s+all\s+select\b|\bunion\s+select\b|\bdrop\s+table\b|\bsleep\s*\(|\bbenchmark\s*\(|\bignore\s+previous\s+instructions\b|\bsystem\s*prompt\b)",
    re.IGNORECASE,
)

DEFAULT_PII_FIELDS = frozenset(
    {
        "phone",
        "mobile_number",
        "email",
        "email_address",
        "address",
        "addresses",
        "shipping_address",
        "billing_address",
        "ssn",
        "pan",
        "aadhaar",
        "tax_id",
        "national_id",
        "card_number",
        "cvv",
        "cvc",
        "account_number",
        "payment_details",
        "password",
        "secret",
        "token",
        "encrypted_credentials",
    }
)


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a valid 32-byte URL-safe base64 Fernet key from an application secret."""
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


class CredentialCipher:
    """
    Encrypts and decrypts external database credentials at rest.
    Supports key rotation via comma-separated DATA_ACCESS_ROTATION_KEYS.
    Never logs or exposes plaintext secrets.
    """

    def __init__(self, primary_secret: Optional[str] = None, fallback_secrets: Optional[Iterable[str]] = None):
        env_primary = (
            primary_secret
            or os.getenv("DATA_ACCESS_MASTER_KEY")
            or os.getenv("TENANT_DB_MASTER_KEY")
            or settings.MASTER_API_KEY
            or os.getenv("DJANGO_SECRET_KEY")
            or "ai-platform-dev-data-access-key-rotate-in-prod"
        )
        keys = [_derive_fernet_key(env_primary)]

        extra_env = os.getenv("DATA_ACCESS_ROTATION_KEYS", "")
        for s in list(fallback_secrets or []) + [x.strip() for x in extra_env.split(",") if x.strip()]:
            k = _derive_fernet_key(s)
            if k not in keys:
                keys.append(k)

        self._fernet = MultiFernet([Fernet(k) for k in keys])

    def encrypt_credentials(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self._fernet.encrypt(raw).decode("ascii")

    def decrypt_credentials(self, token: str) -> Dict[str, Any]:
        if not token:
            return {}
        try:
            plaintext = self._fernet.decrypt(token.encode("ascii"))
            data = json.loads(plaintext.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (InvalidToken, ValueError, TypeError) as exc:
            raise ValidationError("Failed to decrypt database credentials reference.") from exc

    def rotate_token(self, token: str) -> str:
        """Re-encrypts an existing credential token using the active primary key."""
        payload = self.decrypt_credentials(token)
        return self.encrypt_credentials(payload)


def validate_identifier(name: Optional[str], field_label: str = "identifier") -> str:
    """Validate SQL schema/table/column identifiers against a strict allowlist regex."""
    if not name or not isinstance(name, str) or not _IDENTIFIER_RE.match(name.strip()):
        raise ValidationError(f"Invalid SQL {field_label}: '{name}'. Only alphanumeric and underscore characters are allowed.")
    return name.strip()


def validate_read_only_sql(sql: str) -> str:
    """Verify a SQL statement is strictly a single read-only SELECT query."""
    cleaned = (sql or "").strip()
    if not cleaned:
        raise ValidationError("SQL query cannot be empty.")
    if not cleaned.lower().startswith("select"):
        raise PermissionDeniedError("Read-only policy violation: only SELECT statements are permitted.")
    if _SQL_COMMENT_OR_STACK_RE.search(cleaned):
        raise PermissionDeniedError("Read-only policy violation: stacked queries, SQL comments, and UNION injection patterns are blocked.")
    if _FORBIDDEN_SQL_RE.search(cleaned):
        raise PermissionDeniedError("Read-only policy violation: write, DDL, or privilege statements are prohibited.")
    return cleaned


def sanitize_tool_argument(
    value: Any,
    param_name: str,
    *,
    required: bool = True,
    max_len: int = 128,
) -> Optional[str]:
    """Validate and sanitize string arguments passed from LLM tool calls before database access."""
    if value is None:
        if required:
            raise ValidationError(f"Missing required argument '{param_name}'.", param=param_name)
        return None

    if not isinstance(value, str):
        value = str(value)

    cleaned = value.strip()
    if not cleaned:
        if required:
            raise ValidationError(f"Argument '{param_name}' cannot be empty.", param=param_name)
        return None

    if "\x00" in cleaned or len(cleaned) > max_len:
        raise ValidationError(f"Invalid length or null byte in '{param_name}'.", param=param_name)

    if _INJECTION_ARG_RE.search(cleaned):
        raise ValidationError(
            f"Rejected unsafe characters or injection pattern in argument '{param_name}'.",
            param=param_name,
        )
    return cleaned


def mask_phone(phone: Any) -> str:
    digits = re.sub(r"\D", "", str(phone or ""))
    if len(digits) <= 4:
        return "****"
    return ("*" * (len(digits) - 4)) + digits[-4:]


def mask_email(email: Any) -> str:
    s = str(email or "").strip()
    if "@" not in s:
        return "***"
    local, domain = s.split("@", 1)
    prefix = local[:1] if local else "*"
    return f"{prefix}***@{domain}"


def mask_generic_secret(value: Any) -> str:
    s = str(value or "").strip()
    if len(s) <= 4:
        return "****"
    return ("*" * (len(s) - 4)) + s[-4:]


def mask_record_pii(
    record: Dict[str, Any],
    extra_pii_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Mask PII attributes in a normalized business record before returning to LLM/logs."""
    pii_keys = set(DEFAULT_PII_FIELDS)
    if extra_pii_fields:
        pii_keys.update(k.lower().strip() for k in extra_pii_fields)

    def _mask_value(key_lower: str, val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, dict):
            if key_lower in pii_keys:
                return {dk: "***" for dk in val.keys()}
            return mask_record_pii(val, extra_pii_fields=extra_pii_fields)
        if isinstance(val, list):
            if key_lower in pii_keys:
                return ["***" for _ in val]
            return [
                mask_record_pii(item, extra_pii_fields=extra_pii_fields)
                if isinstance(item, dict)
                else item
                for item in val
            ]
        if key_lower in pii_keys or "phone" in key_lower or "mobile" in key_lower or "email" in key_lower:
            if "email" in key_lower:
                return mask_email(val)
            if "phone" in key_lower or "mobile" in key_lower:
                return mask_phone(val)
            if "address" in key_lower:
                s = str(val).strip()
                return f"***{s[-4:]}" if len(s) > 4 else "***"
            return mask_generic_secret(val)
        return val

    return {k: _mask_value(k.lower(), v) for k, v in record.items()}


def enforce_customer_authorization(
    records: List[Dict[str, Any]],
    *,
    authorized_customer_id: Optional[str] = None,
    authorized_phone: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Enforce caller/customer-level authorization when a call or session is scoped
    to a specific customer ID or phone number.
    """
    if not records:
        return records
    if not authorized_customer_id and not authorized_phone:
        return records

    norm_auth_cust = str(authorized_customer_id).strip().lower() if authorized_customer_id else None
    norm_auth_phone = re.sub(r"\D", "", str(authorized_phone))[-10:] if authorized_phone else None

    authorized_rows: List[Dict[str, Any]] = []
    for rec in records:
        rec_cust = str(rec.get("customer_id") or "").strip().lower()
        rec_phone = re.sub(r"\D", "", str(rec.get("phone") or ""))[-10:]

        if norm_auth_cust and rec_cust and rec_cust != norm_auth_cust:
            raise PermissionDeniedError("Unauthorized: record belongs to a different customer account.")
        if norm_auth_cust and not rec_cust and not norm_auth_phone:
            raise PermissionDeniedError("Unauthorized: record lacks customer ownership metadata.")
        if norm_auth_phone and rec_phone and rec_phone != norm_auth_phone:
            raise PermissionDeniedError("Unauthorized: record phone does not match verified caller.")
        authorized_rows.append(rec)

    return authorized_rows

