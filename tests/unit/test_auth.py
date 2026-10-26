import pytest
from platform_core.security.api_keys import generate_api_key, hash_api_key, verify_api_key
from platform_core.security.tenant_context import TenantContext
from platform_core.errors.exceptions import PermissionDeniedError


def test_api_key_generation_and_verification():
    raw_key, key_hash = generate_api_key()
    assert raw_key.startswith("sk_live_")
    assert len(key_hash) == 64
    assert verify_api_key(raw_key, key_hash) is True
    assert verify_api_key("sk_live_wrong_key", key_hash) is False


def test_tenant_context_scopes():
    ctx = TenantContext(
        tenant_id="tenant-123",
        tenant_name="Acme",
        api_key_id="key-123",
        scopes={"chat:completions", "documents:read"},
        is_admin=False,
    )

    assert ctx.has_scope("chat:completions") is True
    assert ctx.has_scope("documents:read") is True
    assert ctx.has_scope("documents:write") is False

    # Check scope enforcement
    ctx.check_scope("chat:completions")
    with pytest.raises(PermissionDeniedError):
        ctx.check_scope("documents:write")


def test_admin_tenant_wildcard_scope():
    admin_ctx = TenantContext(
        tenant_id="tenant-admin",
        tenant_name="Admin",
        api_key_id="key-master",
        scopes={"*"},
        is_admin=True,
    )

    assert admin_ctx.has_scope("anything:allowed") is True
    admin_ctx.check_scope("any:action")
