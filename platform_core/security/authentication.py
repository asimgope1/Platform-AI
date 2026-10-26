import secrets
from typing import Optional
from fastapi import Header, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from platform_core.config.settings import settings
from platform_core.security.tenant_context import TenantContext
from platform_core.security.api_keys import hash_api_key
from platform_core.errors.exceptions import AuthenticationError
from platform_core.observability.logging import tenant_id_ctx
from storage.database import get_db_session
from storage.repositories.api_key_repo import ApiKeyRepository


async def get_current_tenant(
    authorization: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_db_session),
) -> TenantContext:
    """Authenticates the incoming request using the Bearer API key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Missing or malformed Authorization header. Expected 'Bearer <key>'.")

    raw_key = authorization[len("Bearer ") :].strip()
    if not raw_key:
        raise AuthenticationError("API key cannot be empty.")

    # Check Master API Key (only when explicitly configured and matches securely)
    configured_master_key = settings.MASTER_API_KEY
    if (
        configured_master_key
        and raw_key
        and secrets.compare_digest(raw_key, configured_master_key)
    ):
        ctx = TenantContext(
            tenant_id="tenant-master",
            tenant_name="Master Administrator",
            api_key_id="master-key",
            scopes={"*"},
            is_admin=True,
        )
        tenant_id_ctx.set(ctx.tenant_id)
        return ctx

    # Check Tenant Database Keys
    key_hash = hash_api_key(raw_key)
    repo = ApiKeyRepository(session)
    api_key_record = await repo.get_by_hash(key_hash)

    if not api_key_record:
        raise AuthenticationError("Invalid API key.")

    tenant = api_key_record.tenant
    if not tenant or not tenant.is_active:
        raise AuthenticationError("Associated tenant is inactive or not found.")

    scopes_set = set(s.strip() for s in api_key_record.scopes.split(",") if s.strip())
    ctx = TenantContext(
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        api_key_id=api_key_record.id,
        scopes=scopes_set,
        is_admin=False,
    )
    tenant_id_ctx.set(ctx.tenant_id)
    return ctx
