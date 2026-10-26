from typing import Set
from platform_core.security.tenant_context import TenantContext
from platform_core.errors.exceptions import PermissionDeniedError


class Scopes:
    CHAT_READ = "chat:read"
    CHAT_WRITE = "chat:write"
    DOCUMENTS_READ = "documents:read"
    DOCUMENTS_WRITE = "documents:write"
    AUDIO_READ = "audio:read"
    AUDIO_WRITE = "audio:write"
    ADMIN = "admin"


def check_permission(context: TenantContext, required_scope: str) -> None:
    if not context.has_scope(required_scope):
        raise PermissionDeniedError(
            f"Tenant '{context.tenant_id}' lacks required permission scope '{required_scope}'."
        )
