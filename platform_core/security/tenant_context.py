from typing import List, Optional, Set
from pydantic import BaseModel, Field
from platform_core.errors.exceptions import PermissionDeniedError


class TenantContext(BaseModel):
    tenant_id: str
    tenant_name: str
    api_key_id: Optional[str] = None
    scopes: Set[str] = Field(default_factory=lambda: {"*"})
    is_admin: bool = False

    def has_scope(self, required_scope: str) -> bool:
        if "*" in self.scopes or self.is_admin:
            return True
        return required_scope in self.scopes

    def check_scope(self, required_scope: str) -> None:
        if not self.has_scope(required_scope):
            raise PermissionDeniedError(f"Missing required scope: '{required_scope}'")
