from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from storage.models.api_key import ApiKey
from storage.models.tenant import Tenant


class ApiKeyRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        stmt = (
            select(ApiKey)
            .options(selectinload(ApiKey.tenant))
            .where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        tenant_id: str,
        key_hash: str,
        name: str = "Default API Key",
        scopes: str = "*",
    ) -> ApiKey:
        api_key = ApiKey(
            tenant_id=tenant_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            is_active=True,
        )
        self.session.add(api_key)
        await self.session.commit()
        await self.session.refresh(api_key)
        return api_key

    async def list_by_tenant(self, tenant_id: str) -> List[ApiKey]:
        stmt = select(ApiKey).where(ApiKey.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
