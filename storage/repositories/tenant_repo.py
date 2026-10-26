from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from storage.models.tenant import Tenant


class TenantRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, tenant_id: str) -> Optional[Tenant]:
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Tenant]:
        stmt = select(Tenant).where(Tenant.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, name: str) -> Tenant:
        tenant = Tenant(name=name)
        self.session.add(tenant)
        await self.session.commit()
        await self.session.refresh(tenant)
        return tenant

    async def list_tenants(self) -> List[Tenant]:
        stmt = select(Tenant)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
