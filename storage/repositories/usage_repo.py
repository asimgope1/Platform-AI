from sqlalchemy.ext.asyncio import AsyncSession
from storage.models.usage_event import UsageEvent


class UsageRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_event(
        self,
        request_id: str,
        tenant_id: str,
        model_alias: str,
        provider_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        status_code: int = 200,
    ) -> UsageEvent:
        event = UsageEvent(
            request_id=request_id,
            tenant_id=tenant_id,
            model_alias=model_alias,
            provider_id=provider_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            status_code=status_code,
        )
        self.session.add(event)
        await self.session.commit()
        return event
