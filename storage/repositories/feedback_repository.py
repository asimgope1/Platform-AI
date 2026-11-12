"""Tenant-scoped repository for CallFeedback."""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models.call_feedback import CallFeedback


class FeedbackRepository:
    """Async repository for explicit post-resolution caller ratings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_feedback(
        self,
        tenant_id: str,
        call_uuid: str,
        rating: Optional[int],
        feedback_text: Optional[str] = None,
        satisfaction_label: Optional[str] = None,
        issue_resolved: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> CallFeedback:
        row = CallFeedback(
            tenant_id=tenant_id,
            call_uuid=call_uuid,
            rating=rating,
            feedback_text=feedback_text,
            satisfaction_label=satisfaction_label,
            issue_resolved=issue_resolved,
            language=language,
        )
        self.session.add(row)
        await self.session.commit()
        return row

    async def get_call_feedback(self, tenant_id: str, call_uuid: str) -> List[CallFeedback]:
        stmt = select(CallFeedback).where(
            CallFeedback.tenant_id == tenant_id,
            CallFeedback.call_uuid == call_uuid,
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

