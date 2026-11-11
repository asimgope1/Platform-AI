"""Tenant-scoped repository for CallIntelligence, CallEvent, CallTransfer, and ConversationResolution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.models.call_events import CallEvent, CallTransfer, ConversationResolution
from storage.models.call_intelligence import CallIntelligence


class IntelligenceRepository:
    """Async repository for voice intelligence turn telemetry and outcomes."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_turn(
        self,
        tenant_id: str,
        call_uuid: str,
        turn_number: int,
        intent: Optional[str],
        text_emotion: Optional[str],
        voice_emotion: Optional[str],
        emotion_confidence: Optional[float],
        emotion_trend: Optional[str],
        caller_certainty: Optional[float],
        system_confidence: Optional[float],
        satisfaction_score: Optional[float],
        transcript_confidence: Optional[float],
        audio_quality_score: Optional[float],
        resolution_status: Optional[str],
        recommended_action: Optional[str],
        analysis: Optional[Dict[str, Any]] = None,
    ) -> CallIntelligence:
        row = CallIntelligence(
            tenant_id=tenant_id,
            call_uuid=call_uuid,
            turn_number=turn_number,
            intent=intent,
            text_emotion=text_emotion,
            voice_emotion=voice_emotion,
            emotion_confidence=emotion_confidence,
            emotion_trend=emotion_trend,
            caller_certainty=caller_certainty,
            system_confidence=system_confidence,
            satisfaction_score=satisfaction_score,
            transcript_confidence=transcript_confidence,
            audio_quality_score=audio_quality_score,
            resolution_status=resolution_status,
            recommended_action=recommended_action,
            analysis=analysis or {},
        )
        self.session.add(row)
        await self.session.commit()
        return row

    async def record_event(
        self,
        tenant_id: str,
        call_uuid: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> CallEvent:
        row = CallEvent(
            tenant_id=tenant_id,
            call_uuid=call_uuid,
            event_type=event_type,
            event_data=event_data or {},
        )
        self.session.add(row)
        await self.session.commit()
        return row

    async def record_transfer(
        self,
        tenant_id: str,
        call_uuid: str,
        transfer_reason: str,
        emotion_at_transfer: str,
        satisfaction_at_transfer: Optional[float],
        caller_accepted: bool,
        transfer_status: str,
        agent_identifier: Optional[str] = None,
    ) -> CallTransfer:
        row = CallTransfer(
            tenant_id=tenant_id,
            call_uuid=call_uuid,
            transfer_reason=transfer_reason,
            emotion_at_transfer=emotion_at_transfer,
            satisfaction_at_transfer=satisfaction_at_transfer,
            caller_accepted=caller_accepted,
            transfer_status=transfer_status,
            agent_identifier=agent_identifier,
        )
        self.session.add(row)
        await self.session.commit()
        return row

    async def record_resolution(
        self,
        tenant_id: str,
        call_uuid: str,
        initial_intent: Optional[str],
        final_resolution: Optional[str],
        issue_resolved: bool,
        resolved_by: str,
        customer_confirmed: bool,
        escalation_required: bool,
    ) -> ConversationResolution:
        row = ConversationResolution(
            tenant_id=tenant_id,
            call_uuid=call_uuid,
            initial_intent=initial_intent,
            final_resolution=final_resolution,
            issue_resolved=issue_resolved,
            resolved_by=resolved_by,
            customer_confirmed=customer_confirmed,
            escalation_required=escalation_required,
        )
        self.session.add(row)
        await self.session.commit()
        return row

    async def get_call_turns(self, tenant_id: str, call_uuid: str) -> List[CallIntelligence]:
        stmt = (
            select(CallIntelligence)
            .where(CallIntelligence.tenant_id == tenant_id, CallIntelligence.call_uuid == call_uuid)
            .order_by(CallIntelligence.turn_number.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
