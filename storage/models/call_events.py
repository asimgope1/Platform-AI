from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.database import Base


class CallEvent(Base):
    """Important lifecycle and intelligence events observed during a call."""

    __tablename__ = "call_events"
    __table_args__ = (
        Index("idx_call_events_uuid", "tenant_id", "call_uuid"),
        Index("idx_call_events_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    call_uuid: Mapped[str] = mapped_column(String(150), nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    event_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class CallTransfer(Base):
    """Audit trail for human agent escalation offers and handoffs."""

    __tablename__ = "call_transfer"
    __table_args__ = (
        Index("idx_call_transfer_uuid", "tenant_id", "call_uuid"),
        Index("idx_call_transfer_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    call_uuid: Mapped[str] = mapped_column(String(150), nullable=False)
    transfer_reason: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    emotion_at_transfer: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    satisfaction_at_transfer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    caller_accepted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    transfer_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    agent_identifier: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ConversationResolution(Base):
    """Final per-call outcome and confirmed resolution record."""

    __tablename__ = "conversation_resolution"
    __table_args__ = (
        Index("idx_conv_resolution_uuid", "tenant_id", "call_uuid"),
        Index("idx_conv_resolution_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    call_uuid: Mapped[str] = mapped_column(String(150), nullable=False)
    initial_intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    final_resolution: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    issue_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    resolved_by: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    customer_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    escalation_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

