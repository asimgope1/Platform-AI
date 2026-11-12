from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Index, JSON, String, Text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from storage.database import Base

_JSONType = JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")


class Complaint(Base):
    """Customer complaints registered over live IVR calls, chat, or REST APIs."""

    __tablename__ = "complaints"
    __table_args__ = (
        Index("idx_complaints_tenant_ticket", "tenant_id", "ticket_number", unique=True),
        Index("idx_complaints_tenant_status", "tenant_id", "status"),
        Index("idx_complaints_tenant_customer", "tenant_id", "customer_id"),
        Index("idx_complaints_tenant_call", "tenant_id", "call_uuid"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_number: Mapped[str] = mapped_column(String(32), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    call_uuid: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    caller_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    customer_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    order_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="normal")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    source_channel: Mapped[str] = mapped_column(String(32), nullable=False, default="voice_ivr")
    external_api_status: Mapped[str] = mapped_column(String(32), nullable=False, default="local_only")
    external_reference: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    external_response_json: Mapped[Dict[str, Any]] = mapped_column(_JSONType, nullable=False, default=dict)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

