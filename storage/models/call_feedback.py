from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from storage.database import Base


class CallFeedback(Base):
    """Explicit post-resolution caller satisfaction feedback."""

    __tablename__ = "call_feedback"
    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_call_feedback_rating_range"),
        Index("idx_call_feedback_uuid", "tenant_id", "call_uuid"),
        Index("idx_call_feedback_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    call_uuid: Mapped[str] = mapped_column(String(150), nullable=False)
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feedback_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    satisfaction_label: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    issue_resolved: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
