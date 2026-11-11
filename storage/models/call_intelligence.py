from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from storage.database import Base


class CallIntelligence(Base):
    """Per-turn audio, emotion, confidence, and resolution intelligence assessment."""

    __tablename__ = "call_intelligence"
    __table_args__ = (
        UniqueConstraint("tenant_id", "call_uuid", "turn_number", name="uq_call_intelligence_turn"),
        Index("idx_call_intelligence_uuid", "tenant_id", "call_uuid"),
        Index("idx_call_intelligence_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    call_uuid: Mapped[str] = mapped_column(String(150), nullable=False)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    text_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    voice_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    emotion_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    emotion_trend: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    caller_certainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    system_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    satisfaction_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    transcript_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    audio_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    recommended_action: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    analysis: Mapped[Dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        default=dict,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
