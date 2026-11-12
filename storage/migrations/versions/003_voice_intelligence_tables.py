"""Add hybrid voice intelligence, call events, feedback, transfer, and resolution tables.

Revision ID: 003_voice_intelligence_tables
Revises: 002_pgvector_support
Create Date: 2026-11-11 11:05:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "003_voice_intelligence_tables"
down_revision: Union[str, None] = "002_pgvector_support"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB() if is_postgres else sa.JSON()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if "call_intelligence" not in existing:
        op.create_table(
            "call_intelligence",
            sa.Column("id", sa.BigInteger() if is_postgres else sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("tenant_id", sa.String(100), nullable=False),
            sa.Column("call_uuid", sa.String(150), nullable=False),
            sa.Column("turn_number", sa.Integer(), nullable=False),
            sa.Column("intent", sa.String(100), nullable=True),
            sa.Column("text_emotion", sa.String(50), nullable=True),
            sa.Column("voice_emotion", sa.String(50), nullable=True),
            sa.Column("emotion_confidence", sa.Float(), nullable=True),
            sa.Column("emotion_trend", sa.String(30), nullable=True),
            sa.Column("caller_certainty", sa.Float(), nullable=True),
            sa.Column("system_confidence", sa.Float(), nullable=True),
            sa.Column("satisfaction_score", sa.Float(), nullable=True),
            sa.Column("transcript_confidence", sa.Float(), nullable=True),
            sa.Column("audio_quality_score", sa.Float(), nullable=True),
            sa.Column("resolution_status", sa.String(30), nullable=True),
            sa.Column("recommended_action", sa.String(50), nullable=True),
            sa.Column("analysis", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("tenant_id", "call_uuid", "turn_number", name="uq_call_intelligence_turn"),
        )
        op.create_index("idx_call_intelligence_uuid", "call_intelligence", ["tenant_id", "call_uuid"])
        op.create_index("idx_call_intelligence_created", "call_intelligence", ["created_at"])

    if "call_events" not in existing:
        op.create_table(
            "call_events",
            sa.Column("id", sa.BigInteger() if is_postgres else sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("tenant_id", sa.String(100), nullable=False),
            sa.Column("call_uuid", sa.String(150), nullable=False),
            sa.Column("event_type", sa.String(80), nullable=False),
            sa.Column("event_data", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_call_events_uuid", "call_events", ["tenant_id", "call_uuid"])
        op.create_index("idx_call_events_created", "call_events", ["created_at"])

    if "call_feedback" not in existing:
        op.create_table(
            "call_feedback",
            sa.Column("id", sa.BigInteger() if is_postgres else sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("tenant_id", sa.String(100), nullable=False),
            sa.Column("call_uuid", sa.String(150), nullable=False),
            sa.Column("rating", sa.Integer(), sa.CheckConstraint("rating BETWEEN 1 AND 5"), nullable=True),
            sa.Column("feedback_text", sa.Text(), nullable=True),
            sa.Column("satisfaction_label", sa.String(30), nullable=True),
            sa.Column("issue_resolved", sa.Boolean(), nullable=True),
            sa.Column("language", sa.String(10), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_call_feedback_uuid", "call_feedback", ["tenant_id", "call_uuid"])
        op.create_index("idx_call_feedback_created", "call_feedback", ["created_at"])

    if "call_transfer" not in existing:
        op.create_table(
            "call_transfer",
            sa.Column("id", sa.BigInteger() if is_postgres else sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("tenant_id", sa.String(100), nullable=False),
            sa.Column("call_uuid", sa.String(150), nullable=False),
            sa.Column("transfer_reason", sa.String(100), nullable=True),
            sa.Column("emotion_at_transfer", sa.String(50), nullable=True),
            sa.Column("satisfaction_at_transfer", sa.Float(), nullable=True),
            sa.Column("caller_accepted", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("transfer_status", sa.String(30), nullable=True),
            sa.Column("agent_identifier", sa.String(100), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_call_transfer_uuid", "call_transfer", ["tenant_id", "call_uuid"])
        op.create_index("idx_call_transfer_created", "call_transfer", ["created_at"])

    if "conversation_resolution" not in existing:
        op.create_table(
            "conversation_resolution",
            sa.Column("id", sa.BigInteger() if is_postgres else sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("tenant_id", sa.String(100), nullable=False),
            sa.Column("call_uuid", sa.String(150), nullable=False),
            sa.Column("initial_intent", sa.String(100), nullable=True),
            sa.Column("final_resolution", sa.String(100), nullable=True),
            sa.Column("issue_resolved", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("resolved_by", sa.String(30), nullable=True),
            sa.Column("customer_confirmed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("escalation_required", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_conv_resolution_uuid", "conversation_resolution", ["tenant_id", "call_uuid"])
        op.create_index("idx_conv_resolution_created", "conversation_resolution", ["created_at"])


def downgrade() -> None:
    for table in ("conversation_resolution", "call_transfer", "call_feedback", "call_events", "call_intelligence"):
        op.drop_table(table)

