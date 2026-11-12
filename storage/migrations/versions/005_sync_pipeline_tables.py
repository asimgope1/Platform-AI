"""Add sync_events, sync_record_states, and sync_dead_letters tables to internal AI database.

Revision ID: 005_sync_pipeline_tables
Revises: 004_tenant_db_configs
Create Date: 2026-11-11 17:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "005_sync_pipeline_tables"
down_revision: Union[str, None] = "004_tenant_db_configs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB() if is_postgres else sa.JSON()
    existing = set(sa.inspect(bind).get_table_names())

    if "sync_events" not in existing:
        op.create_table(
            "sync_events",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("idempotency_key", sa.String(128), nullable=False),
            sa.Column("entity_type", sa.String(64), nullable=False),
            sa.Column("record_id", sa.String(128), nullable=False),
            sa.Column("event_type", sa.String(32), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("title", sa.String(255), nullable=True),
            sa.Column("payload", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
            sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("tenant_id", "idempotency_key", name="uq_sync_event_tenant_idem"),
        )
        op.create_index("idx_sync_event_status_retry", "sync_events", ["tenant_id", "status", "next_retry_at"])

    if "sync_record_states" not in existing:
        op.create_table(
            "sync_record_states",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("entity_type", sa.String(64), nullable=False),
            sa.Column("record_id", sa.String(128), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("document_id", sa.String(36), nullable=True),
            sa.Column("content_hash", sa.String(64), nullable=True),
            sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("tenant_id", "entity_type", "record_id", name="uq_sync_state_tenant_record"),
        )

    if "sync_dead_letters" not in existing:
        op.create_table(
            "sync_dead_letters",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("sync_event_id", sa.String(36), nullable=False),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("entity_type", sa.String(64), nullable=False),
            sa.Column("record_id", sa.String(128), nullable=False),
            sa.Column("event_type", sa.String(32), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("payload", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("error_message", sa.Text(), nullable=False),
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_sync_dlq_tenant", "sync_dead_letters", ["tenant_id", "failed_at"])


def downgrade() -> None:
    op.drop_table("sync_dead_letters")
    op.drop_table("sync_record_states")
    op.drop_table("sync_events")

