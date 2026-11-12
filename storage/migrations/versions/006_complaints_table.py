"""Add complaints table for call/chat complaint registration and external API tracking.

Revision ID: 006_complaints_table
Revises: 005_sync_pipeline_tables
Create Date: 2026-11-11 17:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "006_complaints_table"
down_revision: Union[str, None] = "005_sync_pipeline_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB() if is_postgres else sa.JSON()
    existing = set(sa.inspect(bind).get_table_names())

    if "complaints" not in existing:
        op.create_table(
            "complaints",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("ticket_number", sa.String(32), nullable=False),
            sa.Column("tenant_id", sa.String(100), nullable=False),
            sa.Column("call_uuid", sa.String(150), nullable=True),
            sa.Column("caller_number", sa.String(64), nullable=True),
            sa.Column("customer_id", sa.String(128), nullable=True),
            sa.Column("order_id", sa.String(128), nullable=True),
            sa.Column("category", sa.String(64), nullable=False, server_default="general"),
            sa.Column("subject", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("priority", sa.String(32), nullable=False, server_default="normal"),
            sa.Column("status", sa.String(32), nullable=False, server_default="open"),
            sa.Column("source_channel", sa.String(32), nullable=False, server_default="voice_ivr"),
            sa.Column("external_api_status", sa.String(32), nullable=False, server_default="local_only"),
            sa.Column("external_reference", sa.String(128), nullable=True),
            sa.Column("external_response_json", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("resolution_notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("tenant_id", "ticket_number", name="idx_complaints_tenant_ticket"),
        )
        op.create_index("idx_complaints_tenant_status", "complaints", ["tenant_id", "status"])
        op.create_index("idx_complaints_tenant_customer", "complaints", ["tenant_id", "customer_id"])
        op.create_index("idx_complaints_tenant_call", "complaints", ["tenant_id", "call_uuid"])


def downgrade() -> None:
    op.drop_table("complaints")

