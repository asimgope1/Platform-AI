"""Add tenant_database_configs and tool_execution_audits tables to internal AI database.

Revision ID: 004_tenant_db_configs
Revises: 003_voice_intelligence_tables
Create Date: 2026-11-11 16:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004_tenant_db_configs"
down_revision: Union[str, None] = "003_voice_intelligence_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    json_type = postgresql.JSONB() if is_postgres else sa.JSON()
    existing = set(sa.inspect(bind).get_table_names())

    if "tenant_database_configs" not in existing:
        op.create_table(
            "tenant_database_configs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("engine", sa.String(32), nullable=False),
            sa.Column("encrypted_credentials", sa.Text(), nullable=False),
            sa.Column("credential_key_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("host", sa.String(255), nullable=True),
            sa.Column("port", sa.Integer(), nullable=True),
            sa.Column("database_name", sa.String(512), nullable=False),
            sa.Column("username", sa.String(128), nullable=True),
            sa.Column("schema_name", sa.String(128), nullable=True),
            sa.Column("connection_options", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("status", sa.String(32), nullable=False, server_default="registered"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("read_only", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("mapping_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("schema_mapping", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("mapping_validated", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("discovered_schema", json_type, nullable=True),
            sa.Column("last_tested_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_test_ok", sa.Boolean(), nullable=True),
            sa.Column("last_test_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_tenant_db_active", "tenant_database_configs", ["tenant_id", "is_active"])

    if "tool_execution_audits" not in existing:
        op.create_table(
            "tool_execution_audits",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("tenant_id", sa.String(36), nullable=False),
            sa.Column("config_id", sa.String(36), nullable=True),
            sa.Column("tool_name", sa.String(80), nullable=False),
            sa.Column("database_engine", sa.String(32), nullable=True),
            sa.Column("caller_subject", sa.String(128), nullable=True),
            sa.Column("arguments_masked", json_type, nullable=False, server_default=sa.text("'{}'")),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
            sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_tool_exec_audit_tenant", "tool_execution_audits", ["tenant_id", "tool_name"])
        op.create_index("idx_tool_exec_audit_created", "tool_execution_audits", ["created_at"])


def downgrade() -> None:
    op.drop_table("tool_execution_audits")
    op.drop_table("tenant_database_configs")

