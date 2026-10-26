"""Add pgvector extension and 384-dimensional embedding column to document_chunks.

Revision ID: 002_pgvector_support
Revises: 001_initial_schema
Create Date: 2026-10-26 14:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = '002_pgvector_support'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add embedding column if not present
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("document_chunks")]
    if "embedding" not in columns:
        if is_postgres:
            op.add_column("document_chunks", sa.Column("embedding", Vector(384), nullable=True))
        else:
            op.add_column("document_chunks", sa.Column("embedding", sa.NullType(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("document_chunks")]
    if "embedding" in columns:
        op.drop_column("document_chunks", "embedding")
