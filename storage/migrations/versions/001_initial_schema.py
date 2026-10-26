"""Initial schema for tenants, api keys, documents, chunks, and usage events.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-26 13:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Tenants
    op.create_table(
        'tenants',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False, unique=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    # 2. API Keys
    op.create_table(
        'api_keys',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('tenant_id', sa.String(length=36), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False, unique=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('scopes', sa.String(length=255), nullable=False, server_default='*'),
        sa.Column('is_revoked', sa.Boolean(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_api_keys_tenant_id', 'api_keys', ['tenant_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])

    # 3. Documents
    op.create_table(
        'documents',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('tenant_id', sa.String(length=36), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False, server_default='text'),
        sa.Column('total_chunks', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_documents_tenant_id', 'documents', ['tenant_id'])

    # 4. Document Chunks
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('document_id', sa.String(length=36), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('tenant_id', sa.String(length=36), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding_json', sa.Text(), nullable=False),
        sa.Column('meta_json', sa.Text(), nullable=True),
    )
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('ix_document_chunks_tenant_id', 'document_chunks', ['tenant_id'])

    # 5. Usage Events
    op.create_table(
        'usage_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('request_id', sa.String(length=64), nullable=False),
        sa.Column('tenant_id', sa.String(length=36), sa.ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_alias', sa.String(length=100), nullable=False),
        sa.Column('provider_id', sa.String(length=100), nullable=False),
        sa.Column('prompt_tokens', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('completion_tokens', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('total_tokens', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('status_code', sa.Integer(), nullable=False, server_default=sa.text('200')),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_usage_events_request_id', 'usage_events', ['request_id'])
    op.create_index('ix_usage_events_tenant_id', 'usage_events', ['tenant_id'])


def downgrade() -> None:
    op.drop_table('usage_events')
    op.drop_table('document_chunks')
    op.drop_table('documents')
    op.drop_table('api_keys')
    op.drop_table('tenants')
