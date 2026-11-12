"""
Internal AI database SQLAlchemy models for per-tenant external DB configurations,
schema discovery snapshots, versioned entity mappings, and tool execution audits.

IMPORTANT: These tables live ONLY in the internal AI platform database — never
migrated onto customer business databases.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from storage.database import Base


class TenantDatabaseConfig(Base):
    __tablename__ = "tenant_database_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="primary")
    engine: Mapped[str] = mapped_column(String(32), nullable=False)  # postgresql|mysql|mssql|oracle|mongodb|sqlite

    # Encrypted JSON credentials reference (never exposed in API responses or LLM prompts)
    encrypted_credentials: Mapped[str] = mapped_column(Text, nullable=False, default="")
    credential_key_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    host: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    database_name: Mapped[str] = mapped_column(String(512), nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    schema_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    connection_options: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Lifecycle & safety posture
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="registered")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    read_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Versioned entity mapping & validation state
    mapping_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    schema_mapping: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    mapping_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    discovered_schema: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    last_tested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_test_ok: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    last_test_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("idx_tenant_db_active", "tenant_id", "is_active"),
    )


class ToolExecutionAudit(Base):
    __tablename__ = "tool_execution_audits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    config_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    database_engine: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    caller_subject: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    arguments_masked: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False)  # success|denied|error|timeout
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

