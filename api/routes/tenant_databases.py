"""
Tenant Database Onboarding, Schema Discovery, Entity Mapping, Activation & Monitoring APIs.
Strictly isolated by authenticated TenantContext; never exposes encrypted credentials.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.connectors.base import DiscoveredSchema
from data_access.entity_mapping import (
    EntityMappingConfig,
    EntityMappingSpec,
    suggest_entity_mapping,
)
from data_access.models import TenantDatabaseConfig, ToolExecutionAudit
from data_access.service import UniversalDataAccessService
from platform_core.security.authentication import get_current_tenant
from platform_core.security.tenant_context import TenantContext
from storage.database import get_db_session

router = APIRouter(prefix="/v1/tenants/databases", tags=["Tenant Databases"])


class RegisterDatabaseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    engine: Literal["postgresql", "mysql", "mssql", "oracle", "mongodb", "sqlite"]
    database_name: str = Field(..., min_length=1, max_length=512)
    host: Optional[str] = Field(default=None, max_length=255)
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    username: Optional[str] = Field(default=None, max_length=128)
    password: Optional[str] = Field(default=None, max_length=512)
    connection_uri: Optional[str] = Field(default=None, max_length=1024)
    schema_name: Optional[str] = Field(default=None, max_length=128)
    connection_options: Dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True


class UpdateMappingRequest(BaseModel):
    entities: Dict[str, EntityMappingSpec]


class RotateCredentialsRequest(BaseModel):
    new_password: Optional[str] = Field(default=None, max_length=512)
    new_connection_uri: Optional[str] = Field(default=None, max_length=1024)


def _serialize_config(cfg: TenantDatabaseConfig) -> Dict[str, Any]:
    """Serialize config metadata safely — NEVER includes encrypted or plaintext credentials."""
    return {
        "id": cfg.id,
        "tenant_id": cfg.tenant_id,
        "name": cfg.name,
        "engine": cfg.engine,
        "host": cfg.host,
        "port": cfg.port,
        "database_name": cfg.database_name,
        "username": cfg.username,
        "schema_name": cfg.schema_name,
        "status": cfg.status,
        "is_active": cfg.is_active,
        "read_only": cfg.read_only,
        "mapping_version": cfg.mapping_version,
        "mapping_validated": cfg.mapping_validated,
        "credential_key_version": cfg.credential_key_version,
        "credentials_configured": bool(cfg.encrypted_credentials),
        "last_tested_at": cfg.last_tested_at.isoformat() if cfg.last_tested_at else None,
        "last_test_ok": cfg.last_test_ok,
        "last_test_error": cfg.last_test_error,
        "created_at": cfg.created_at.isoformat() if cfg.created_at else None,
        "updated_at": cfg.updated_at.isoformat() if cfg.updated_at else None,
    }


@router.get("/metrics")
async def get_database_metrics(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Retrieve per-tenant query latency, connection failures, and recent tool audit events."""
    svc = UniversalDataAccessService(session)
    audits_res = await session.execute(
        select(ToolExecutionAudit)
        .where(ToolExecutionAudit.tenant_id == context.tenant_id)
        .order_by(ToolExecutionAudit.created_at.desc())
        .limit(20)
    )
    audits = audits_res.scalars().all()
    return {
        "metrics": svc.manager.metrics.snapshot_for_tenant(context.tenant_id),
        "recent_audits": [
            {
                "id": a.id,
                "tool_name": a.tool_name,
                "database_engine": a.database_engine,
                "status": a.status,
                "latency_ms": a.latency_ms,
                "row_count": a.row_count,
                "error_message": a.error_message,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in audits
        ],
    }


@router.get("")
async def list_tenant_databases(
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    res = await session.execute(
        select(TenantDatabaseConfig)
        .where(TenantDatabaseConfig.tenant_id == context.tenant_id)
        .order_by(TenantDatabaseConfig.created_at.desc())
    )
    configs = res.scalars().all()
    return {"data": [_serialize_config(c) for c in configs]}


@router.post("", status_code=status.HTTP_201_CREATED)
async def register_tenant_database(
    payload: RegisterDatabaseRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    cfg = await svc.register_config(
        tenant_id=context.tenant_id,
        name=payload.name,
        engine=payload.engine,
        database_name=payload.database_name,
        host=payload.host,
        port=payload.port,
        username=payload.username,
        password=payload.password,
        connection_uri=payload.connection_uri,
        schema_name=payload.schema_name,
        connection_options=payload.connection_options,
        read_only=payload.read_only,
    )
    return _serialize_config(cfg)


@router.post("/{config_id}/test")
async def test_tenant_database(
    config_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    res = await svc.test_config_connection(context.tenant_id, config_id)
    return {
        "ok": res["ok"],
        "message": res["message"],
        "config": _serialize_config(res["config"]),
    }


@router.post("/{config_id}/discover")
async def discover_tenant_database_schema(
    config_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    discovered = await svc.discover_and_save_schema(context.tenant_id, config_id)
    suggested = suggest_entity_mapping(discovered)
    return {
        "schema": discovered.model_dump(),
        "suggested_mapping": suggested.model_dump(),
    }


@router.get("/{config_id}/schema")
async def get_discovered_schema(
    config_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    cfg = await svc.get_config(context.tenant_id, config_id)
    return {
        "config_id": cfg.id,
        "schema": cfg.discovered_schema,
        "mapping_version": cfg.mapping_version,
        "mapping_validated": cfg.mapping_validated,
        "schema_mapping": cfg.schema_mapping,
    }


@router.put("/{config_id}/mapping")
async def update_entity_mapping(
    config_id: str,
    payload: UpdateMappingRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    cfg = await svc.update_mapping(
        context.tenant_id,
        config_id,
        EntityMappingConfig(entities=payload.entities),
    )
    return _serialize_config(cfg)


@router.post("/{config_id}/mapping/validate")
async def validate_entity_mapping(
    config_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    result = await svc.validate_mapping(context.tenant_id, config_id)
    return result.model_dump()


@router.post("/{config_id}/activate")
async def activate_tenant_database(
    config_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    cfg = await svc.activate_config(context.tenant_id, config_id)
    return _serialize_config(cfg)


@router.post("/{config_id}/deactivate")
async def deactivate_tenant_database(
    config_id: str,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    cfg = await svc.deactivate_config(context.tenant_id, config_id)
    return _serialize_config(cfg)


@router.post("/{config_id}/rotate-credentials")
async def rotate_database_credentials(
    config_id: str,
    payload: RotateCredentialsRequest,
    context: TenantContext = Depends(get_current_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    svc = UniversalDataAccessService(session)
    cfg = await svc.rotate_credentials(
        context.tenant_id,
        config_id,
        new_password=payload.new_password,
        new_connection_uri=payload.new_connection_uri,
    )
    return _serialize_config(cfg)

