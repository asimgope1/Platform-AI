"""
Universal Data Access Service: the single tenant-isolated entry point used by AI tools
and database onboarding routes to query customer schemas without leaking SQL or credentials.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.connectors.base import (
    ConnectionConfigSpec,
    DiscoveredSchema,
    QueryFilter,
    QueryJoin,
    QueryResult,
    QuerySpec,
)
from data_access.connectors.connection_manager import ConnectionManager, connection_manager
from data_access.entity_mapping import (
    EntityMappingConfig,
    MappingValidationResult,
    validate_mapping_against_schema,
)
from data_access.models import TenantDatabaseConfig, ToolExecutionAudit
from data_access.schema_discovery import SchemaDiscoveryService
from data_access.security import (
    CredentialCipher,
    enforce_customer_authorization,
    mask_record_pii,
)
from platform_core.errors.exceptions import PermissionDeniedError, ValidationError


class UniversalDataAccessService:
    """Orchestrates tenant DB config resolution, schema validation, query translation, and audit logging."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        manager: ConnectionManager = connection_manager,
        cipher: Optional[CredentialCipher] = None,
    ):
        self.session = session
        self.manager = manager
        self.cipher = cipher or CredentialCipher()
        self.discovery = SchemaDiscoveryService(manager)

    def _to_connection_spec(self, cfg: TenantDatabaseConfig) -> ConnectionConfigSpec:
        creds = self.cipher.decrypt_credentials(cfg.encrypted_credentials)
        return ConnectionConfigSpec(
            tenant_id=cfg.tenant_id,
            config_id=cfg.id,
            engine=cfg.engine,  # type: ignore[arg-type]
            host=cfg.host,
            port=cfg.port,
            database_name=cfg.database_name,
            username=cfg.username,
            password=creds.get("password"),
            connection_uri=creds.get("connection_uri"),
            schema_name=cfg.schema_name,
            read_only=bool(cfg.read_only),
            options=dict(cfg.connection_options or {}),
        )

    async def get_config(self, tenant_id: str, config_id: str) -> TenantDatabaseConfig:
        res = await self.session.execute(
            select(TenantDatabaseConfig).where(
                TenantDatabaseConfig.id == config_id,
                TenantDatabaseConfig.tenant_id == tenant_id,
            )
        )
        cfg = res.scalar_one_or_none()
        if not cfg:
            raise PermissionDeniedError(
                f"Database configuration '{config_id}' not found for tenant '{tenant_id}'."
            )
        return cfg

    async def get_active_config(self, tenant_id: str) -> TenantDatabaseConfig:
        res = await self.session.execute(
            select(TenantDatabaseConfig)
            .where(
                TenantDatabaseConfig.tenant_id == tenant_id,
                TenantDatabaseConfig.is_active == True,  # noqa: E712
            )
            .order_by(TenantDatabaseConfig.updated_at.desc())
            .limit(1)
        )
        cfg = res.scalar_one_or_none()
        if not cfg:
            raise ValidationError(
                f"No active customer database configured for tenant '{tenant_id}'."
            )
        if not cfg.mapping_validated:
            raise PermissionDeniedError(
                "Tenant database mapping has not been validated and cannot serve queries."
            )
        if not cfg.read_only:
            raise PermissionDeniedError("Read-only enforcement requires external DB config to have read_only=True.")
        return cfg

    async def register_config(
        self,
        *,
        tenant_id: str,
        name: str,
        engine: str,
        database_name: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_uri: Optional[str] = None,
        schema_name: Optional[str] = None,
        connection_options: Optional[Dict[str, Any]] = None,
        read_only: bool = True,
    ) -> TenantDatabaseConfig:
        if not read_only:
            raise PermissionDeniedError("External customer databases must be registered as read_only=True.")

        enc = self.cipher.encrypt_credentials(
            {
                k: v
                for k, v in {"password": password, "connection_uri": connection_uri}.items()
                if v is not None
            }
        )
        cfg = TenantDatabaseConfig(
            tenant_id=tenant_id,
            name=name,
            engine=engine.lower().strip(),
            encrypted_credentials=enc,
            credential_key_version=1,
            host=host,
            port=port,
            database_name=database_name,
            username=username,
            schema_name=schema_name,
            connection_options=connection_options or {},
            status="registered",
            is_active=False,
            read_only=True,
            mapping_version=1,
            schema_mapping={},
            mapping_validated=False,
        )
        self.session.add(cfg)
        await self.session.commit()
        await self.session.refresh(cfg)
        return cfg

    async def test_config_connection(self, tenant_id: str, config_id: str) -> Dict[str, Any]:
        cfg = await self.get_config(tenant_id, config_id)
        spec = self._to_connection_spec(cfg)
        adapter = await self.manager.get_adapter(spec)
        ok, msg = await adapter.test_connection()
        cfg.last_tested_at = datetime.now(timezone.utc)
        cfg.last_test_ok = ok
        cfg.last_test_error = None if ok else msg
        if ok and cfg.status == "registered":
            cfg.status = "tested"
        elif not ok:
            self.manager.metrics.record_connection_failure(tenant_id)
            cfg.status = "error"
        await self.session.commit()
        await self.session.refresh(cfg)
        return {"ok": ok, "message": msg, "config": cfg}

    async def discover_and_save_schema(self, tenant_id: str, config_id: str) -> DiscoveredSchema:
        cfg = await self.get_config(tenant_id, config_id)
        spec = self._to_connection_spec(cfg)
        discovered = await self.discovery.inspect_database(spec)
        cfg.discovered_schema = discovered.model_dump()
        cfg.last_tested_at = datetime.now(timezone.utc)
        cfg.last_test_ok = True
        cfg.last_test_error = None
        if cfg.status in ("registered", "error"):
            cfg.status = "tested"
        await self.session.commit()
        await self.session.refresh(cfg)
        return discovered

    async def update_mapping(
        self,
        tenant_id: str,
        config_id: str,
        mapping: EntityMappingConfig,
    ) -> TenantDatabaseConfig:
        cfg = await self.get_config(tenant_id, config_id)
        next_version = (cfg.mapping_version or 0) + 1
        updated_mapping = mapping.model_copy(update={"version": next_version})
        cfg.mapping_version = next_version
        cfg.schema_mapping = updated_mapping.model_dump()
        # Any mapping edit invalidates previous validation & requires re-validation before activation
        cfg.mapping_validated = False
        if cfg.is_active:
            cfg.is_active = False
            cfg.status = "inactive"
            await self.manager.invalidate(tenant_id, config_id)
        await self.session.commit()
        await self.session.refresh(cfg)
        return cfg

    async def validate_mapping(self, tenant_id: str, config_id: str) -> MappingValidationResult:
        cfg = await self.get_config(tenant_id, config_id)
        if not cfg.discovered_schema:
            return MappingValidationResult(
                valid=False,
                errors=["Schema has not been discovered yet. Run discovery before validating mapping."],
            )
        discovered = DiscoveredSchema(**cfg.discovered_schema)
        mapping = EntityMappingConfig(**(cfg.schema_mapping or {}))
        result = validate_mapping_against_schema(mapping, discovered)
        cfg.mapping_validated = result.valid
        if result.valid and not cfg.is_active:
            cfg.status = "validated"
        await self.session.commit()
        await self.session.refresh(cfg)
        return result

    async def activate_config(self, tenant_id: str, config_id: str) -> TenantDatabaseConfig:
        cfg = await self.get_config(tenant_id, config_id)
        if not cfg.last_test_ok:
            raise ValidationError("Cannot activate database connection before a successful connection test.")
        if not cfg.mapping_validated:
            raise ValidationError("Cannot activate an unvalidated schema mapping. Validate mapping first.")

        # Deactivate any other active config for this tenant cleanly
        existing_res = await self.session.execute(
            select(TenantDatabaseConfig).where(
                TenantDatabaseConfig.tenant_id == tenant_id,
                TenantDatabaseConfig.is_active == True,  # noqa: E712
                TenantDatabaseConfig.id != config_id,
            )
        )
        for other in existing_res.scalars().all():
            other.is_active = False
            other.status = "inactive"
            await self.manager.invalidate(tenant_id, other.id)

        cfg.is_active = True
        cfg.status = "active"
        await self.session.commit()
        await self.session.refresh(cfg)
        return cfg

    async def deactivate_config(self, tenant_id: str, config_id: str) -> TenantDatabaseConfig:
        cfg = await self.get_config(tenant_id, config_id)
        cfg.is_active = False
        cfg.status = "inactive"
        await self.manager.invalidate(tenant_id, config_id)
        await self.session.commit()
        await self.session.refresh(cfg)
        return cfg

    async def rotate_credentials(
        self,
        tenant_id: str,
        config_id: str,
        *,
        new_password: Optional[str] = None,
        new_connection_uri: Optional[str] = None,
    ) -> TenantDatabaseConfig:
        cfg = await self.get_config(tenant_id, config_id)
        if new_password is not None or new_connection_uri is not None:
            current = self.cipher.decrypt_credentials(cfg.encrypted_credentials)
            if new_password is not None:
                current["password"] = new_password
            if new_connection_uri is not None:
                current["connection_uri"] = new_connection_uri
            cfg.encrypted_credentials = self.cipher.encrypt_credentials(current)
        else:
            cfg.encrypted_credentials = self.cipher.rotate_token(cfg.encrypted_credentials)

        cfg.credential_key_version = (cfg.credential_key_version or 1) + 1
        await self.manager.invalidate(tenant_id, config_id)
        await self.session.commit()
        await self.session.refresh(cfg)
        return cfg

    async def query_entity(
        self,
        *,
        tenant_id: str,
        tool_name: str,
        entity: str,
        filters: Dict[str, Any],
        filter_operators: Optional[Dict[str, str]] = None,
        select_fields: Optional[List[str]] = None,
        include_related_customer: bool = False,
        authorized_customer_id: Optional[str] = None,
        authorized_phone: Optional[str] = None,
        mask_pii: bool = True,
        order_by: Optional[str] = None,
        descending: bool = False,
        limit: int = 10,
        timeout_seconds: float = 20.0,
        caller_subject: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a universal entity lookup for a single tenant, automatically translating
        canonical entities/fields to that tenant's physical schema and auditing execution.
        """
        t0 = time.perf_counter()
        cfg: Optional[TenantDatabaseConfig] = None
        audit_status = "success"
        err_msg: Optional[str] = None
        rows: List[Dict[str, Any]] = []

        try:
            cfg = await self.get_active_config(tenant_id)
            mapping = EntityMappingConfig(**(cfg.schema_mapping or {}))
            ent_spec = mapping.entities.get(entity)
            if not ent_spec:
                raise ValidationError(
                    f"Entity '{entity}' is not mapped for tenant '{tenant_id}'."
                )

            # Build canonical -> physical projection for requested (or all mapped) fields
            if select_fields:
                projection = {
                    f: ent_spec.fields[f]
                    for f in select_fields
                    if f in ent_spec.fields
                }
            else:
                projection = dict(ent_spec.fields)

            if not projection:
                raise ValidationError(f"No mapped columns available for entity '{entity}'.")

            # Enforce caller customer ownership on explicit customer filter BEFORE querying
            if authorized_customer_id and "customer_id" in filters and filters["customer_id"] is not None:
                if str(filters["customer_id"]).strip().lower() != str(authorized_customer_id).strip().lower():
                    raise PermissionDeniedError(
                        "Unauthorized: caller cannot query records belonging to another customer account."
                    )

            query_filters: List[QueryFilter] = []
            ops = filter_operators or {}
            for canonical_key, filter_val in filters.items():
                if filter_val is None:
                    continue
                if canonical_key not in ent_spec.fields:
                    raise ValidationError(
                        f"Filter field '{canonical_key}' is not mapped on entity '{entity}'."
                    )
                op = ops.get(canonical_key, "eq")
                if op not in ("eq", "ilike", "in"):
                    op = "eq"
                query_filters.append(
                    QueryFilter(
                        column=ent_spec.fields[canonical_key],
                        operator=op,
                        value=filter_val,
                        table=ent_spec.table,
                    )
                )

            resolved_order_by = None
            if order_by and order_by in ent_spec.fields:
                resolved_order_by = ent_spec.fields[order_by]

            joins: List[QueryJoin] = []
            if "status" in ent_spec.relations:
                status_rel = ent_spec.relations["status"]
                status_spec = mapping.entities.get(status_rel.target_entity)
                if status_spec:
                    status_proj: Dict[str, str] = {}
                    if "phase_name" in status_spec.fields and "status_name" not in projection:
                        status_proj["status_name"] = status_spec.fields["phase_name"]
                    if "phase_type" in status_spec.fields and "phase_type" not in projection:
                        status_proj["phase_type"] = status_spec.fields["phase_type"]
                    if status_proj:
                        joins.append(
                            QueryJoin(
                                right_table=status_spec.table,
                                right_schema=status_spec.schema_name,
                                left_column=status_rel.local_column,
                                right_column=status_rel.target_column,
                                select_columns=status_proj,
                            )
                        )
            if include_related_customer and "customer" in ent_spec.relations:
                rel = ent_spec.relations["customer"]
                cust_spec = mapping.entities.get(rel.target_entity)
                if cust_spec:
                    cust_proj: Dict[str, str] = {}
                    for cf in ("name", "phone", "email", "tier", "customer_id"):
                        if cf in cust_spec.fields and cf not in projection:
                            out_key = "customer_name" if cf == "name" else cf
                            cust_proj[out_key] = cust_spec.fields[cf]
                    if cust_proj:
                        joins.append(
                            QueryJoin(
                                right_table=cust_spec.table,
                                right_schema=cust_spec.schema_name,
                                left_column=rel.local_column,
                                right_column=rel.target_column,
                                select_columns=cust_proj,
                            )
                        )

            qspec = QuerySpec(
                table=ent_spec.table,
                schema_name=ent_spec.schema_name,
                select_columns=projection,
                filters=query_filters,
                joins=joins,
                order_by=resolved_order_by,
                descending=descending,
                limit=max(1, min(int(limit), 50)),
            )

            adapter = await self.manager.get_adapter(self._to_connection_spec(cfg))
            qres: QueryResult = await adapter.execute_read_only(
                qspec, timeout_seconds=timeout_seconds, max_rows=50
            )
            rows = list(qres.records)
            self.manager.metrics.record_query(tenant_id, qres.latency_ms, ok=True)

            # Enforce caller/customer authorization BEFORE PII masking strips raw values
            rows = enforce_customer_authorization(
                rows,
                authorized_customer_id=authorized_customer_id,
                authorized_phone=authorized_phone,
            )

            if mask_pii:
                rows = [mask_record_pii(r) for r in rows]

            return rows

        except PermissionDeniedError as exc:
            audit_status = "denied"
            err_msg = str(exc)
            self.manager.metrics.record_query(
                tenant_id, (time.perf_counter() - t0) * 1000.0, ok=False, rejected=True
            )
            raise
        except Exception as exc:
            audit_status = "timeout" if "timeout" in str(exc).lower() else "error"
            err_msg = str(exc)
            self.manager.metrics.record_query(
                tenant_id, (time.perf_counter() - t0) * 1000.0, ok=False
            )
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            masked_args = mask_record_pii(
                {k: v for k, v in filters.items() if v is not None}
            )
            try:
                audit = ToolExecutionAudit(
                    tenant_id=tenant_id,
                    config_id=cfg.id if cfg else None,
                    tool_name=tool_name,
                    database_engine=cfg.engine if cfg else None,
                    caller_subject=caller_subject,
                    arguments_masked=masked_args,
                    status=audit_status,
                    latency_ms=elapsed_ms,
                    row_count=len(rows) if audit_status == "success" else 0,
                    error_message=err_msg,
                )
                self.session.add(audit)
                await self.session.commit()
            except Exception:
                await self.session.rollback()

