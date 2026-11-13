"""
Auto-bootstrap a tenant's live external CRM database from environment settings.

When CRM_DB_ENABLED=true and CRM_DB_PASSWORD is provided in .env, registers (or
refreshes) the TenantDatabaseConfig, discovers its schema, auto-maps canonical
entities (Customers, ComplainBookings, Products, etc.), and activates it so
UniversalDataAccessService tools query the live CRM DB automatically.
"""
from __future__ import annotations

from sqlalchemy import select

from data_access.connectors.base import DiscoveredSchema
from data_access.entity_mapping import (
    EntityMappingSpec,
    EntityRelationSpec,
    suggest_entity_mapping,
)
from data_access.models import TenantDatabaseConfig
from data_access.service import UniversalDataAccessService
from platform_core.config.settings import Settings
from platform_core.observability.logging import logger
from storage.database import async_session
from storage.models.tenant import Tenant


async def bootstrap_crm_database(cfg: Settings) -> None:
    """Register, discover, auto-map, and activate CRM_DB_* when enabled and configured."""
    if not getattr(cfg, "CRM_DB_ENABLED", False):
        return

    username = (getattr(cfg, "CRM_DB_USERNAME", None) or "").strip()
    password = getattr(cfg, "CRM_DB_PASSWORD", None) or ""
    if not username or not password:
        logger.info(
            "[CRM_DB] CRM_DB_ENABLED=true but CRM_DB_USERNAME or CRM_DB_PASSWORD is empty; "
            "skipping direct CRM DB registration."
        )
        return

    async with async_session() as session:
        tenant_id = (getattr(cfg, "CRM_DB_TENANT_ID", None) or "").strip()
        if not tenant_id:
            res = await session.execute(
                select(Tenant.id)
                .where(Tenant.is_active.is_(True), Tenant.name == "tenant-ivr-support")
                .limit(1)
            )
            tenant_id = res.scalar_one_or_none() or ""
        if not tenant_id:
            res = await session.execute(select(Tenant.id).where(Tenant.is_active.is_(True)).limit(1))
            tenant_id = res.scalar_one_or_none() or ""
        if not tenant_id:
            logger.warning("[CRM_DB] No active tenant found; skipping direct CRM DB bootstrap.")
            return

        existing_res = await session.execute(
            select(TenantDatabaseConfig).where(
                TenantDatabaseConfig.tenant_id == tenant_id,
                TenantDatabaseConfig.name == cfg.CRM_DB_NAME,
            )
        )
        existing = existing_res.scalar_one_or_none()

        uda = UniversalDataAccessService(session)
        if existing is None:
            config = await uda.register_config(
                tenant_id=tenant_id,
                name=cfg.CRM_DB_NAME,
                engine=cfg.CRM_DB_ENGINE,
                database_name=cfg.CRM_DB_NAME,
                host=cfg.CRM_DB_HOST,
                port=int(cfg.CRM_DB_PORT),
                username=username,
                password=password,
                schema_name=cfg.CRM_DB_SCHEMA,
                connection_options={"sslmode": "disable"},
                read_only=bool(cfg.CRM_DB_READ_ONLY),
            )
        else:
            config = existing

        # Discover schema & auto-suggest entity mapping if not yet active/mapped/validated
        # or if ComplainPhaseMaster is discovered but complaint_phase is not yet mapped.
        existing_entities = (config.schema_mapping or {}).get("entities", {})
        needs_phase_upgrade = "complaint_phase" not in existing_entities
        if not config.is_active or not config.schema_mapping or not config.mapping_validated or needs_phase_upgrade:
            if config.discovered_schema:
                schema = DiscoveredSchema(**config.discovered_schema)
            else:
                schema = await uda.discover_and_save_schema(tenant_id, config.id)
            table_names = {t.name for t in schema.tables}
            if (
                not config.schema_mapping
                or not config.mapping_validated
                or ("ComplainPhaseMaster" in table_names and needs_phase_upgrade)
            ):
                suggested = suggest_entity_mapping(schema)
                if "Customers" in table_names and (
                    "customer" not in suggested.entities
                    or "customer_id" not in suggested.entities["customer"].fields
                ):
                    suggested.entities["customer"] = EntityMappingSpec(
                        table="Customers",
                        schema_name=cfg.CRM_DB_SCHEMA,
                        primary_key="customers_id",
                        fields={
                            "customer_id": "customers_id",
                            "name": "name",
                            "phone": "phonenumber",
                            "email": "email",
                        },
                    )
                if "ComplainPhaseMaster" in table_names and "complaint_phase" not in suggested.entities:
                    suggested.entities["complaint_phase"] = EntityMappingSpec(
                        table="ComplainPhaseMaster",
                        schema_name=cfg.CRM_DB_SCHEMA,
                        primary_key="complain_phase_sl",
                        fields={
                            "phase_id": "complain_phase_sl",
                            "phase_name": "complain_phase_name",
                            "phase_type": "phase_type",
                            "is_initial": "initial",
                        },
                    )
                if "ComplainBookings" in table_names and (
                    "support_ticket" not in suggested.entities
                    or "ticket_id" not in suggested.entities["support_ticket"].fields
                    or "ticket_status" not in suggested.entities["support_ticket"].fields
                ):
                    ticket_relations = {
                        "customer": EntityRelationSpec(
                            target_entity="customer",
                            local_column="customers_id",
                            target_column="customers_id",
                        )
                    }
                    if "complaint_phase" in suggested.entities:
                        ticket_relations["status"] = EntityRelationSpec(
                            target_entity="complaint_phase",
                            local_column="complain_phase_sl",
                            target_column="complain_phase_sl",
                        )
                    suggested.entities["support_ticket"] = EntityMappingSpec(
                        table="ComplainBookings",
                        schema_name=cfg.CRM_DB_SCHEMA,
                        primary_key="complainsl",
                        fields={
                            "ticket_id": "complainid",
                            "customer_id": "customers_id",
                            "subject": "othernotes",
                            "ticket_status": "complain_phase_sl",
                            "created_at": "complaindatetime",
                            "updated_at": "modifydate",
                        },
                        relations=ticket_relations,
                    )
                if "Products" in table_names and (
                    "product" not in suggested.entities
                    or "product_id" not in suggested.entities["product"].fields
                    or "name" not in suggested.entities["product"].fields
                ):
                    suggested.entities["product"] = EntityMappingSpec(
                        table="Products",
                        schema_name=cfg.CRM_DB_SCHEMA,
                        primary_key="product_sl",
                        fields={
                            "product_id": "product_code",
                            "sku": "product_code",
                            "name": "product_name",
                        },
                    )
                await uda.update_mapping(tenant_id, config.id, suggested)
            val = await uda.validate_mapping(tenant_id, config.id)
            if val.valid and not config.is_active:
                await uda.activate_config(tenant_id, config.id)
            elif not val.valid:
                logger.warning(f"[CRM_DB] Entity mapping validation failed: {val.errors}")

        logger.info(
            f"[CRM_DB] Direct CRM database '{cfg.CRM_DB_NAME}' "
            f"({cfg.CRM_DB_HOST}:{cfg.CRM_DB_PORT}) active for tenant {tenant_id}."
        )
