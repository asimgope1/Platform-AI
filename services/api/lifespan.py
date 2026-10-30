import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from platform_core.config.settings import settings, validate_master_key
from platform_core.observability.logging import setup_logging, logger
from platform_core.security.api_keys import generate_api_key, hash_api_key
from storage.database import init_db, async_session
from storage.models.tenant import Tenant
from storage.models.api_key import ApiKey
from providers.registry import provider_registry
from telephony.pbx_connector import pbx_connector
from telephony.pbx_connector import pbx_connector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate critical configuration (e.g., MASTER_API_KEY security in production)
    validate_master_key(settings)

    # Setup structured JSON logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting AI Platform API service...")


    # Initialize database tables
    await init_db()
    logger.info("Database initialized.")

    # Initialize Provider Registry
    if os.path.exists(settings.PROVIDERS_CONFIG_PATH):
        provider_registry.initialize_from_file(settings.PROVIDERS_CONFIG_PATH)
        logger.info(f"Loaded provider configurations from {settings.PROVIDERS_CONFIG_PATH}")
    else:
        logger.warning(f"Provider config not found at {settings.PROVIDERS_CONFIG_PATH}, using fallback defaults.")

    # Bootstrap default tenant and demo API key if table is empty
    async with async_session() as session:
        from sqlalchemy import select
        res = await session.execute(select(Tenant).limit(1))
        first_tenant = res.scalar_one_or_none()
        if not first_tenant:
            tenant = Tenant(name="Default Organization")
            session.add(tenant)
            await session.commit()
            await session.refresh(tenant)

            demo_key = "sk_live_demo_test_key_12345"
            key_hash = hash_api_key(demo_key)
            api_key = ApiKey(
                tenant_id=tenant.id,
                key_hash=key_hash,
                name="Demo Key",
                scopes="*",
            )
            session.add(api_key)
            await session.commit()
            logger.info(f"Bootstrapped default tenant '{tenant.id}' with demo key: {demo_key}")

    # Start persistent PBX WebSocket connector (dials out to Asterisk)
    pbx_ws_url = settings.PBX_WS_URL or os.getenv("PBX_WS_URL", "")
    if pbx_ws_url:
        pbx_connector.start()
        logger.info(f"PBX connector started → {pbx_ws_url}")
    else:
        logger.info("PBX_WS_URL not set — PBX connector disabled (set it in .env to enable)")

    yield

    # Shutdown
    pbx_connector.stop()
    logger.info("AI Platform API service shutting down.")
