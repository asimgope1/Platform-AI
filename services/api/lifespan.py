import os
import asyncio
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

    # Sync validated remote Django business tools into ToolRegistry if enabled
    if settings.REMOTE_TOOLS_ENABLED and settings.REMOTE_TOOLS_BASE_URL:
        try:
            from tools.remote.django_client import DjangoRemoteToolClient
            from tools.remote.registry_sync import sync_remote_django_tools

            remote_client = DjangoRemoteToolClient.from_settings(settings)
            synced = await sync_remote_django_tools(
                remote_client,
                prefer_remote=settings.REMOTE_TOOLS_PREFER_REMOTE,
            )
            logger.info(f"Synced {len(synced)} remote Django business tools from {settings.REMOTE_TOOLS_BASE_URL}.")
        except Exception as remote_err:
            logger.warning(f"Remote Django tool sync unavailable at startup (non-fatal): {remote_err}")

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

    # Optional: Auto-register, discover, and activate direct CRM DB from CRM_DB_* env vars
    if settings.CRM_DB_ENABLED:
        try:
            from data_access.bootstrap import bootstrap_crm_database
            await bootstrap_crm_database(settings)
        except Exception as crm_err:
            logger.warning(f"[CRM_DB] Direct CRM DB bootstrap skipped (non-fatal): {crm_err}")

    # Start persistent PBX WebSocket connector (dials out to Asterisk)
    pbx_ws_url = settings.PBX_WS_URL or os.getenv("PBX_WS_URL", "")
    if pbx_ws_url:
        pbx_connector.start()
        logger.info(f"PBX connector started → {pbx_ws_url}")
    else:
        logger.info("PBX_WS_URL not set — PBX connector disabled (set it in .env to enable)")

    # Autonomous AI Warmup Daemon: Pre-warms Whisper, TTS, Ollama Gemma (keep_alive=-1),
    # SentenceTransformers embeddings, pgvector DB pool, and starts continuous keep-alive heartbeat loop
    try:
        from services.ai_warmup_daemon import warmup_all_ai_components, start_ai_keepalive_loop
        asyncio.create_task(warmup_all_ai_components())
        start_ai_keepalive_loop()
        logger.info("[AI Warmup] Autonomous AI warmup & keep-alive daemon successfully launched.")
    except Exception as warmup_err:
        logger.warning(f"[AI Warmup] Failed to launch AI warmup daemon: {warmup_err}")

    yield

    # Shutdown
    try:
        from services.ai_warmup_daemon import stop_ai_keepalive_loop
        stop_ai_keepalive_loop()
    except Exception:
        pass

    pbx_connector.stop()
    try:
        from data_access.connectors.connection_manager import connection_manager
        await connection_manager.close_all()
    except Exception as pool_err:
        logger.warning(f"Failed disposing external DB pools on shutdown: {pool_err}")
    logger.info("AI Platform API service shutting down.")
