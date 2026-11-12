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

    # Start persistent PBX WebSocket connector (dials out to Asterisk)
    pbx_ws_url = settings.PBX_WS_URL or os.getenv("PBX_WS_URL", "")
    if pbx_ws_url:
        pbx_connector.start()
        logger.info(f"PBX connector started → {pbx_ws_url}")
    else:
        logger.info("PBX_WS_URL not set — PBX connector disabled (set it in .env to enable)")

    # Pre-warm IVR prompts and Whisper STT in background so incoming callers experience 0ms latency
    try:
        from audio.service import AudioService
        from telephony.ivr_audio_cache import warm_up_ivr_prompts
        audio_service = AudioService()
        asyncio.create_task(warm_up_ivr_prompts(audio_service))
        logger.info("[IVR] Audio prompt cache warmup initiated in background.")

        stt_adapter = provider_registry.get_stt_adapter("default")
        if hasattr(stt_adapter, "warmup"):
            asyncio.create_task(asyncio.to_thread(stt_adapter.warmup))

        if "semantic-default" in getattr(provider_registry, "embedding_adapters", {}):
            emb_adapter = provider_registry.get_embedding_adapter("semantic-default")
            if hasattr(emb_adapter, "warmup"):
                asyncio.create_task(asyncio.to_thread(emb_adapter.warmup))
                logger.info("[IVR] Semantic embedding model warmup initiated in background.")
    except Exception as cache_err:
        logger.warning(f"[IVR] Failed to initiate audio cache warmup: {cache_err}")

    yield

    # Shutdown
    pbx_connector.stop()
    try:
        from data_access.connectors.connection_manager import connection_manager
        await connection_manager.close_all()
    except Exception as pool_err:
        logger.warning(f"Failed disposing external DB pools on shutdown: {pool_err}")
    logger.info("AI Platform API service shutting down.")
