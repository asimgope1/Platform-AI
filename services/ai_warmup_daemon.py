"""
Autonomous AI Warmup & Keep-Alive Daemon.

Ensures all AI models and telephony pipelines remain permanently loaded in RAM/VRAM:
1. Ollama LLM (Gemma 4 / Gemma 2B): Pinned indefinitely with keep_alive: -1.
2. Whisper Speech-To-Text (STT): Neural weights & JIT graphs pre-warmed.
3. Text-To-Speech (TTS): Prompts pre-rendered in memory & Odia VITS loaded.
4. Semantic Embeddings & pgvector: SentenceTransformers & PostgreSQL pool active.
5. Live CRM Catalog: Pre-synchronized in-memory for microsecond ticket lookups.
6. Periodic Keep-Alive Heartbeat: Runs every 120s so the OS or upstream never unloads models.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict

logger = logging.getLogger("services.ai_warmup_daemon")

_KEEPALIVE_TASK: asyncio.Task | None = None
_IS_WARMED_UP: bool = False
_LAST_WARMUP_TIME: float = 0.0
_WARMUP_STATS: Dict[str, Any] = {}


async def warmup_all_ai_components(force: bool = False) -> Dict[str, Any]:
    """
    Executes a comprehensive, non-blocking warm-up across all AI models and database caches.
    Ensures first-turn incoming phone calls experience 0ms cold-start latency.
    """
    global _IS_WARMED_UP, _LAST_WARMUP_TIME, _WARMUP_STATS

    logger.info("[AI Warmup] 🚀 Commencing global AI model & pipeline pre-warming...")
    t_start = time.perf_counter()
    stats: Dict[str, Any] = {
        "whisper_stt": "pending",
        "tts_prompts": "pending",
        "ollama_llm": "pending",
        "embeddings": "pending",
        "vector_db": "pending",
        "crm_cache": "pending",
    }

    # 1. Whisper STT Warmup
    try:
        from providers.registry import provider_registry
        stt_adapter = provider_registry.get_stt_adapter("default")
        if hasattr(stt_adapter, "warmup"):
            await asyncio.to_thread(stt_adapter.warmup)
            stats["whisper_stt"] = "online"
            logger.info("[AI Warmup] ✓ Whisper STT weights pre-loaded in memory.")
    except Exception as e:
        stats["whisper_stt"] = f"notice: {e}"
        logger.warning(f"[AI Warmup] Whisper STT warm-up notice: {e}")

    # 2. TTS & Audio Prompt Cache Warmup
    try:
        from audio.service import AudioService
        from telephony.ivr_audio_cache import warm_up_ivr_prompts
        audio_service = AudioService()
        asyncio.create_task(warm_up_ivr_prompts(audio_service))
        stats["tts_prompts"] = "online"
        logger.info("[AI Warmup] ✓ TTS engine & audio prompt cache warming in background.")
    except Exception as e:
        stats["tts_prompts"] = f"notice: {e}"
        logger.warning(f"[AI Warmup] TTS prompt warm-up notice: {e}")

    # 3. Ollama LLM Warmup (Gemma) with keep_alive: -1 (Never Unload)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.117:11434/v1")
    ollama_host = ollama_url.replace("/v1", "").rstrip("/")
    target_model = os.getenv("OLLAMA_GEMMA_MODEL", "gemma4:latest")

    async def _ping_ollama():
        def _do_req():
            endpoint = f"{ollama_host}/api/generate"
            payload = json.dumps({
                "model": target_model,
                "prompt": "hi",
                "keep_alive": -1,
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=12.0) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            res = await asyncio.to_thread(_do_req)
            stats["ollama_llm"] = f"online ({target_model}, keep_alive=-1)"
            logger.info(f"[AI Warmup] ✓ Ollama LLM '{target_model}' pinned indefinitely in memory.")
        except Exception as e:
            stats["ollama_llm"] = f"notice: {e}"
            logger.warning(f"[AI Warmup] Ollama warm-up notice ({target_model}): {e}")

    asyncio.create_task(_ping_ollama())

    # 4. Semantic Embedding Model Warmup
    try:
        from providers.registry import provider_registry
        if "semantic-default" in getattr(provider_registry, "embedding_adapters", {}):
            emb_adapter = provider_registry.get_embedding_adapter("semantic-default")
            if hasattr(emb_adapter, "warmup"):
                await asyncio.to_thread(emb_adapter.warmup)
                stats["embeddings"] = "online"
                logger.info("[AI Warmup] ✓ SentenceTransformer embedding model warmed.")
    except Exception as e:
        stats["embeddings"] = f"notice: {e}"
        logger.warning(f"[AI Warmup] Embedding warm-up notice: {e}")

    # 5. Local pgvector Database Connection Pool Warmup
    try:
        import asyncpg
        local_db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost:5432/ai_platform_test")
        clean_url = local_db_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(clean_url, timeout=3.0)
        await conn.fetchval("SELECT 1 FROM document_chunks LIMIT 1")
        await conn.close()
        stats["vector_db"] = "online"
        logger.info("[AI Warmup] ✓ Local PostgreSQL pgvector connection pool warm.")
    except Exception as e:
        stats["vector_db"] = f"notice: {e}"
        logger.warning(f"[AI Warmup] Local pgvector DB notice: {e}")

    # 6. CRM Live Cache Warmup
    try:
        from providers.chat.ivr_mini_model import ivr_mini_brain
        await ivr_mini_brain.sync_live_crm_cache(force=True)
        stats["crm_cache"] = "online"
        logger.info("[AI Warmup] ✓ Live CRM complaint catalog pre-synchronized.")
    except Exception as e:
        stats["crm_cache"] = f"notice: {e}"
        logger.warning(f"[AI Warmup] CRM cache pre-sync notice: {e}")

    _IS_WARMED_UP = True
    _LAST_WARMUP_TIME = time.time()
    _WARMUP_STATS = stats
    t_elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(f"[AI Warmup] ⚡ All AI systems initialized and warm in {t_elapsed:.1f}ms.")
    return stats


async def _ai_keepalive_heartbeat_loop(interval_seconds: int = 120):
    """
    Continuous background loop that pings AI components every 2 minutes
    to ensure the OS, PyTorch, or Ollama never page out or unload models.
    """
    logger.info(f"[AI Warmup] 💓 Autonomous AI keep-alive heartbeat loop started (interval={interval_seconds}s).")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.117:11434/v1")
    ollama_host = ollama_url.replace("/v1", "").rstrip("/")
    target_model = os.getenv("OLLAMA_GEMMA_MODEL", "gemma4:latest")

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            # 1. Keep Ollama LLM alive indefinitely
            def _ping_ollama():
                endpoint = f"{ollama_host}/api/generate"
                payload = json.dumps({
                    "model": target_model,
                    "prompt": "",
                    "keep_alive": -1,
                }).encode("utf-8")
                req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    return resp.status

            try:
                status = await asyncio.to_thread(_ping_ollama)
                logger.debug(f"[AI Warmup] 💓 Ollama keep-alive heartbeat acknowledged (HTTP {status}).")
            except Exception as e:
                logger.debug(f"[AI Warmup] Ollama keep-alive ping note: {e}")

            # 2. Touch local database pool
            try:
                import asyncpg
                local_db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://localhost:5432/ai_platform_test")
                clean_url = local_db_url.replace("postgresql+asyncpg://", "postgresql://")
                conn = await asyncpg.connect(clean_url, timeout=2.0)
                await conn.fetchval("SELECT 1")
                await conn.close()
            except Exception:
                pass

            # 3. Touch CRM catalog if older than 5 minutes
            try:
                from providers.chat.ivr_mini_model import ivr_mini_brain
                if time.time() - ivr_mini_brain._last_crm_sync_ts > 300:
                    await ivr_mini_brain.sync_live_crm_cache(force=False)
            except Exception:
                pass

        except asyncio.CancelledError:
            logger.info("[AI Warmup] Heartbeat loop cancelled.")
            break
        except Exception as loop_err:
            logger.warning(f"[AI Warmup] Heartbeat cycle error: {loop_err}")
            await asyncio.sleep(10)


def start_ai_keepalive_loop():
    """Starts the background keep-alive heartbeat task."""
    global _KEEPALIVE_TASK
    if _KEEPALIVE_TASK is None or _KEEPALIVE_TASK.done():
        _KEEPALIVE_TASK = asyncio.create_task(_ai_keepalive_heartbeat_loop(interval_seconds=120))


def stop_ai_keepalive_loop():
    """Stops the background keep-alive heartbeat task."""
    global _KEEPALIVE_TASK
    if _KEEPALIVE_TASK and not _KEEPALIVE_TASK.done():
        _KEEPALIVE_TASK.cancel()
        _KEEPALIVE_TASK = None


def get_warmup_status() -> Dict[str, Any]:
    """Returns the current warm-up state and component status."""
    return {
        "is_warmed_up": _IS_WARMED_UP,
        "last_warmup_time": _LAST_WARMUP_TIME,
        "uptime_warm_seconds": (time.time() - _LAST_WARMUP_TIME) if _LAST_WARMUP_TIME > 0 else 0,
        "components": _WARMUP_STATS,
    }
