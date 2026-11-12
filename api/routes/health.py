import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from platform_core.config.settings import settings
from storage.database import get_db_session
from platform_core.observability.metrics import metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db_session)):
    """Health and readiness check verifying database connectivity, remote tools, and operational status."""
    from tools.remote.registry_sync import get_remote_sync_state
    db_status = "healthy"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    remote_state = get_remote_sync_state()
    return {
        "status": "ok" if "unhealthy" not in db_status else "degraded",
        "timestamp": int(time.time()),
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": db_status,
            "remote_tools": {
                "enabled": settings.REMOTE_TOOLS_ENABLED,
                "synced": bool(remote_state.get("last_synced_at")) and not bool(remote_state.get("last_error")),
                "stale": bool(remote_state.get("stale")),
                "tool_count": int(remote_state.get("tool_count") or 0),
                "last_synced_at": remote_state.get("last_synced_at"),
            },
        },
        "metrics": metrics.get_summary(),
    }
