import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from storage.database import get_db_session
from platform_core.observability.metrics import metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db_session)):
    """Health and readiness check verifying database connectivity and operational status."""
    db_status = "healthy"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    return {
        "status": "ok" if "unhealthy" not in db_status else "degraded",
        "timestamp": int(time.time()),
        "services": {
            "database": db_status,
        },
        "metrics": metrics.get_summary(),
    }
