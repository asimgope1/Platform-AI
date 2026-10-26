import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Dict, Any
from platform_core.observability.logging import logger


@asynccontextmanager
async def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
    """Lightweight tracing span context manager recording duration and logs."""
    start_time = time.perf_counter()
    span_data: Dict[str, Any] = {"name": name, "attributes": attributes or {}}
    logger.debug(f"[Trace Start] {name}", extra={"span": span_data})
    try:
        yield span_data
    except Exception as exc:
        span_data["error"] = str(exc)
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        span_data["duration_ms"] = round(duration_ms, 2)
        logger.debug(f"[Trace End] {name} completed in {duration_ms:.2f}ms", extra={"span": span_data})
