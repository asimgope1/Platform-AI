import logging
import sys
import json
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variable for tracking request ID across async tasks
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id_ctx", default=None)
tenant_id_ctx: ContextVar[Optional[str]] = ContextVar("tenant_id_ctx", default=None)


class JSONLogFormatter(logging.Formatter):
    """Formats logs as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "tenant_id": tenant_id_ctx.get(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_entry.update(record.extra)
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLogFormatter())

    # Avoid duplicate handlers
    root_logger.handlers = [handler]


logger = logging.getLogger("ai_platform")
