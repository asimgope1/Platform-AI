import time
from collections import defaultdict
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from platform_core.errors.api_errors import create_error_response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window in-memory rate limiter per client IP or Authorization header."""

    def __init__(self, app, max_requests_per_minute: int = 120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.clients: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip health endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_key = request.headers.get("authorization") or request.client.host if request.client else "default"
        now = time.time()
        window_start = now - 60.0

        # Filter out timestamps outside window
        timestamps = [ts for ts in self.clients[client_key] if ts > window_start]
        if len(timestamps) >= self.max_requests:
            return create_error_response(
                message="Rate limit exceeded. Maximum 120 requests per minute.",
                code="rate_limit_exceeded",
                status_code=429,
            )

        timestamps.append(now)
        self.clients[client_key] = timestamps

        return await call_next(request)
