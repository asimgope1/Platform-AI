import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from platform_core.observability.logging import request_id_ctx


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns or propagates X-Request-ID header on every HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        token = request_id_ctx.set(req_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = req_id
            return response
        finally:
            request_id_ctx.reset(token)
