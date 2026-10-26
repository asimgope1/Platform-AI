from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse
from platform_core.errors.exceptions import PlatformError


def create_error_response(
    message: str,
    code: str,
    status_code: int = 400,
    param: Optional[str] = None,
    error_type: Optional[str] = None,
) -> JSONResponse:
    if not error_type:
        if status_code == 401:
            error_type = "authentication_error"
        elif status_code == 403:
            error_type = "permission_error"
        elif status_code == 404:
            error_type = "invalid_request_error"
        elif status_code == 429:
            error_type = "rate_limit_error"
        elif status_code >= 500:
            error_type = "api_error"
        else:
            error_type = "invalid_request_error"

    payload: Dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
    return create_error_response(
        message=exc.message,
        code=exc.code,
        status_code=exc.status_code,
        param=exc.param,
    )
