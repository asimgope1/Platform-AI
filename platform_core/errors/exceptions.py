from typing import Optional, Dict, Any


class PlatformError(Exception):
    """Base exception for all AI platform errors."""

    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
        param: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.param = param
        self.details = details or {}


class AuthenticationError(PlatformError):
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message=message, code="invalid_api_key", status_code=401)


class PermissionDeniedError(PlatformError):
    def __init__(self, message: str = "Permission denied for this operation or tenant"):
        super().__init__(message=message, code="permission_denied", status_code=403)


class ModelNotFoundError(PlatformError):
    def __init__(self, model_alias: str):
        super().__init__(
            message=f"The model '{model_alias}' does not exist or you do not have access to it.",
            code="model_not_found",
            status_code=404,
            param="model",
        )


class CapabilityNotSupportedError(PlatformError):
    def __init__(self, model_alias: str, capability: str):
        super().__init__(
            message=f"Model '{model_alias}' does not support required capability: '{capability}'.",
            code="capability_not_supported",
            status_code=400,
            param="capabilities",
        )


class ProviderError(PlatformError):
    def __init__(self, provider_id: str, message: str, status_code: int = 502):
        super().__init__(
            message=f"Upstream provider error [{provider_id}]: {message}",
            code="provider_error",
            status_code=status_code,
        )


class RateLimitExceededError(PlatformError):
    def __init__(self, message: str = "Rate limit exceeded. Please throttle your requests."):
        super().__init__(message=message, code="rate_limit_exceeded", status_code=429)


class ToolExecutionError(PlatformError):
    def __init__(self, tool_name: str, message: str):
        super().__init__(
            message=f"Failed to execute tool '{tool_name}': {message}",
            code="tool_execution_failed",
            status_code=400,
        )


class ValidationError(PlatformError):
    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(
            message=message,
            code="invalid_request_error",
            status_code=400,
            param=param,
        )
