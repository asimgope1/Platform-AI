import inspect
import json
import asyncio
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from data_access.security import sanitize_tool_argument
from tools.schemas import ToolExecutionResult
from tools.registry import tool_registry
from platform_core.security.tenant_context import TenantContext
from platform_core.security.permissions import check_permission
from platform_core.errors.exceptions import ToolExecutionError

_RESERVED_INTERNAL_ARGS = frozenset(
    {
        "tenant_context",
        "db_session",
        "tenant_id",
        "authorized_customer_id",
        "authorized_phone",
        "call_uuid",
        "caller_number",
        "service_category",
        "source_channel",
    }
)
_LONG_TEXT_ARG_NAMES = frozenset(
    {
        "description",
        "subject",
        "note",
        "notes",
        "reason",
        "summary",
        "resolution_notes",
        "desired_resolution",
        "product_reference",
        "query",
        "keyword",
        "path",
    }
)


def _sanitize_nested_payload(value: Any, name: str, max_len: int = 500) -> Any:
    if isinstance(value, str):
        return sanitize_tool_argument(value, name, required=False, max_len=max_len)
    if isinstance(value, dict):
        return {k: _sanitize_nested_payload(v, f"{name}.{k}", max_len=max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nested_payload(item, f"{name}[]", max_len=max_len) for item in value[:50]]
    return value


class ToolExecutor:
    """Safely executes named tools with parameter validation, permission checks, and timeouts."""

    def __init__(self, timeout_seconds: float = 10.0, db_session: Optional[AsyncSession] = None):
        self.timeout_seconds = timeout_seconds
        self.db_session = db_session

    async def execute_tool(
        self,
        call_id: str,
        name: str,
        arguments_json: str,
        context: Optional[TenantContext] = None,
        trusted_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        tool_defn = tool_registry.get_tool(name)
        if not tool_defn:
            return ToolExecutionResult(
                call_id=call_id,
                tool_name=name,
                output=None,
                is_error=True,
                error_message=f"Tool '{name}' is not registered or permitted.",
            )

        if context and tool_defn.required_scope:
            try:
                check_permission(context, tool_defn.required_scope)
            except Exception as e:
                return ToolExecutionResult(
                    call_id=call_id,
                    tool_name=name,
                    output=None,
                    is_error=True,
                    error_message=str(e),
                )

        handler = tool_registry.get_handler(name)
        if not handler:
            return ToolExecutionResult(
                call_id=call_id,
                tool_name=name,
                output=None,
                is_error=True,
                error_message=f"No execution handler registered for tool '{name}'.",
            )

        # Parse JSON arguments
        kwargs: Dict[str, Any] = {}
        if arguments_json and arguments_json.strip():
            try:
                parsed = json.loads(arguments_json)
                if not isinstance(parsed, dict):
                    return ToolExecutionResult(
                        call_id=call_id,
                        tool_name=name,
                        output=None,
                        is_error=True,
                        error_message="Tool arguments must be a JSON object.",
                    )
                kwargs = parsed
            except Exception as e:
                return ToolExecutionResult(
                    call_id=call_id,
                    tool_name=name,
                    output=None,
                    is_error=True,
                    error_message=f"Invalid JSON arguments: {e}",
                )

        # Strip reserved internal arguments so LLM/prompt injection cannot spoof tenant_id or customer auth
        kwargs = {k: v for k, v in kwargs.items() if k not in _RESERVED_INTERNAL_ARGS}

        # Validate required arguments & sanitize string values against SQLi/prompt injection
        schema_params = tool_defn.function.parameters or {}
        allowed_props = set((schema_params.get("properties") or {}).keys())
        required_props = list(schema_params.get("required") or [])
        # Normalize common schema alias pairs across local and remote Django schemas
        if "order_reference" in allowed_props and "order_reference" not in kwargs:
            if "order_id" in kwargs:
                kwargs["order_reference"] = kwargs.pop("order_id")
            elif "order_number" in kwargs:
                kwargs["order_reference"] = kwargs.pop("order_number")
        elif "order_number" in allowed_props and "order_id" not in allowed_props and "order_id" in kwargs and "order_number" not in kwargs:
            kwargs["order_number"] = kwargs.pop("order_id")
        elif "order_id" in allowed_props and "order_number" not in allowed_props and "order_number" in kwargs and "order_id" not in kwargs:
            kwargs["order_id"] = kwargs.pop("order_number")

        if "complaint_reference" in allowed_props and "ticket_number" not in allowed_props and "ticket_number" in kwargs and "complaint_reference" not in kwargs:
            kwargs["complaint_reference"] = kwargs.pop("ticket_number")
        elif "ticket_number" in allowed_props and "complaint_reference" not in allowed_props and "complaint_reference" in kwargs and "ticket_number" not in kwargs:
            kwargs["ticket_number"] = kwargs.pop("complaint_reference")
        if "customer_reference" in allowed_props and "customer_id" not in allowed_props and "customer_id" in kwargs and "customer_reference" not in kwargs:
            kwargs["customer_reference"] = kwargs.pop("customer_id")
        elif "customer_id" in allowed_props and "customer_reference" not in allowed_props and "customer_reference" in kwargs and "customer_id" not in kwargs:
            kwargs["customer_id"] = kwargs.pop("customer_reference")
        if allowed_props:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed_props}

        try:
            for req in required_props:
                if req not in kwargs or kwargs[req] is None or str(kwargs[req]).strip() == "":
                    return ToolExecutionResult(
                        call_id=call_id,
                        tool_name=name,
                        output=None,
                        is_error=True,
                        error_message=f"Missing required argument '{req}'.",
                    )
            for arg_name, arg_val in list(kwargs.items()):
                if isinstance(arg_val, str):
                    max_allowed = 1000 if arg_name in _LONG_TEXT_ARG_NAMES else 128
                    kwargs[arg_name] = sanitize_tool_argument(
                        arg_val, arg_name, required=(arg_name in required_props), max_len=max_allowed
                    )
                elif isinstance(arg_val, (dict, list)):
                    kwargs[arg_name] = _sanitize_nested_payload(arg_val, arg_name)
        except Exception as val_err:
            return ToolExecutionResult(
                call_id=call_id,
                tool_name=name,
                output=None,
                is_error=True,
                error_message=str(val_err),
            )

        # Inject trusted runtime context ONLY when declared on the handler signature
        sig_params = inspect.signature(handler).parameters
        if "tenant_context" in sig_params:
            kwargs["tenant_context"] = context
        if "db_session" in sig_params and self.db_session is not None:
            kwargs["db_session"] = self.db_session
        if "call_id" in sig_params:
            kwargs["call_id"] = call_id
        if trusted_metadata:
            for meta_key in ("authorized_customer_id", "call_uuid", "caller_number", "service_category", "source_channel"):
                if meta_key in sig_params and trusted_metadata.get(meta_key):
                    kwargs[meta_key] = trusted_metadata[meta_key]

        try:
            output = await asyncio.wait_for(handler(**kwargs), timeout=self.timeout_seconds)
            return ToolExecutionResult(
                call_id=call_id,
                tool_name=name,
                output=output,
                is_error=False,
            )
        except asyncio.TimeoutError:
            return ToolExecutionResult(
                call_id=call_id,
                tool_name=name,
                output=None,
                is_error=True,
                error_message=f"Tool execution timed out after {self.timeout_seconds}s.",
            )
        except Exception as e:
            return ToolExecutionResult(
                call_id=call_id,
                tool_name=name,
                output=None,
                is_error=True,
                error_message=str(e),
            )
