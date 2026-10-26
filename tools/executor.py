import json
import asyncio
from typing import Dict, Any, Optional
from tools.schemas import ToolExecutionResult
from tools.registry import tool_registry
from platform_core.security.tenant_context import TenantContext
from platform_core.security.permissions import check_permission
from platform_core.errors.exceptions import ToolExecutionError


class ToolExecutor:
    """Safely executes named tools with parameter validation, permission checks, and timeouts."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    async def execute_tool(
        self,
        call_id: str,
        name: str,
        arguments_json: str,
        context: Optional[TenantContext] = None,
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
                kwargs = json.loads(arguments_json)
                if not isinstance(kwargs, dict):
                    kwargs = {}
            except Exception as e:
                return ToolExecutionResult(
                    call_id=call_id,
                    tool_name=name,
                    output=None,
                    is_error=True,
                    error_message=f"Invalid JSON arguments: {e}",
                )

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
