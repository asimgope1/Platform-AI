from typing import Dict, Any, Callable, Awaitable, List, Optional
from tools.schemas import ToolDefinition, ToolFunctionSchema
from tools.business_data.reports import count_open_purchase_requests
from tools.business_data.customers import get_customer_info


class ToolRegistry:
    """Registry holding allowlisted tools with their schemas and callable functions."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Awaitable[Any]],
        required_scope: str = "tools:execute",
    ) -> None:
        defn = ToolDefinition(
            function=ToolFunctionSchema(
                name=name,
                description=description,
                parameters=parameters,
            ),
            required_scope=required_scope,
        )
        self._tools[name] = defn
        self._handlers[name] = handler

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable[..., Awaitable[Any]]]:
        return self._handlers.get(name)

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": t.type,
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                },
            }
            for t in self._tools.values()
        ]


tool_registry = ToolRegistry()

# Register default safe read-only tools
tool_registry.register(
    name="count_open_purchase_requests",
    description="Returns the total count of open purchase requests across departments.",
    parameters={
        "type": "object",
        "properties": {
            "filter_status": {
                "type": "string",
                "enum": ["open", "pending", "closed", "all"],
                "description": "Status filter",
            }
        },
        "required": [],
    },
    handler=count_open_purchase_requests,
)

tool_registry.register(
    name="get_customer_info",
    description="Returns business details and active license seats for a specific customer ID.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Unique identifier of the customer account",
            }
        },
        "required": ["customer_id"],
    },
    handler=get_customer_info,
)
