from typing import Dict, Any, Callable, Awaitable, List, Optional
from tools.schemas import ToolDefinition, ToolFunctionSchema
from tools.business_data.reports import count_open_purchase_requests
from tools.business_data.customers import get_customer_info
from tools.business_data.orders import get_order_status
from tools.business_data.universal_tools import (
    get_customer_complaints,
    get_customer_details,
    get_customer_order_summary,
    get_customer_orders,
    get_invoice_details,
    get_order_tracking,
    get_payment_status,
    search_customer,
    search_database_records,
    search_knowledge_documents,
)
from tools.business_data.complaints import (
    call_external_api,
    get_complaint_status,
    register_complaint,
)


class ToolRegistry:
    """Registry holding allowlisted tools with their schemas and callable functions."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._sources: Dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Awaitable[Any]],
        required_scope: str = "tools:execute",
        source: str = "local",
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
        self._sources[name] = source

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable[..., Awaitable[Any]]]:
        return self._handlers.get(name)

    def get_tool_source(self, name: str) -> str:
        return self._sources.get(name, "local")

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools_with_sources(self) -> Dict[str, str]:
        return dict(self._sources)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._handlers.pop(name, None)
        self._sources.pop(name, None)

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

tool_registry.register(
    name="get_order_status",
    description="Retrieves live order status, expected delivery, and fulfillment summary for a given order ID from the tenant's business database.",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order identifier or order reference number (e.g. 'ORD-1001').",
            },
            "customer_id": {
                "type": "string",
                "description": "Optional customer account identifier to scope ownership verification.",
            },
        },
        "required": ["order_id"],
    },
    handler=get_order_status,
)

tool_registry.register(
    name="search_customer",
    description="Searches customer accounts in the tenant business database by name, phone, email, customer ID, or search query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Customer ID or name search string."},
            "name": {"type": "string", "description": "Partial or full customer name."},
            "phone": {"type": "string", "description": "Customer phone number."},
            "email": {"type": "string", "description": "Customer email address."},
            "customer_id": {"type": "string", "description": "Exact customer identifier."},
            "limit": {"type": "integer", "description": "Maximum number of customer matches (1-25)."},
        },
        "required": [],
    },
    handler=search_customer,
)

tool_registry.register(
    name="get_customer_details",
    description="Retrieves a customer's profile details (tier, status, masked contact metadata) by customer ID or phone.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "Customer identifier."},
            "phone": {"type": "string", "description": "Verified customer phone number."},
        },
        "required": [],
    },
    handler=get_customer_details,
)

tool_registry.register(
    name="get_customer_orders",
    description="Lists orders placed by a verified customer account, optionally filtered by status.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "Customer identifier."},
            "order_status": {"type": "string", "description": "Optional order status filter."},
            "limit": {"type": "integer", "description": "Maximum orders to return (1-25)."},
        },
        "required": ["customer_id"],
    },
    handler=get_customer_orders,
)

tool_registry.register(
    name="get_order_tracking",
    description="Retrieves live carrier, tracking number, shipment status, and delivery ETA for an order or tracking ID.",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order number or ID."},
            "tracking_number": {"type": "string", "description": "Carrier tracking number / AWB."},
            "customer_id": {"type": "string", "description": "Optional customer ID for ownership verification."},
        },
        "required": [],
    },
    handler=get_order_tracking,
)

tool_registry.register(
    name="get_payment_status",
    description="Checks live payment transaction status, payment method, and amount for a payment ID, order ID, or customer ID.",
    parameters={
        "type": "object",
        "properties": {
            "payment_id": {"type": "string", "description": "Payment transaction identifier."},
            "order_id": {"type": "string", "description": "Order identifier linked to payment."},
            "customer_id": {"type": "string", "description": "Customer identifier."},
        },
        "required": [],
    },
    handler=get_payment_status,
)

tool_registry.register(
    name="get_invoice_details",
    description="Retrieves billing invoice status, total amount, issue date, and due date for an invoice number, ID, or order.",
    parameters={
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string", "description": "Invoice number / reference."},
            "invoice_id": {"type": "string", "description": "Internal invoice ID."},
            "order_id": {"type": "string", "description": "Order ID associated with the invoice."},
            "customer_id": {"type": "string", "description": "Customer identifier."},
        },
        "required": [],
    },
    handler=get_invoice_details,
)

tool_registry.register(
    name="get_customer_complaints",
    description="Retrieves a customer's complaint history with human-readable status names by customer ID or phone number.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "Customer unique identifier."},
            "phone": {"type": "string", "description": "Customer phone number (used to resolve customer_id when customer_id is not known)."},
            "limit": {"type": "integer", "description": "Maximum number of complaints to return (default 10).", "default": 10},
        },
        "required": [],
    },
    handler=get_customer_complaints,
)

tool_registry.register(
    name="get_customer_order_summary",
    description="Aggregates a customer's profile, total order count, status breakdown, total spend, and recent orders.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "Customer identifier."},
        },
        "required": ["customer_id"],
    },
    handler=get_customer_order_summary,
)

tool_registry.register(
    name="search_database_records",
    description="Runs a schema-validated read-only search across a canonical business entity (customer, order, payment, invoice, product, support_ticket, inventory, shipment).",
    parameters={
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "enum": [
                    "customer",
                    "order",
                    "payment",
                    "invoice",
                    "product",
                    "support_ticket",
                    "inventory",
                    "shipment",
                ],
                "description": "Canonical business entity to query.",
            },
            "query": {"type": "string", "description": "Optional search keyword mapped to entity's primary identifier/name."},
            "filters": {"type": "object", "description": "Optional canonical field equality filters."},
            "limit": {"type": "integer", "description": "Maximum records to return (1-25)."},
        },
        "required": ["entity"],
    },
    handler=search_database_records,
)

tool_registry.register(
    name="search_knowledge_documents",
    description="Searches tenant knowledge base documents and policy articles using hybrid semantic and lexical retrieval.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query."},
            "category": {"type": "string", "description": "Optional policy category filter."},
            "top_k": {"type": "integer", "description": "Maximum matching document chunks (1-10)."},
        },
        "required": ["query"],
    },
    handler=search_knowledge_documents,
)

tool_registry.register(
    name="register_complaint",
    description=(
        "Registers a customer complaint during a call or chat conversation, generates a unique "
        "complaint ticket number (CMP-...), persists it, and dispatches to the configured external CRM/Django API if available."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Clear description of the caller's complaint or issue.",
            },
            "category": {
                "type": "string",
                "enum": [
                    "order_delivery",
                    "damaged_item",
                    "payment_upi",
                    "refund_billing",
                    "card_account",
                    "service_quality",
                    "general",
                ],
                "description": "Complaint category.",
            },
            "subject": {"type": "string", "description": "Short summary title for the complaint."},
            "order_id": {"type": "string", "description": "Related order ID or order number if mentioned."},
            "customer_id": {"type": "string", "description": "Customer ID or reference if known."},
            "caller_phone": {"type": "string", "description": "Callback phone number if provided by caller."},
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "description": "Urgency level of the complaint.",
            },
        },
        "required": ["description"],
    },
    handler=register_complaint,
)

tool_registry.register(
    name="get_complaint_status",
    description="Checks the status of an existing customer complaint using ticket number (CMP-...), order ID, customer ID, or current call.",
    parameters={
        "type": "object",
        "properties": {
            "ticket_number": {"type": "string", "description": "Complaint ticket number (e.g., CMP-20261111-XXXX)."},
            "order_id": {"type": "string", "description": "Order ID associated with the complaint."},
            "customer_id": {"type": "string", "description": "Customer identifier."},
            "limit": {"type": "integer", "description": "Maximum complaints to return (1-20)."},
        },
        "required": [],
    },
    handler=get_complaint_status,
)

tool_registry.register(
    name="call_external_api",
    description=(
        "Calls a configured allowlisted backend HTTP API endpoint (such as remote Django or CRM webhook) "
        "during a conversation to fetch live data or trigger a backend workflow."
    ),
    parameters={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["remote_django", "complaint_api", "configured"],
                "description": "Configured external service alias to invoke.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH"],
                "description": "HTTP method to send.",
            },
            "path": {
                "type": "string",
                "description": "Relative endpoint path (e.g. '/api/v1/tools/', '/api/v1/complaints/') on the allowlisted service.",
            },
            "payload": {
                "type": "object",
                "description": "Optional JSON request body for POST/PUT/PATCH calls.",
            },
            "query_params": {
                "type": "object",
                "description": "Optional URL query parameters.",
            },
        },
        "required": ["path"],
    },
    handler=call_external_api,
)
