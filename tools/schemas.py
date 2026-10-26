from typing import Dict, Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[list] = None


class ToolFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunctionSchema
    required_scope: str = "tools:execute"


class ToolExecutionResult(BaseModel):
    call_id: str
    tool_name: str
    output: Any
    is_error: bool = False
    error_message: Optional[str] = None
