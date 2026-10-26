from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class NormalizedChatRequest(BaseModel):
    model_alias: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class NormalizedChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class NormalizedChatResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: List[NormalizedChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None


class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: Optional[str] = None


class StreamChunk(BaseModel):
    id: str
    model: str
    created: int
    choices: List[StreamChoice]
