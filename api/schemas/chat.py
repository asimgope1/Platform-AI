from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ChatMessageSchema(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequestSchema(BaseModel):
    model: str
    messages: List[ChatMessageSchema]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None
    enable_rag: Optional[bool] = False


class ChoiceSchema(BaseModel):
    index: int
    message: ChatMessageSchema
    finish_reason: Optional[str] = "stop"


class UsageSchema(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseSchema(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChoiceSchema]
    usage: UsageSchema
    system_fingerprint: Optional[str] = None
