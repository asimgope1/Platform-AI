import time
import uuid
import json
import asyncio
from typing import AsyncIterator, Dict, Any, List
from platform_core.contracts.capabilities import ModelMetadata
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ChatMessage,
    UsageInfo,
    ToolCall,
    FunctionCall,
    StreamChunk,
    StreamChoice,
    StreamDelta,
)
from platform_core.contracts.providers import BaseChatAdapter


class MockChatAdapter(BaseChatAdapter):
    """Deterministic, high-performance mock chat adapter for tests, CI, and local verification."""

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        req_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
        created_ts = int(time.time())

        # Check if the user's last message is triggering a tool call
        last_msg = request.messages[-1].content if request.messages else ""
        if isinstance(last_msg, list):
            last_msg = str(last_msg)

        # If tools are provided and user asks for purchase requests / report, simulate tool call
        if request.tools and ("purchase request" in last_msg.lower() or "report" in last_msg.lower() or "customer" in last_msg.lower()):
            tool_name = "count_open_purchase_requests"
            if "customer" in last_msg.lower():
                tool_name = "get_customer_info"
            elif "report" in last_msg.lower():
                tool_name = "generate_summary_report"

            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=tool_name,
                    arguments=json.dumps({"filter_status": "open" if "open" in last_msg.lower() else "all"}),
                ),
            )
            return NormalizedChatResponse(
                id=req_id,
                model=model_meta.alias,
                created=created_ts,
                choices=[
                    NormalizedChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[tool_call],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageInfo(prompt_tokens=15, completion_tokens=20, total_tokens=35),
            )

        # Standard text reply
        reply_content = f"Mock response for [{model_meta.alias}]: Hello! I have received your message: '{last_msg}'."
        return NormalizedChatResponse(
            id=req_id,
            model=model_meta.alias,
            created=created_ts,
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=12, completion_tokens=18, total_tokens=30),
        )

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> AsyncIterator[StreamChunk]:
        req_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
        created_ts = int(time.time())

        # Yield role chunk
        yield StreamChunk(
            id=req_id,
            model=model_meta.alias,
            created=created_ts,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        words = ["Hello", " from", f" {model_meta.alias}", "!", " Streamed", " mock", " completion", " is", " working."]
        for word in words:
            await asyncio.sleep(0.01)
            yield StreamChunk(
                id=req_id,
                model=model_meta.alias,
                created=created_ts,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=word),
                        finish_reason=None,
                    )
                ],
            )

        # Final stop chunk
        yield StreamChunk(
            id=req_id,
            model=model_meta.alias,
            created=created_ts,
            choices=[
                StreamChoice(
                    index=0,
                    delta=StreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
