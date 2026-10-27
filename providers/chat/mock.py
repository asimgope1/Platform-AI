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

        # Context-aware intelligent mock responses
        lower_msg = last_msg.lower()
        if any(w in lower_msg for w in ["hi", "hello", "hey", "नमस्ते", "hii", "heya"]):
            reply_content = "Hello! How can I assist you today? You can ask me about UPI refunds, Video KYC requirements, order tracking, or test PostgreSQL RAG retrieval."
        elif any(w in lower_msg for w in ["refund", "upi", "रिफंड", "cut", "kat gaya", "paise"]):
            reply_content = "If your bank account was debited via UPI, the refund will be automatically credited back within 24 to 48 hours per banking regulations. You can track this using your 12-digit UTR reference."
        elif any(w in lower_msg for w in ["kyc", "केवाईसी", "video kyc", "pan"]):
            reply_content = "For Video KYC, please keep your physical PAN card and Aadhaar-linked mobile ready. The verification takes only 2 minutes and unlocks full account privileges within 1 hour."
        elif any(w in lower_msg for w in ["order", "delivery", "track", "ऑर्डर", "डिलीवरी"]):
            reply_content = "Your order has been processed and standard delivery takes 2 to 4 business days to arrive safely at your doorstep."
        else:
            reply_content = f"Hello! I have processed your inquiry: '{last_msg}'. How else can I assist you today?"

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

        last_msg = request.messages[-1].content if request.messages else ""
        if isinstance(last_msg, list):
            last_msg = str(last_msg)
        lower_msg = last_msg.lower()

        if any(w in lower_msg for w in ["hi", "hello", "hey", "नमस्ते", "hii", "heya"]):
            reply_content = "Hello! How can I assist you today? You can ask me about UPI refunds, Video KYC requirements, order tracking, or test PostgreSQL RAG retrieval."
        elif any(w in lower_msg for w in ["refund", "upi", "रिफंड", "cut", "kat gaya", "paise"]):
            reply_content = "If your bank account was debited via UPI, the refund will be automatically credited back within 24 to 48 hours per banking regulations. You can track this using your 12-digit UTR reference."
        elif any(w in lower_msg for w in ["kyc", "केवाईसी", "video kyc", "pan"]):
            reply_content = "For Video KYC, please keep your physical PAN card and Aadhaar-linked mobile ready. The verification takes only 2 minutes and unlocks full account privileges within 1 hour."
        elif any(w in lower_msg for w in ["order", "delivery", "track", "ऑर्डर", "डिलीवरी"]):
            reply_content = "Your order has been processed and standard delivery takes 2 to 4 business days to arrive safely at your doorstep."
        else:
            reply_content = f"Hello! I have processed your inquiry: '{last_msg}'. How else can I assist you today?"

        words = reply_content.split(" ")
        for i, word in enumerate(words):
            prefix = "" if i == 0 else " "
            await asyncio.sleep(0.015)
            yield StreamChunk(
                id=req_id,
                model=model_meta.alias,
                created=created_ts,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=prefix + word),
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
