from typing import Dict, Any
from platform_core.contracts.requests import NormalizedChatResponse


class ResponseFormatter:
    """Formats normalized response to OpenAI-compatible dict."""

    def format_chat_response(self, response: NormalizedChatResponse) -> Dict[str, Any]:
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        **({"tool_calls": [tc.model_dump() for tc in c.message.tool_calls]} if c.message.tool_calls else {}),
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in response.choices
            ],
            "usage": response.usage.model_dump(),
            **({"system_fingerprint": response.system_fingerprint} if response.system_fingerprint else {}),
        }
