from platform_core.contracts.requests import (
    NormalizedChatRequest,
    ChatMessage,
    NormalizedChatResponse,
    NormalizedChoice,
    UsageInfo,
)
from orchestration.response_formatter import ResponseFormatter


def test_response_formatter_openai_structure():
    formatter = ResponseFormatter()
    norm_resp = NormalizedChatResponse(
        id="chatcmpl-test-123",
        created=1710000000,
        model="mock-chat",
        choices=[
            NormalizedChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello! How can I assist you?"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=8, total_tokens=18),
    )

    formatted = formatter.format_chat_response(norm_resp)
    assert formatted["id"] == "chatcmpl-test-123"
    assert formatted["object"] == "chat.completion"
    assert len(formatted["choices"]) == 1
    assert formatted["choices"][0]["message"]["role"] == "assistant"
    assert formatted["choices"][0]["message"]["content"] == "Hello! How can I assist you?"
    assert formatted["usage"]["total_tokens"] == 18
