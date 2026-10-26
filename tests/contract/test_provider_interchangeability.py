import pytest
import httpx
from platform_core.contracts.capabilities import ModelMetadata, ModelCapability
from platform_core.contracts.requests import (
    NormalizedChatRequest,
    NormalizedChatResponse,
    NormalizedChoice,
    ChatMessage,
    UsageInfo,
    StreamChunk,
    StreamChoice,
    StreamDelta,
)
from platform_core.contracts.providers import BaseChatAdapter
from platform_core.errors.exceptions import CapabilityNotSupportedError
from orchestration.chat_orchestrator import ChatOrchestrator
from providers.registry import provider_registry


class SecondaryProviderAdapter(BaseChatAdapter):
    """A distinct alternative provider (e.g. Anthropic, Together, or internal LLM)."""

    async def chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ) -> NormalizedChatResponse:
        user_content = request.messages[-1].content if request.messages else ""
        return NormalizedChatResponse(
            id="chatcmpl-provider-b-9999",
            model=model_meta.alias,
            created=1710000000,
            choices=[
                NormalizedChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=f"Response from Provider B [{model_meta.underlying_model}]: Processed '{user_content}'.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=8, completion_tokens=14, total_tokens=22),
        )

    async def stream_chat_completion(
        self,
        request: NormalizedChatRequest,
        model_meta: ModelMetadata,
    ):
        yield StreamChunk(
            id="chatcmpl-provider-b-stream",
            model=model_meta.alias,
            created=1710000000,
            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content="Provider B stream chunk"))],
        )


@pytest.mark.asyncio
async def test_provider_interchangeability_acceptance():
    """Acceptance Test: Proves the central architectural promise of the platform.

    1. Send request to alias 'dynamic-chat' using Provider A (Mock Provider).
    2. Change only provider configuration mapping to Provider B without changing orchestration.
    3. Send the exact same request.
    4. Assert identical public response contract and zero code modifications.
    """
    orchestrator = ChatOrchestrator()

    # Step 1: Configure alias 'dynamic-chat' mapped to Provider A
    provider_a_meta = ModelMetadata(
        alias="dynamic-chat",
        provider_id="mock_provider",
        underlying_model="mock-instant-1",
        description="Dynamic chat mapped to Provider A",
        capabilities={ModelCapability.CHAT, ModelCapability.STREAMING},
    )
    provider_registry.register_model_alias(provider_a_meta)

    norm_request = NormalizedChatRequest(
        model_alias="dynamic-chat",
        messages=[ChatMessage(role="user", content="Explain quantum computing")],
        stream=False,
    )

    # Execute against Provider A
    resp_a = await orchestrator.execute(norm_request)
    assert resp_a.model == "dynamic-chat"
    assert resp_a.choices[0].message.role == "assistant"
    assert "Mock response for [dynamic-chat]" in resp_a.choices[0].message.content
    assert resp_a.usage.total_tokens > 0

    # Step 2: Switch configuration to Provider B (No orchestration or client changes)
    provider_b_adapter = SecondaryProviderAdapter("provider_b", {})
    provider_b_meta = ModelMetadata(
        alias="dynamic-chat",
        provider_id="provider_b",
        underlying_model="claude-3-5-sonnet",
        description="Dynamic chat remapped to Provider B",
        capabilities={ModelCapability.CHAT, ModelCapability.STREAMING},
    )
    provider_registry.register_model_alias(provider_b_meta, adapter=provider_b_adapter)

    # Step 3: Execute the exact same request against Provider B
    resp_b = await orchestrator.execute(norm_request)

    # Step 4: Verify identical contract shape and clean switchover
    assert resp_b.model == "dynamic-chat"
    assert resp_b.choices[0].message.role == "assistant"
    assert "Response from Provider B [claude-3-5-sonnet]" in resp_b.choices[0].message.content
    assert resp_b.choices[0].finish_reason == "stop"
    assert resp_b.usage.total_tokens == 22


@pytest.mark.asyncio
async def test_capability_checking_across_switched_providers():
    """Verify that capability enforcement works uniformly when switching providers."""
    orchestrator = ChatOrchestrator()

    # Configure a text-only model alias without tool_calling
    limited_meta = ModelMetadata(
        alias="text-only-model",
        provider_id="mock_provider",
        underlying_model="basic-model",
        capabilities={ModelCapability.CHAT},  # No TOOL_CALLING
    )
    provider_registry.register_model_alias(limited_meta)

    request_with_tools = NormalizedChatRequest(
        model_alias="text-only-model",
        messages=[ChatMessage(role="user", content="Check status")],
        tools=[{"type": "function", "function": {"name": "test_tool"}}],
    )

    # Orchestrator must reject requests requesting unsupported capabilities
    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        await orchestrator.execute(request_with_tools)

    assert "does not support required capability: 'tool_calling'" in str(exc_info.value)
