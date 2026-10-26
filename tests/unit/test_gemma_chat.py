import pytest
from unittest.mock import MagicMock
from platform_core.contracts.capabilities import ModelMetadata, ModelCapability
from platform_core.contracts.requests import NormalizedChatRequest, ChatMessage
from providers.chat.gemma_transformers import GemmaTransformersChatAdapter


@pytest.mark.asyncio
async def test_gemma_adapter_prompt_slicing_and_usage():
    """Verify that Gemma adapter correctly slices out the prompt tokens and builds normalized response."""
    adapter = GemmaTransformersChatAdapter("gemma_transformers", {"model": "google/gemma-4-E2B-it"})

    # Mock tokenizer and model to test contract adherence without downloading weights
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Simulate apply_chat_template returning 5 tokens for the prompt
    import torch
    dummy_input_ids = torch.tensor([[101, 102, 103, 104, 105]])
    mock_tokenizer.apply_chat_template.return_value = {"input_ids": dummy_input_ids}

    # Simulate model.generate returning 5 prompt tokens + 4 newly generated tokens = 9 total tokens
    mock_model.generate.return_value = torch.tensor([[101, 102, 103, 104, 105, 201, 202, 203, 204]])

    # Tokenizer decode should be called ONLY on the 4 newly generated tokens [201, 202, 203, 204]
    mock_tokenizer.decode.return_value = "This is a Gemma answer."

    adapter._tokenizer = mock_tokenizer
    adapter._model = mock_model

    req = NormalizedChatRequest(
        model_alias="gemma-local",
        messages=[ChatMessage(role="user", content="Explain gravity in simple terms")],
    )
    meta = ModelMetadata(
        alias="gemma-local",
        provider_id="gemma_transformers",
        underlying_model="google/gemma-4-E2B-it",
        capabilities={ModelCapability.CHAT},
    )

    resp = await adapter.chat_completion(req, meta)

    # 1. Output content matches newly generated tokens only
    assert resp.choices[0].message.content == "This is a Gemma answer."
    assert resp.choices[0].finish_reason == "stop"

    # 2. Token counts accurately reflect prompt vs generated
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 4
    assert resp.usage.total_tokens == 9
    assert resp.model == "gemma-local"
