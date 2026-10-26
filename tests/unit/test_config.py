from platform_core.config.loader import load_providers_config, build_model_metadata_map
from platform_core.contracts.capabilities import ModelCapability


def test_load_providers_config():
    config = load_providers_config("config/providers.yaml")
    assert "general-chat" in config.model_aliases
    assert "mock-chat" in config.model_aliases
    assert "hosted_chat" in config.providers
    assert "mock_provider" in config.providers


def test_model_metadata_capabilities():
    config = load_providers_config("config/providers.yaml")
    metadata_map = build_model_metadata_map(config)

    mock_chat = metadata_map["mock-chat"]
    assert mock_chat.alias == "mock-chat"
    assert mock_chat.supports(ModelCapability.CHAT) is True
    assert mock_chat.supports(ModelCapability.STREAMING) is True
    assert mock_chat.supports(ModelCapability.TOOL_CALLING) is True
