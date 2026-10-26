from platform_core.contracts.capabilities import ModelMetadata
from providers.registry import provider_registry


class ModelResolver:
    """Resolves requested model aliases to platform metadata and provider adapters."""

    def resolve(self, model_alias: str) -> ModelMetadata:
        return provider_registry.get_model_metadata(model_alias)
