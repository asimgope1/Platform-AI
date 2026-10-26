from platform_core.contracts.capabilities import ModelMetadata, ModelCapability
from platform_core.contracts.requests import NormalizedChatRequest
from platform_core.errors.exceptions import CapabilityNotSupportedError


class CapabilityChecker:
    """Verifies that the requested model adapter supports all requested features."""

    def check(self, request: NormalizedChatRequest, model_meta: ModelMetadata) -> None:
        # Base chat capability
        if not model_meta.supports(ModelCapability.CHAT):
            raise CapabilityNotSupportedError(model_meta.alias, "chat")

        # Streaming capability
        if request.stream and not model_meta.supports(ModelCapability.STREAMING):
            raise CapabilityNotSupportedError(model_meta.alias, "streaming")

        # Tool calling capability
        if request.tools and not model_meta.supports(ModelCapability.TOOL_CALLING):
            raise CapabilityNotSupportedError(model_meta.alias, "tool_calling")
