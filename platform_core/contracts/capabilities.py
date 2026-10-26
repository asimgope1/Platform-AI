from enum import Enum
from typing import Set, List
from pydantic import BaseModel, Field


class ModelCapability(str, Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    EMBEDDINGS = "embeddings"
    JSON_MODE = "json_mode"


class ModelMetadata(BaseModel):
    alias: str
    provider_id: str
    underlying_model: str
    description: str = ""
    capabilities: Set[ModelCapability] = Field(default_factory=set)

    def supports(self, capability: ModelCapability) -> bool:
        return capability in self.capabilities

    def require(self, *required_capabilities: ModelCapability) -> None:
        missing = [cap for cap in required_capabilities if cap not in self.capabilities]
        if missing:
            raise ValueError(
                f"Model '{self.alias}' does not support required capabilities: {', '.join(missing)}"
            )
