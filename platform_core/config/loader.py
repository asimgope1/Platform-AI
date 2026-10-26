import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from platform_core.config.validation import ProvidersConfigFile
from platform_core.contracts.capabilities import ModelMetadata, ModelCapability
from platform_core.errors.exceptions import PlatformError


def load_providers_config(config_path: str) -> ProvidersConfigFile:
    path = Path(config_path)
    if not path.is_file():
        raise PlatformError(f"Providers configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ProvidersConfigFile.model_validate(data)


def build_model_metadata_map(config: ProvidersConfigFile) -> Dict[str, ModelMetadata]:
    model_map: Dict[str, ModelMetadata] = {}

    for alias, alias_cfg in config.model_aliases.items():
        capabilities = set()
        for cap_str in alias_cfg.capabilities:
            try:
                capabilities.add(ModelCapability(cap_str.lower()))
            except ValueError:
                pass

        model_map[alias] = ModelMetadata(
            alias=alias,
            provider_id=alias_cfg.provider,
            underlying_model=alias_cfg.model,
            description=alias_cfg.description or "",
            capabilities=capabilities,
        )

    return model_map
