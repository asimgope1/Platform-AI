from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    API_KEY_HEADER: str = "Authorization"
    # Disabled unless explicitly configured via environment
    MASTER_API_KEY: Optional[str] = None

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./ai_platform.db"

    # Config files
    PROVIDERS_CONFIG_PATH: str = "config/providers.yaml"
    ROUTING_CONFIG_PATH: str = "config/routing.yaml"

    # Upstream API credentials
    HOSTED_AI_BASE_URL: Optional[str] = "https://api.openai.com/v1"
    HOSTED_AI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = "https://api.openai.com/v1"


def validate_master_key(cfg: Settings) -> None:
    """Validates that production environments enforce a secure, non-placeholder MASTER_API_KEY."""
    if cfg.ENVIRONMENT.lower() == "production":
        key = cfg.MASTER_API_KEY
        if not key or len(key) < 32:
            raise ValueError(
                "Production requires an explicitly configured "
                "MASTER_API_KEY of at least 32 characters."
            )
        if key == "sk-platform-master-key-change-in-production":
            raise ValueError(
                "The default placeholder cannot be used in production."
            )


settings = Settings()
