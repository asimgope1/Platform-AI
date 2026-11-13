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

    # Telephony / PBX
    PBX_WS_URL: Optional[str] = None
    PBX_IGNORE_TRANSCRIPT: bool = True
    IVR_CHAT_MODEL: str = "mock-chat"
    GEMMA_MODEL_ALIAS: str = "gemma4"
    IVR_LANGUAGE: str = "en"

    # Audio Intelligence (Milestone 1)
    AUDIO_INTELLIGENCE_ENABLED: bool = True
    AUDIO_SAMPLE_RATE: int = 8000
    VAD_PROVIDER: str = "adaptive"
    VAD_THRESHOLD: float = 0.42
    NOISE_SUPPRESSION_ENABLED: bool = True
    NOISE_SUPPRESSION_PROVIDER: str = "spectral"
    ECHO_CANCELLATION_ENABLED: bool = False

    # Remote Business Tool Service (Windows Django Backend)
    REMOTE_TOOLS_ENABLED: bool = False
    REMOTE_TOOLS_BASE_URL: Optional[str] = None
    REMOTE_TOOLS_AUTH_TOKEN: Optional[str] = None
    REMOTE_TOOLS_API_KEY: Optional[str] = None
    REMOTE_TOOLS_HMAC_SECRET: Optional[str] = None
    REMOTE_TOOLS_REQUIRE_AUTH: bool = False
    REMOTE_TOOLS_REQUIRE_HTTPS_IN_PROD: bool = True
    REMOTE_TOOLS_TIMEOUT_SECONDS: float = 5.0
    REMOTE_TOOLS_MAX_RETRIES: int = 2
    REMOTE_TOOLS_PER_TENANT_CONCURRENCY: int = 8
    REMOTE_TOOLS_RATE_LIMIT_PER_MIN: int = 120
    REMOTE_TOOLS_DEFAULT_TENANT_SLUG: Optional[str] = None
    REMOTE_TOOLS_TENANT_MAP: str = "{}"
    REMOTE_TOOLS_PREFER_REMOTE: bool = False
    DATA_ACCESS_MASTER_KEY: Optional[str] = None

    # External Complaint / Business Action API Integration & Call Tooling
    COMPLAINT_API_URL: Optional[str] = None
    COMPLAINT_API_TOKEN: Optional[str] = None
    EXTERNAL_API_ALLOWED_BASES: str = ""
    IVR_ENABLE_TOOLS: bool = True
    IVR_TOOL_TURN_TIMEOUT_SECONDS: float = 95.0

    # Direct Tenant CRM Database (Universal Data Access)
    CRM_DB_ENABLED: bool = False
    CRM_DB_NAME: str = "crm_backend_db"
    CRM_DB_ENGINE: str = "postgresql"
    CRM_DB_HOST: str = "192.168.1.124"
    CRM_DB_PORT: int = 5435
    CRM_DB_SCHEMA: str = "public"
    CRM_DB_USERNAME: Optional[str] = None
    CRM_DB_PASSWORD: Optional[str] = None
    CRM_DB_TENANT_ID: Optional[str] = None
    CRM_DB_READ_ONLY: bool = True


def validate_master_key(cfg: Settings) -> None:
    """Validates that production environments enforce secure keys and fail-closed remote auth."""
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
        if cfg.REMOTE_TOOLS_ENABLED:
            if not cfg.REMOTE_TOOLS_REQUIRE_AUTH:
                raise ValueError(
                    "Production requires REMOTE_TOOLS_REQUIRE_AUTH=True when REMOTE_TOOLS_ENABLED=True."
                )
            has_cred = any(
                (v or "").strip()
                for v in (cfg.REMOTE_TOOLS_AUTH_TOKEN, cfg.REMOTE_TOOLS_API_KEY, cfg.REMOTE_TOOLS_HMAC_SECRET)
            )
            if not has_cred:
                raise ValueError(
                    "Production requires at least one remote tool credential "
                    "(REMOTE_TOOLS_API_KEY, REMOTE_TOOLS_AUTH_TOKEN, or REMOTE_TOOLS_HMAC_SECRET)."
                )
            if cfg.REMOTE_TOOLS_REQUIRE_HTTPS_IN_PROD and cfg.REMOTE_TOOLS_BASE_URL:
                if not cfg.REMOTE_TOOLS_BASE_URL.strip().lower().startswith("https://"):
                    raise ValueError(
                        "Production requires an https:// REMOTE_TOOLS_BASE_URL unless "
                        "REMOTE_TOOLS_REQUIRE_HTTPS_IN_PROD=False is explicitly set for a private mTLS tunnel."
                    )


settings = Settings()
