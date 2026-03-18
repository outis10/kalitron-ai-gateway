from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import RuntimeConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    PROJECT_NAME: str = "AI Gateway"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ALLOWED_ORIGINS: str = "http://localhost,http://127.0.0.1"
    AI_PROVIDER: str = "anthropic"
    AI_PROVIDER_IDENTITY: str = ""
    AI_PROVIDER_RECEIPT: str = ""

    # Claude / Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    ANTHROPIC_TIMEOUT_SECONDS: float = 20.0
    ANTHROPIC_MAX_RETRIES: int = 1

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_TIMEOUT_SECONDS: float = 20.0
    OPENAI_MAX_RETRIES: int = 1

    # API Key auth (comma-separated list)
    API_KEYS: str = "dev-key-1"

    # Confidence score thresholds (0-100)
    SCORE_AUTO_APPROVE: float = 95.0
    SCORE_HUMAN_REVIEW: float = 70.0

    # Max file size in MB
    MAX_FILE_SIZE_MB: int = 10

    @property
    def api_keys_list(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def cors_allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def ai_provider(self) -> str:
        return self.AI_PROVIDER.strip().lower()

    @staticmethod
    def _normalize_provider(value: str) -> str:
        return value.strip().lower()

    def provider_for_pipeline(self, pipeline_name: str) -> str:
        provider_by_pipeline = {
            "identity": self.AI_PROVIDER_IDENTITY,
            "receipt": self.AI_PROVIDER_RECEIPT,
        }
        configured_provider = provider_by_pipeline.get(pipeline_name, "")
        provider = self._normalize_provider(configured_provider or self.AI_PROVIDER)
        if provider not in {"anthropic", "openai"}:
            raise RuntimeConfigurationError(
                f"Provider for pipeline '{pipeline_name}' must be 'anthropic' or 'openai', got "
                f"'{configured_provider or self.AI_PROVIDER}'."
            )
        return provider

    def validate_runtime(self) -> None:
        """Fail fast for unsafe production-like configurations."""
        for provider_name, raw_value in (
            ("AI_PROVIDER", self.AI_PROVIDER),
            ("AI_PROVIDER_IDENTITY", self.AI_PROVIDER_IDENTITY),
            ("AI_PROVIDER_RECEIPT", self.AI_PROVIDER_RECEIPT),
        ):
            if raw_value.strip() and self._normalize_provider(raw_value) not in {
                "anthropic",
                "openai",
            }:
                raise RuntimeConfigurationError(
                    f"{provider_name} must be 'anthropic' or 'openai', got '{raw_value}'."
                )

        if self.ENVIRONMENT.lower() == "development":
            return

        if not self.api_keys_list or self.API_KEYS.strip() == "dev-key-1":
            raise RuntimeConfigurationError("API_KEYS must be configured outside development.")

        if not self.cors_allowed_origins:
            raise RuntimeConfigurationError("CORS_ALLOWED_ORIGINS must define at least one origin.")

        configured_providers = {
            self.provider_for_pipeline("identity"),
            self.provider_for_pipeline("receipt"),
        }

        if "anthropic" in configured_providers and not self.ANTHROPIC_API_KEY.strip():
            raise RuntimeConfigurationError(
                "ANTHROPIC_API_KEY must be configured when an Anthropic pipeline is enabled."
            )

        if "openai" in configured_providers and not self.OPENAI_API_KEY.strip():
            raise RuntimeConfigurationError(
                "OPENAI_API_KEY must be configured when an OpenAI pipeline is enabled."
            )


settings = Settings()
