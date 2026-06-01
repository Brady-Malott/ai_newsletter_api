from __future__ import annotations

import re

from pydantic import Field, PostgresDsn, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingRequiredSettingsError(Exception):
    """Raised when required settings are missing."""

    def __init__(self, missing_fields: list[str]) -> None:
        """Initialize with list of missing field names."""
        self.missing_fields = missing_fields
        super().__init__(f"Missing required environment variables: {', '.join(missing_fields)}")


class InvalidSettingsError(Exception):
    """Raised when settings are invalid."""

    def __init__(self, invalid_fields: list[tuple[str, str]]) -> None:
        """Initialize with list of invalid field names and messages."""
        self.invalid_fields = invalid_fields
        summary = ", ".join(f"{field}: {message}" for field, message in invalid_fields)
        super().__init__(f"Invalid environment variables: {summary}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required environment variables
    postgres_user: str = Field(..., description="Postgres user (required)")
    postgres_password: str = Field(..., description="Postgres password (required)")
    postgres_host: str = Field(..., description="Postgres host (required)")
    postgres_port: int = Field(..., description="Postgres port (required)")
    postgres_db: str = Field(..., description="Postgres database name (required)")
    redis_host: str = Field(..., description="Redis host (required)")
    redis_port: int = Field(..., description="Redis port (required)")
    redis_db: int = Field(..., description="Redis database number (required)")
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    jwt_secret_key: str = Field(..., description="JWT secret key for token signing (required)")
    jwt_access_token_expire_minutes: int = Field(
        ..., description="JWT token expiration in minutes (required)"
    )

    # Database/Redis urls built from components
    database_url: PostgresDsn | None = Field(
        default=None,
        description="Database connection URL",
    )
    rate_limit_storage_url: str | None = Field(
        default=None,
        description="Rate limit storage URL",
    )

    @model_validator(mode="after")
    def build_derived_urls(self) -> Settings:
        """Build URLs from components when missing."""
        if self.database_url is None:
            self.database_url = PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        if self.rate_limit_storage_url is None:
            self.rate_limit_storage_url = (
                f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        return self

    # Optional environment variables (defaults provided)
    app_name: str = "ai-newsletter-api"
    environment: str = "local"
    log_level: str = "INFO"
    jwt_algorithm: str = "HS256"
    locale: str = Field(
        default="en",
        description="Default locale for user-facing strings (e.g. en, es)",
    )

    @staticmethod
    def _is_strong_jwt_secret(secret: str) -> bool:
        if len(secret) < 32:
            return False
        has_lower = re.search(r"[a-z]", secret) is not None
        has_upper = re.search(r"[A-Z]", secret) is not None
        has_digit = re.search(r"\d", secret) is not None
        has_symbol = re.search(r"[^a-zA-Z0-9\s]", secret) is not None
        return has_lower and has_upper and has_digit and has_symbol

    @model_validator(mode="after")
    def validate_jwt_secret_strength(self) -> Settings:
        if not self._is_strong_jwt_secret(self.jwt_secret_key):
            raise ValueError(
                "JWT secret key must be at least 32 characters and include upper, lower, "
                "number, and symbol characters."
            )
        return self


def validate_settings() -> Settings:
    """Validate settings and raise exception for missing required fields.

    Raises:
        MissingRequiredSettingsError: If required environment variables are missing
        ValidationError: For other validation errors
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as e:
        missing_fields: list[str] = []
        for error in e.errors():
            if error["type"] == "missing":
                field_name = error["loc"][0] if error["loc"] else "unknown"
                if isinstance(field_name, str):
                    missing_fields.append(field_name.upper())
                else:
                    missing_fields.append(str(field_name).upper())

        if missing_fields:
            raise MissingRequiredSettingsError(missing_fields) from e

        invalid_fields: list[tuple[str, str]] = []
        for error in e.errors():
            if error["type"] == "missing":
                continue
            field_path = ".".join(str(part) for part in error.get("loc", []))
            message = error.get("msg", "Invalid value")
            invalid_fields.append((field_path or "unknown", message))

        if invalid_fields:
            raise InvalidSettingsError(invalid_fields) from e

        # Re-raise if it's a different validation error
        raise


# Validate settings at import time.
# Exceptions will propagate to the importing module (e.g., app/main.py)
settings = validate_settings()
