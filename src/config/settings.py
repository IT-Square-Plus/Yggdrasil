"""
Yggdrasil MCP Memory Settings
Context7 Best Practice: @lru_cache for singleton pattern
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Context7 Best Practices:
    - model_config instead of Config class
    - @lru_cache for singleton pattern
    - Field validators for custom validation
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown environment variables
        validate_default=True,
    )

    # ============================================
    # CHROMA CLOUD CONFIGURATION (REQUIRED)
    # ============================================

    chroma_api_key: str = Field(
        ...,
        description="Chroma Cloud API Key",
    )

    chroma_tenant: str = Field(
        ...,
        description="Chroma Cloud Tenant name",
    )

    chroma_database: str = Field(
        ...,
        description="Chroma Cloud Database name",
    )

    chroma_collection: str = Field(
        default="default_memories",
        description="Collection name - dynamic per project",
    )

    # ============================================
    # MCP SERVER CONFIGURATION
    # ============================================

    mcp_server_host: str = Field(
        default="0.0.0.0",
        description="Server host",
    )

    mcp_server_port: int = Field(
        default=8080,
        description="Server port",
        ge=1,
        le=65535,
    )

    transport: Literal["http", "stdio"] = Field(
        default="http",
        description="Transport protocol",
    )

    # ============================================
    # LOGGING
    # ============================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # ============================================
    # VALIDATORS
    # ============================================

    @field_validator("chroma_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate Chroma API key format."""
        if not v or len(v) < 10:
            raise ValueError("Invalid Chroma API key: must be at least 10 characters")
        return v

    @field_validator("chroma_tenant", "chroma_database", "chroma_collection")
    @classmethod
    def validate_name_fields(cls, v: str) -> str:
        """Validate name fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get application settings (singleton).

    Context7 Best Practice: Use @lru_cache to ensure only one Settings
    instance is created throughout the application lifecycle.

    Returns:
        Settings: Application settings instance
    """
    return Settings()
