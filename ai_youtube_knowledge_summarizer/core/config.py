from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger("ai_youtube_knowledge_summarizer.config")


EnvironmentName = Literal["local", "dev", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VectorStoreProvider = Literal["chroma"]
WhisperModelName = Literal["tiny", "base", "small", "medium", "large"]

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="AI YouTube Knowledge Summarizer")
    environment: EnvironmentName = Field(default="local")
    debug: bool = Field(default=False)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    openai_api_key: SecretStr | None = Field(default=None)
    openai_chat_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    whisper_model: WhisperModelName = Field(default="base")
    whisper_device: Literal["auto", "cpu", "cuda"] = Field(default="auto")

    vector_store_provider: VectorStoreProvider = Field(default="chroma")
    vector_store_dir: Path = Field(default=Path("ai_youtube_knowledge_summarizer/storage/vector_store"))

    videos_dir: Path = Field(default=Path("ai_youtube_knowledge_summarizer/storage/videos"))
    transcripts_dir: Path = Field(default=Path("ai_youtube_knowledge_summarizer/storage/transcripts"))

    log_dir: Path = Field(default=Path("var/log"))
    log_level: LogLevel = Field(default="INFO")
    log_to_stdout: bool = Field(default=True)
    log_to_file: bool = Field(default=True)

    youtube_download_timeout_seconds: int = Field(default=300, ge=10)
    transcription_timeout_seconds: int = Field(default=1800, ge=30)
    max_video_duration_seconds: int = Field(default=7200, ge=60)

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    retriever_top_k: int = Field(default=4, ge=1, le=20)

    request_timeout_seconds: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_base_delay_seconds: float = Field(default=0.5, ge=0.0)
    retry_max_delay_seconds: float = Field(default=8.0, ge=0.0)

    @field_validator(
        "videos_dir",
        "transcripts_dir",
        "vector_store_dir",
        "log_dir",
        mode="before",
    )
    @classmethod
    def normalize_path(cls, value: str | Path) -> Path:
        """Convert configured paths to Path objects."""
        logger.debug("Normalizing configured path: value=%s", value)
        return Path(value)

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> "Settings":
        """Validate cross-field runtime settings."""
        logger.info("Validating application configuration")

        if self.environment in {"staging", "production"} and self.debug:
            logger.error("Invalid configuration: debug enabled in %s", self.environment)
            raise ValueError("debug must be disabled in staging and production")

        if self.chunk_overlap >= self.chunk_size:
            logger.error(
                "Invalid chunk settings: chunk_overlap=%s chunk_size=%s",
                self.chunk_overlap,
                self.chunk_size,
            )
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            logger.error(
                "Invalid retry settings: retry_max_delay_seconds=%s retry_base_delay_seconds=%s",
                self.retry_max_delay_seconds,
                self.retry_base_delay_seconds,
            )
            raise ValueError("retry_max_delay_seconds must be >= retry_base_delay_seconds")

        if self.environment in {"staging", "production"} and self.openai_api_key is None:
            logger.error("Missing OPENAI_API_KEY for %s environment", self.environment)
            raise ValueError("OPENAI_API_KEY is required in staging and production")

        logger.info("Application configuration validation completed successfully")
        return self

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        logger.info("Ensuring runtime directories exist")

        for directory in [
            self.videos_dir,
            self.transcripts_dir,
            self.vector_store_dir,
            self.log_dir,
        ]:
            logger.debug("Creating directory if missing: path=%s", directory)
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Runtime directories are ready")

    def validate_startup(self) -> None:
        """Run startup validation before serving API requests."""
        logger.info("Starting application startup validation")

        self.ensure_directories()

        if not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not configured; OpenAI-dependent features will fail until configured"
            )

        if self.whisper_device == "cuda":
            logger.info("Whisper device explicitly configured as CUDA")
        elif self.whisper_device == "cpu":
            logger.info("Whisper device explicitly configured as CPU")
        else:
            logger.info("Whisper device configured as auto")

        logger.info(
            "Startup validation completed: app=%s environment=%s vector_store=%s",
            self.app_name,
            self.environment,
            self.vector_store_provider,
        )

    def safe_summary(self) -> dict[str, object]:
        """Return non-secret configuration values for logs and diagnostics."""
        logger.debug("Building safe configuration summary")

        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "debug": self.debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "openai_chat_model": self.openai_chat_model,
            "openai_embedding_model": self.openai_embedding_model,
            "openai_temperature": self.openai_temperature,
            "openai_api_key_configured": self.openai_api_key is not None,
            "whisper_model": self.whisper_model,
            "whisper_device": self.whisper_device,
            "vector_store_provider": self.vector_store_provider,
            "vector_store_dir": str(self.vector_store_dir),
            "videos_dir": str(self.videos_dir),
            "transcripts_dir": str(self.transcripts_dir),
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "log_to_stdout": self.log_to_stdout,
            "log_to_file": self.log_to_file,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retriever_top_k": self.retriever_top_k,
            "max_retries": self.max_retries,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings."""
    logger.info("Loading application settings")

    settings = Settings()

    logger.info(
        "Application settings loaded successfully: environment=%s app=%s",
        settings.environment,
        settings.app_name,
    )

    return settings


def validate_settings_on_startup() -> Settings:
    """Load settings and execute startup validation."""
    logger.info("Running configuration startup validation")

    settings = get_settings()
    settings.validate_startup()

    logger.info("Configuration startup validation finished successfully")

    return settings
