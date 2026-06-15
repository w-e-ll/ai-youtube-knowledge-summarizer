from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from ai_youtube_knowledge_summarizer.core.config import (
    Settings,
    get_settings,
    validate_settings_on_startup,
)


def test_settings_load_default_values() -> None:
    """Verify settings can be created with default local configuration."""
    settings = Settings(_env_file=None)

    assert settings.app_name == "AI YouTube Knowledge Summarizer"
    assert settings.environment == "local"
    assert settings.debug is False
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.openai_chat_model == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.whisper_model == "base"
    assert settings.vector_store_provider == "chroma"


def test_settings_rejects_invalid_api_port() -> None:
    """Verify API port validation rejects invalid values."""
    with pytest.raises(PydanticValidationError):
        Settings(api_port=70000, _env_file=None)


def test_settings_rejects_invalid_environment() -> None:
    """Verify environment validation rejects unsupported values."""
    with pytest.raises(PydanticValidationError):
        Settings(environment="bad-env", _env_file=None)


def test_settings_rejects_debug_in_production() -> None:
    """Verify production environment cannot run with debug enabled."""
    with pytest.raises(PydanticValidationError, match="debug must be disabled"):
        Settings(environment="production", debug=True, _env_file=None)


def test_settings_requires_openai_key_in_production() -> None:
    """Verify production environment requires OpenAI API key."""
    with pytest.raises(PydanticValidationError, match="OPENAI_API_KEY is required"):
        Settings(environment="production", openai_api_key=None, _env_file=None)


def test_settings_allows_openai_key_in_production() -> None:
    """Verify production configuration accepts configured OpenAI API key."""
    settings = Settings(
        environment="production",
        debug=False,
        openai_api_key="sk-test",
        _env_file=None,
    )

    assert settings.environment == "production"
    assert settings.openai_api_key is not None


def test_settings_rejects_invalid_chunk_overlap() -> None:
    """Verify chunk overlap must be smaller than chunk size."""
    with pytest.raises(PydanticValidationError, match="chunk_overlap"):
        Settings(chunk_size=100, chunk_overlap=100, _env_file=None)


def test_settings_rejects_invalid_retry_delay_order() -> None:
    """Verify retry max delay must be greater than or equal to base delay."""
    with pytest.raises(PydanticValidationError, match="retry_max_delay_seconds"):
        Settings(
            retry_base_delay_seconds=5,
            retry_max_delay_seconds=1,
            _env_file=None,
        )


def test_settings_normalizes_path_fields(tmp_path: Path) -> None:
    """Verify string path settings are normalized to Path objects."""
    settings = Settings(
        videos_dir=str(tmp_path / "videos"),
        transcripts_dir=str(tmp_path / "transcripts"),
        vector_store_dir=str(tmp_path / "vector_store"),
        log_dir=str(tmp_path / "log"),
        _env_file=None,
    )

    assert isinstance(settings.videos_dir, Path)
    assert isinstance(settings.transcripts_dir, Path)
    assert isinstance(settings.vector_store_dir, Path)
    assert isinstance(settings.log_dir, Path)


def test_ensure_directories_creates_runtime_paths(tmp_path: Path) -> None:
    """Verify ensure_directories creates required runtime directories."""
    settings = Settings(
        videos_dir=tmp_path / "videos",
        transcripts_dir=tmp_path / "transcripts",
        vector_store_dir=tmp_path / "vector_store",
        log_dir=tmp_path / "log",
        _env_file=None,
    )

    settings.ensure_directories()

    assert settings.videos_dir.exists()
    assert settings.transcripts_dir.exists()
    assert settings.vector_store_dir.exists()
    assert settings.log_dir.exists()


def test_validate_startup_creates_directories(tmp_path: Path) -> None:
    """Verify startup validation prepares runtime directories."""
    settings = Settings(
        videos_dir=tmp_path / "videos",
        transcripts_dir=tmp_path / "transcripts",
        vector_store_dir=tmp_path / "vector_store",
        log_dir=tmp_path / "log",
        _env_file=None,
    )

    settings.validate_startup()

    assert settings.videos_dir.exists()
    assert settings.transcripts_dir.exists()
    assert settings.vector_store_dir.exists()
    assert settings.log_dir.exists()


def test_safe_summary_does_not_expose_secret() -> None:
    """Verify safe_summary never exposes OpenAI secret value."""
    settings = Settings(openai_api_key="sk-secret-value", _env_file=None)

    summary = settings.safe_summary()

    assert "sk-secret-value" not in str(summary)
    assert summary["openai_api_key_configured"] is True


def test_safe_summary_includes_operational_values() -> None:
    """Verify safe_summary includes important non-secret runtime values."""
    settings = Settings(_env_file=None)

    summary = settings.safe_summary()

    assert summary["app_name"] == settings.app_name
    assert summary["environment"] == settings.environment
    assert summary["api_port"] == settings.api_port
    assert summary["vector_store_provider"] == settings.vector_store_provider
    assert summary["chunk_size"] == settings.chunk_size
    assert summary["retriever_top_k"] == settings.retriever_top_k


def test_get_settings_returns_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_settings returns a cached settings instance."""
    get_settings.cache_clear()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    first = get_settings()
    second = get_settings()

    assert first is second

    get_settings.cache_clear()


def test_validate_settings_on_startup_returns_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify startup helper loads settings and validates runtime paths."""
    get_settings.cache_clear()

    monkeypatch.setenv("VIDEOS_DIR", str(tmp_path / "videos"))
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "log"))

    settings = validate_settings_on_startup()

    assert settings.videos_dir.exists()
    assert settings.transcripts_dir.exists()
    assert settings.vector_store_dir.exists()
    assert settings.log_dir.exists()

    get_settings.cache_clear()
