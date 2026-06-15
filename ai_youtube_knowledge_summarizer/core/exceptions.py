from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any


logger = logging.getLogger("ai_youtube_knowledge_summarizer.exceptions")


class AppError(Exception):
    """Base exception for all application-level errors."""

    default_status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code: str = "application_error"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize application error with structured metadata."""
        if not message or not message.strip():
            logger.error("AppError initialized with empty message")
            raise ValueError("Exception message must not be empty")

        self.message = message.strip()
        self.error_code = error_code or self.default_error_code
        self.status_code = status_code or self.default_status_code
        self.details = details or {}

        logger.debug(
            "Application exception initialized: error_code=%s status_code=%s message=%s",
            self.error_code,
            self.status_code,
            self.message,
        )

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception into a safe API response payload."""
        logger.debug("Converting exception to response payload: error_code=%s", self.error_code)

        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


class ConfigurationError(AppError):
    """Raised when application configuration is invalid or incomplete."""

    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code = "configuration_error"


class StartupValidationError(AppError):
    """Raised when startup validation fails before the app can serve traffic."""

    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code = "startup_validation_error"


class ValidationError(AppError):
    """Raised when user input or request data fails validation."""

    default_status_code = HTTPStatus.BAD_REQUEST
    default_error_code = "validation_error"


class YouTubeUrlValidationError(ValidationError):
    """Raised when a provided YouTube URL is empty or unsupported."""

    default_error_code = "youtube_url_validation_error"


class DownloadError(AppError):
    """Raised when video or audio download fails."""

    default_status_code = HTTPStatus.BAD_GATEWAY
    default_error_code = "download_error"


class VideoMetadataError(AppError):
    """Raised when video metadata cannot be extracted or validated."""

    default_status_code = HTTPStatus.BAD_GATEWAY
    default_error_code = "video_metadata_error"


class VideoTooLongError(ValidationError):
    """Raised when a video exceeds the configured maximum duration."""

    default_error_code = "video_too_long"


class TranscriptionError(AppError):
    """Raised when Whisper transcription fails."""

    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code = "transcription_error"


class TranscriptStorageError(AppError):
    """Raised when transcript persistence or loading fails."""

    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code = "transcript_storage_error"


class ChunkingError(AppError):
    """Raised when transcript chunking fails."""

    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code = "chunking_error"


class EmbeddingError(AppError):
    """Raised when embedding generation fails."""

    default_status_code = HTTPStatus.BAD_GATEWAY
    default_error_code = "embedding_error"


class VectorStoreError(AppError):
    """Raised when vector store operations fail."""

    default_status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    default_error_code = "vector_store_error"


class VectorStoreDocumentNotFoundError(AppError):
    """Raised when requested transcript/vector data is unavailable."""

    default_status_code = HTTPStatus.NOT_FOUND
    default_error_code = "vector_store_document_not_found"


class SummarizationError(AppError):
    """Raised when summary generation fails."""

    default_status_code = HTTPStatus.BAD_GATEWAY
    default_error_code = "summarization_error"


class RetrievalError(AppError):
    """Raised when retrieval or QA chain execution fails."""

    default_status_code = HTTPStatus.BAD_GATEWAY
    default_error_code = "retrieval_error"


class ExternalServiceError(AppError):
    """Raised when an external provider fails or returns invalid data."""

    default_status_code = HTTPStatus.BAD_GATEWAY
    default_error_code = "external_service_error"


class TimeoutError(AppError):
    """Raised when an operation exceeds its configured timeout."""

    default_status_code = HTTPStatus.GATEWAY_TIMEOUT
    default_error_code = "timeout_error"


class RateLimitError(AppError):
    """Raised when local or provider rate limits are exceeded."""

    default_status_code = HTTPStatus.TOO_MANY_REQUESTS
    default_error_code = "rate_limit_error"


class UnsafeOperationError(AppError):
    """Raised when a request attempts an unsupported or unsafe operation."""

    default_status_code = HTTPStatus.FORBIDDEN
    default_error_code = "unsafe_operation_error"


def log_exception(exc: Exception, *, context: dict[str, Any] | None = None) -> None:
    """Log an exception with optional structured context."""
    context = context or {}

    if isinstance(exc, AppError):
        logger.error(
            "Application error occurred: error_code=%s status_code=%s message=%s context=%s",
            exc.error_code,
            exc.status_code,
            exc.message,
            context,
            exc_info=True,
        )
        return

    logger.exception(
        "Unexpected exception occurred: exception_type=%s context=%s",
        type(exc).__name__,
        context,
    )


def error_response_from_exception(exc: Exception) -> tuple[dict[str, Any], int]:
    """Convert an exception into an API-safe response payload and status code."""
    logger.debug("Converting exception into API response: exception_type=%s", type(exc).__name__)

    if isinstance(exc, AppError):
        return exc.to_dict(), exc.status_code

    return (
        {
            "error": {
                "code": "internal_server_error",
                "message": "Unexpected internal error",
                "details": {},
            }
        },
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )
