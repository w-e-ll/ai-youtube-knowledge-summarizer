from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOGGER_NAME = "ai_youtube_knowledge_summarizer"


class MaxLevelFilter(logging.Filter):
    """Allow log records up to and including a maximum level."""

    def __init__(self, max_level: int) -> None:
        """Store the maximum accepted log level."""
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True when record level is lower than or equal to max_level."""
        return record.levelno <= self.max_level


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for production-friendly parsing."""

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record into a JSON string."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        for field in (
            "event",
            "request_id",
            "job_id",
            "video_id",
            "transcript_id",
            "operation",
            "duration_ms",
            "status",
            "provider",
            "model",
            "attempt",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class RequestContextFilter(logging.Filter):
    """Ensure common structured fields exist on every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate missing structured context fields with safe defaults."""
        for field in ("request_id", "job_id", "video_id", "transcript_id"):
            if not hasattr(record, field):
                setattr(record, field, None)

        return True


def parse_log_level(level: str | int) -> int:
    """Convert log level name or number into a logging level."""
    if isinstance(level, int):
        return level

    normalized = level.strip().upper()

    if normalized not in logging._nameToLevel:
        raise ValueError(f"Unsupported log level: {level}")

    return logging._nameToLevel[normalized]


def setup_logging(
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    log_dir: str | Path | None = "var/log",
    level: str | int = logging.INFO,
    stdout: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Configure application logging for stdout and rotating log files."""
    log_level = parse_log_level(level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonFormatter()
    context_filter = RequestContextFilter()

    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(context_filter)
        logger.addHandler(stdout_handler)

    if file_logging:
        if log_dir is None:
            raise ValueError("log_dir must be provided when file_logging=True")

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        date_suffix = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        info_log_path = log_path / f"ai-youtube-knowledge-info-{date_suffix}.log"
        error_log_path = log_path / f"ai-youtube-knowledge-error-{date_suffix}.log"

        info_handler = logging.handlers.RotatingFileHandler(
            info_log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        info_handler.setLevel(logging.DEBUG)
        info_handler.addFilter(MaxLevelFilter(logging.WARNING))
        info_handler.addFilter(context_filter)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(context_filter)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    logger.info(
        "Logging initialized",
        extra={
            "event": "logging_initialized",
            "operation": "setup_logging",
            "status": "success",
        },
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific application logger."""
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")


def bind_logger(
    logger: logging.Logger,
    *,
    request_id: str | None = None,
    job_id: str | None = None,
    video_id: str | None = None,
    transcript_id: str | None = None,
) -> logging.LoggerAdapter:
    """Attach request/job/video context to a logger."""
    return logging.LoggerAdapter(
        logger,
        {
            "request_id": request_id,
            "job_id": job_id,
            "video_id": video_id,
            "transcript_id": transcript_id,
        },
    )


def log_step(
    logger: logging.Logger | logging.LoggerAdapter,
    *,
    event: str,
    operation: str,
    status: str,
    message: str,
    **extra: Any,
) -> None:
    """Log a production-visible pipeline step."""
    logger.info(
        message,
        extra={
            "event": event,
            "operation": operation,
            "status": status,
            **extra,
        },
    )


def log_failure(
    logger: logging.Logger | logging.LoggerAdapter,
    *,
    event: str,
    operation: str,
    message: str,
    exc: Exception,
    **extra: Any,
) -> None:
    """Log a production-visible failure with exception details."""
    logger.error(
        message,
        extra={
            "event": event,
            "operation": operation,
            "status": "failed",
            "exception_type": type(exc).__name__,
            **extra,
        },
        exc_info=True,
    )
