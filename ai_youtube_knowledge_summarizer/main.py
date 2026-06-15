from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_youtube_knowledge_summarizer.api.routes import router
from ai_youtube_knowledge_summarizer.core.config import Settings, validate_settings_on_startup
from ai_youtube_knowledge_summarizer.core.exceptions import (
    AppError,
    error_response_from_exception,
    log_exception,
)
from ai_youtube_knowledge_summarizer.core.logging import setup_logging


logger = logging.getLogger("ai_youtube_knowledge_summarizer.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown lifecycle operations."""
    startup_started_at = time.perf_counter()

    app_logger = setup_logging()

    app_logger.info(
        "Application startup started",
        extra={
            "event": "application_startup_started",
            "operation": "startup",
            "status": "started",
        },
    )

    try:
        settings = validate_settings_on_startup()

        app.state.settings = settings

        app_logger.info(
            "Application startup validation completed",
            extra={
                "event": "application_startup_validation_completed",
                "operation": "startup",
                "status": "success",
                "settings": settings.safe_summary(),
            },
        )

        duration_ms = int((time.perf_counter() - startup_started_at) * 1000)

        app_logger.info(
            "Application startup completed successfully",
            extra={
                "event": "application_startup_completed",
                "operation": "startup",
                "status": "success",
                "duration_ms": duration_ms,
            },
        )

        yield

    except Exception as exc:
        log_exception(exc, context={"operation": "startup"})

        app_logger.error(
            "Application startup failed",
            extra={
                "event": "application_startup_failed",
                "operation": "startup",
                "status": "failed",
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )

        raise

    finally:
        app_logger.info(
            "Application shutdown started",
            extra={
                "event": "application_shutdown_started",
                "operation": "shutdown",
                "status": "started",
            },
        )

        app_logger.info(
            "Application shutdown completed",
            extra={
                "event": "application_shutdown_completed",
                "operation": "shutdown",
                "status": "success",
            },
        )


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="AI YouTube Knowledge Summarizer",
        description=(
            "Production-style AI SaaS backend for downloading YouTube audio, "
            "transcribing with Whisper, summarizing with LLMs, and answering "
            "questions using RAG over transcript embeddings."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/api/v1", tags=["AI YouTube Knowledge Summarizer"])

    register_exception_handlers(app)
    register_request_logging_middleware(app)

    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register API exception handlers."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Handle known application errors."""
        request_id = getattr(request.state, "request_id", None)

        log_exception(
            exc,
            context={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        payload, status_code = error_response_from_exception(exc)

        return JSONResponse(
            status_code=status_code,
            content=payload,
            headers={"X-Request-ID": request_id or ""},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected application errors safely."""
        request_id = getattr(request.state, "request_id", None)

        log_exception(
            exc,
            context={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        payload, status_code = error_response_from_exception(exc)

        return JSONResponse(
            status_code=status_code,
            content=payload,
            headers={"X-Request-ID": request_id or ""},
        )


def register_request_logging_middleware(app: FastAPI) -> None:
    """Register middleware for request correlation and access logging."""

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log request lifecycle with correlation ID and duration."""
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        started_at = time.perf_counter()

        logger.info(
            "HTTP request started",
            extra={
                "event": "http_request_started",
                "operation": "http_request",
                "status": "started",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        try:
            response = await call_next(request)

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            logger.info(
                "HTTP request completed",
                extra={
                    "event": "http_request_completed",
                    "operation": "http_request",
                    "status": "success",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)

            logger.error(
                "HTTP request failed",
                extra={
                    "event": "http_request_failed",
                    "operation": "http_request",
                    "status": "failed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "exception_type": type(exc).__name__,
                },
                exc_info=True,
            )

            raise


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings: Settings = validate_settings_on_startup()

    uvicorn.run(
        "ai_youtube_knowledge_summarizer.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
