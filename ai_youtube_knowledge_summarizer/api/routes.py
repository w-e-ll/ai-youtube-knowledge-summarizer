from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from ai_youtube_knowledge_summarizer.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    HealthResponse,
    ProcessVideoRequest,
    ProcessVideoResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from ai_youtube_knowledge_summarizer.core.config import Settings, get_settings
from ai_youtube_knowledge_summarizer.core.exceptions import (
    AppError,
    error_response_from_exception,
    log_exception,
)
from ai_youtube_knowledge_summarizer.core.logging import bind_logger
from ai_youtube_knowledge_summarizer.services.chunker import TextChunkingService
from ai_youtube_knowledge_summarizer.services.downloader import YouTubeDownloaderService
from ai_youtube_knowledge_summarizer.services.qa_service import QAService
from ai_youtube_knowledge_summarizer.services.summarizer import SummarizationService
from ai_youtube_knowledge_summarizer.services.transcriber import WhisperTranscriptionService
from ai_youtube_knowledge_summarizer.services.vector_store import VectorStoreService


router = APIRouter()
logger = logging.getLogger("ai_youtube_knowledge_summarizer.api.routes")


def get_request_id(request: Request) -> str:
    """Return request ID from headers or generate a new one."""
    return request.headers.get("X-Request-ID", str(uuid4()))


def raise_api_error(exc: Exception) -> None:
    """Convert internal exceptions into HTTP exceptions."""
    payload, status_code = error_response_from_exception(exc)
    raise HTTPException(status_code=status_code, detail=payload["error"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return application health and safe runtime status."""
    logger.info(
        "Health check requested",
        extra={
            "event": "health_check_requested",
            "operation": "health_check",
            "status": "started",
        },
    )

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        openai_configured=settings.openai_api_key is not None,
        vector_store_provider=settings.vector_store_provider,
    )


@router.post("/videos/process", response_model=ProcessVideoResponse)
def process_video(
    payload: ProcessVideoRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ProcessVideoResponse:
    """Download, transcribe, chunk, embed, and index a YouTube video."""
    request_id = get_request_id(request)
    request_logger = bind_logger(logger, request_id=request_id)

    request_logger.info(
        "Video processing request received",
        extra={
            "event": "video_processing_requested",
            "operation": "process_video",
            "status": "started",
            "youtube_url": str(payload.youtube_url),
        },
    )

    try:
        downloader = YouTubeDownloaderService(settings=settings, logger=request_logger)
        transcriber = WhisperTranscriptionService(settings=settings, logger=request_logger)
        chunker = TextChunkingService(settings=settings, logger=request_logger)
        vector_store = VectorStoreService(settings=settings, logger=request_logger)
        summarizer = SummarizationService(settings=settings, logger=request_logger)

        video = downloader.download(payload.youtube_url)
        transcript = transcriber.transcribe(video.file_path, video_id=video.video_id)
        chunks = chunker.chunk_text(
            text=transcript.text,
            metadata={
                "video_id": video.video_id,
                "title": video.title,
                "source_url": str(payload.youtube_url),
            },
        )

        vector_store.add_documents(
            transcript_id=transcript.transcript_id,
            documents=chunks,
        )

        summary = None
        if payload.generate_summary:
            summary = summarizer.summarize(transcript.text, mode=payload.summary_mode)

        request_logger.info(
            "Video processing completed",
            extra={
                "event": "video_processing_completed",
                "operation": "process_video",
                "status": "success",
                "video_id": video.video_id,
                "transcript_id": transcript.transcript_id,
                "chunk_count": len(chunks),
            },
        )

        return ProcessVideoResponse(
            request_id=request_id,
            video_id=video.video_id,
            transcript_id=transcript.transcript_id,
            title=video.title,
            author=video.author,
            duration_seconds=video.duration_seconds,
            chunk_count=len(chunks),
            summary=summary,
        )

    except AppError as exc:
        log_exception(exc, context={"request_id": request_id, "operation": "process_video"})
        raise_api_error(exc)

    except Exception as exc:
        log_exception(exc, context={"request_id": request_id, "operation": "process_video"})
        raise_api_error(exc)


@router.post("/summaries", response_model=SummarizeResponse)
def summarize_transcript(
    payload: SummarizeRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> SummarizeResponse:
    """Generate a summary for a stored transcript."""
    request_id = get_request_id(request)
    request_logger = bind_logger(
        logger,
        request_id=request_id,
        transcript_id=payload.transcript_id,
    )

    request_logger.info(
        "Summary request received",
        extra={
            "event": "summary_requested",
            "operation": "summarize_transcript",
            "status": "started",
            "summary_mode": payload.mode,
        },
    )

    try:
        vector_store = VectorStoreService(settings=settings, logger=request_logger)
        summarizer = SummarizationService(settings=settings, logger=request_logger)

        transcript_text = vector_store.get_transcript_text(payload.transcript_id)
        summary = summarizer.summarize(transcript_text, mode=payload.mode)

        request_logger.info(
            "Summary generated successfully",
            extra={
                "event": "summary_generated",
                "operation": "summarize_transcript",
                "status": "success",
            },
        )

        return SummarizeResponse(
            request_id=request_id,
            transcript_id=payload.transcript_id,
            mode=payload.mode,
            summary=summary,
        )

    except AppError as exc:
        log_exception(exc, context={"request_id": request_id, "operation": "summarize_transcript"})
        raise_api_error(exc)

    except Exception as exc:
        log_exception(exc, context={"request_id": request_id, "operation": "summarize_transcript"})
        raise_api_error(exc)


@router.post("/qa", response_model=AskQuestionResponse)
def ask_question(
    payload: AskQuestionRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> AskQuestionResponse:
    """Answer a question using retrieval over indexed transcript chunks."""
    request_id = get_request_id(request)
    request_logger = bind_logger(
        logger,
        request_id=request_id,
        transcript_id=payload.transcript_id,
    )

    request_logger.info(
        "Question answering request received",
        extra={
            "event": "qa_requested",
            "operation": "ask_question",
            "status": "started",
        },
    )

    try:
        qa_service = QAService(settings=settings, logger=request_logger)

        answer = qa_service.answer_question(
            transcript_id=payload.transcript_id,
            question=payload.question,
            top_k=payload.top_k,
        )

        request_logger.info(
            "Question answering completed",
            extra={
                "event": "qa_completed",
                "operation": "ask_question",
                "status": "success",
                "source_count": len(answer.sources),
            },
        )

        return AskQuestionResponse(
            request_id=request_id,
            transcript_id=payload.transcript_id,
            question=payload.question,
            answer=answer.answer,
            sources=answer.sources,
        )

    except AppError as exc:
        log_exception(exc, context={"request_id": request_id, "operation": "ask_question"})
        raise_api_error(exc)

    except Exception as exc:
        log_exception(exc, context={"request_id": request_id, "operation": "ask_question"})
        raise_api_error(exc)
