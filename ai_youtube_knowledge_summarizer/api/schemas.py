from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


logger = logging.getLogger("ai_youtube_knowledge_summarizer.api.schemas")


SummaryMode = Literal["stuff", "map_reduce", "refine"]


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str = Field(description="Application health status.")
    app_name: str = Field(description="Application name.")
    environment: str = Field(description="Runtime environment.")
    openai_configured: bool = Field(description="Whether OpenAI API key is configured.")
    vector_store_provider: str = Field(description="Configured vector store provider.")


class ProcessVideoRequest(BaseModel):
    """Request payload for processing a YouTube video."""

    youtube_url: HttpUrl = Field(description="YouTube video URL to process.")
    generate_summary: bool = Field(default=True, description="Generate summary after indexing.")
    summary_mode: SummaryMode = Field(default="map_reduce", description="Summary strategy.")

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, value: HttpUrl) -> HttpUrl:
        """Validate that the URL points to a supported YouTube domain."""
        logger.debug("Validating YouTube URL: url=%s", value)

        host = value.host or ""
        allowed_hosts = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
        }

        if host.lower() not in allowed_hosts:
            logger.error("Unsupported YouTube URL host: host=%s", host)
            raise ValueError("Only YouTube URLs are supported")

        logger.debug("YouTube URL validation completed: host=%s", host)
        return value


class ProcessVideoResponse(BaseModel):
    """Response returned after video processing completes."""

    request_id: str = Field(description="Request correlation ID.")
    video_id: str = Field(description="Internal video identifier.")
    transcript_id: str = Field(description="Internal transcript identifier.")
    title: str | None = Field(default=None, description="Video title.")
    author: str | None = Field(default=None, description="Video author/uploader.")
    duration_seconds: int | None = Field(default=None, description="Video duration in seconds.")
    chunk_count: int = Field(description="Number of indexed transcript chunks.")
    summary: str | None = Field(default=None, description="Generated summary, if requested.")


class SummarizeRequest(BaseModel):
    """Request payload for generating a transcript summary."""

    transcript_id: str = Field(min_length=1, description="Transcript identifier.")
    mode: SummaryMode = Field(default="map_reduce", description="Summary strategy.")

    @field_validator("transcript_id")
    @classmethod
    def validate_transcript_id(cls, value: str) -> str:
        """Validate transcript identifier."""
        logger.debug("Validating transcript_id: transcript_id=%s", value)

        normalized = value.strip()
        if not normalized:
            logger.error("Invalid transcript_id: empty value")
            raise ValueError("transcript_id must not be empty")

        logger.debug("Transcript_id validation completed: transcript_id=%s", normalized)
        return normalized


class SummarizeResponse(BaseModel):
    """Response returned after summary generation."""

    request_id: str = Field(description="Request correlation ID.")
    transcript_id: str = Field(description="Transcript identifier.")
    mode: SummaryMode = Field(description="Summary strategy used.")
    summary: str = Field(description="Generated summary.")


class AskQuestionRequest(BaseModel):
    """Request payload for retrieval-based question answering."""

    transcript_id: str = Field(min_length=1, description="Transcript identifier.")
    question: str = Field(min_length=3, max_length=2000, description="Question to answer.")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve.")

    @field_validator("transcript_id")
    @classmethod
    def validate_transcript_id(cls, value: str) -> str:
        """Validate transcript identifier."""
        logger.debug("Validating QA transcript_id: transcript_id=%s", value)

        normalized = value.strip()
        if not normalized:
            logger.error("Invalid QA transcript_id: empty value")
            raise ValueError("transcript_id must not be empty")

        logger.debug("QA transcript_id validation completed: transcript_id=%s", normalized)
        return normalized

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate user question."""
        logger.debug("Validating QA question")

        normalized = value.strip()
        if not normalized:
            logger.error("Invalid QA question: empty value")
            raise ValueError("question must not be empty")

        logger.debug("QA question validation completed")
        return normalized


class SourceChunk(BaseModel):
    """Source transcript chunk used to generate an answer."""

    content: str = Field(description="Retrieved chunk content.")
    score: float | None = Field(default=None, description="Similarity score if available.")
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description="Chunk metadata.",
    )


class AskQuestionResponse(BaseModel):
    """Response returned after retrieval-based question answering."""

    request_id: str = Field(description="Request correlation ID.")
    transcript_id: str = Field(description="Transcript identifier.")
    question: str = Field(description="Original question.")
    answer: str = Field(description="Generated answer.")
    sources: list[SourceChunk] = Field(default_factory=list, description="Retrieved source chunks.")


class ErrorPayload(BaseModel):
    """Structured API error payload."""

    code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error message.")
    details: dict = Field(default_factory=dict, description="Additional error context.")


class ErrorResponse(BaseModel):
    """Structured API error response."""

    error: ErrorPayload = Field(description="Error payload.")


class PipelineStatusResponse(BaseModel):
    """Generic response for future async pipeline status checks."""

    request_id: str = Field(description="Request correlation ID.")
    job_id: str = Field(description="Pipeline job identifier.")
    status: Literal["pending", "running", "success", "failed"] = Field(
        description="Current pipeline status."
    )
    message: str | None = Field(default=None, description="Optional status message.")


class VideoMetadata(BaseModel):
    """Public video metadata returned by API responses."""

    video_id: str = Field(description="Internal video identifier.")
    title: str | None = Field(default=None, description="Video title.")
    author: str | None = Field(default=None, description="Video uploader.")
    source_url: str = Field(description="Original video URL.")
    duration_seconds: int | None = Field(default=None, description="Video duration in seconds.")


class TranscriptMetadata(BaseModel):
    """Public transcript metadata returned by API responses."""

    transcript_id: str = Field(description="Internal transcript identifier.")
    video_id: str = Field(description="Internal video identifier.")
    chunk_count: int | None = Field(default=None, description="Number of transcript chunks.")
    language: str | None = Field(default=None, description="Detected transcript language.")
