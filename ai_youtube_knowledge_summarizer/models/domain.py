from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


logger = logging.getLogger("ai_youtube_knowledge_summarizer.models.domain")


class VideoMetadata(BaseModel):
    """Domain model representing downloaded YouTube video metadata."""

    video_id: str = Field(default_factory=lambda: str(uuid4()))
    source_url: str
    title: str | None = None
    author: str | None = None
    duration_seconds: int | None = Field(default=None, ge=0)
    file_path: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        """Validate source URL value."""
        logger.debug("Validating source_url for VideoMetadata")

        normalized = value.strip()

        if not normalized:
            logger.error("VideoMetadata source_url validation failed: empty value")
            raise ValueError("source_url must not be empty")

        return normalized

    @field_validator("file_path", mode="before")
    @classmethod
    def normalize_file_path(cls, value: str | Path) -> Path:
        """Convert file path into Path object."""
        logger.debug("Normalizing video file path: value=%s", value)
        return Path(value)

    @model_validator(mode="after")
    def validate_video_metadata(self) -> "VideoMetadata":
        """Validate cross-field video metadata consistency."""
        logger.debug(
            "Running VideoMetadata model validation: video_id=%s",
            self.video_id,
        )

        if self.duration_seconds is not None and self.duration_seconds <= 0:
            logger.error(
                "Invalid video duration detected: video_id=%s duration_seconds=%s",
                self.video_id,
                self.duration_seconds,
            )
            raise ValueError("duration_seconds must be positive")

        logger.debug(
            "VideoMetadata validation completed successfully: video_id=%s",
            self.video_id,
        )

        return self


class TranscriptDocument(BaseModel):
    """Domain model representing generated transcript content."""

    transcript_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str
    text: str
    language: str | None = None
    transcript_path: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        """Validate associated video ID."""
        logger.debug("Validating transcript video_id")

        normalized = value.strip()

        if not normalized:
            logger.error("TranscriptDocument validation failed: empty video_id")
            raise ValueError("video_id must not be empty")

        return normalized

    @field_validator("text")
    @classmethod
    def validate_transcript_text(cls, value: str) -> str:
        """Validate transcript content."""
        logger.debug("Validating transcript text")

        normalized = value.strip()

        if not normalized:
            logger.error("TranscriptDocument validation failed: empty transcript")
            raise ValueError("Transcript text must not be empty")

        return normalized

    @field_validator("transcript_path", mode="before")
    @classmethod
    def normalize_transcript_path(cls, value: str | Path | None) -> Path | None:
        """Convert transcript path into Path object."""
        if value is None:
            return None

        logger.debug("Normalizing transcript path: value=%s", value)

        return Path(value)


class ChunkDocument(BaseModel):
    """Domain model representing a transcript chunk."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    transcript_id: str
    chunk_index: int = Field(ge=0)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("transcript_id")
    @classmethod
    def validate_transcript_id(cls, value: str) -> str:
        """Validate transcript identifier."""
        logger.debug("Validating chunk transcript_id")

        normalized = value.strip()

        if not normalized:
            logger.error("ChunkDocument validation failed: empty transcript_id")
            raise ValueError("transcript_id must not be empty")

        return normalized

    @field_validator("text")
    @classmethod
    def validate_chunk_text(cls, value: str) -> str:
        """Validate chunk text."""
        logger.debug("Validating chunk text")

        normalized = value.strip()

        if not normalized:
            logger.error("ChunkDocument validation failed: empty text")
            raise ValueError("Chunk text must not be empty")

        return normalized


class SummaryDocument(BaseModel):
    """Domain model representing generated summaries."""

    summary_id: str = Field(default_factory=lambda: str(uuid4()))
    transcript_id: str
    summary: str
    mode: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        """Validate generated summary."""
        logger.debug("Validating summary text")

        normalized = value.strip()

        if not normalized:
            logger.error("SummaryDocument validation failed: empty summary")
            raise ValueError("summary must not be empty")

        return normalized

    @field_validator("mode")
    @classmethod
    def validate_summary_mode(cls, value: str) -> str:
        """Validate summary generation mode."""
        logger.debug("Validating summary mode: mode=%s", value)

        normalized = value.strip()

        allowed_modes = {"stuff", "map_reduce", "refine"}

        if normalized not in allowed_modes:
            logger.error(
                "Unsupported summary mode detected: mode=%s",
                normalized,
            )
            raise ValueError(f"Unsupported summary mode: {normalized}")

        return normalized


class QuestionAnswerDocument(BaseModel):
    """Domain model representing retrieval QA results."""

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    transcript_id: str
    question: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate user question."""
        logger.debug("Validating QA question")

        normalized = value.strip()

        if not normalized:
            logger.error("QuestionAnswerDocument validation failed: empty question")
            raise ValueError("question must not be empty")

        return normalized

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """Validate generated answer."""
        logger.debug("Validating QA answer")

        normalized = value.strip()

        if not normalized:
            logger.error("QuestionAnswerDocument validation failed: empty answer")
            raise ValueError("answer must not be empty")

        return normalized


class ProcessingMetrics(BaseModel):
    """Domain model representing pipeline execution metrics."""

    operation: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    success: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        """Validate operation name."""
        logger.debug("Validating processing operation")

        normalized = value.strip()

        if not normalized:
            logger.error("ProcessingMetrics validation failed: empty operation")
            raise ValueError("operation must not be empty")

        return normalized

    @model_validator(mode="after")
    def validate_processing_metrics(self) -> "ProcessingMetrics":
        """Validate processing metrics consistency."""
        logger.debug("Validating ProcessingMetrics model")

        if self.completed_at and self.completed_at < self.started_at:
            logger.error(
                "Invalid processing timestamps detected: operation=%s",
                self.operation,
            )
            raise ValueError("completed_at must be greater than started_at")

        return self
