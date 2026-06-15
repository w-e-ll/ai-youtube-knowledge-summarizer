from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import whisper

from ai_youtube_knowledge_summarizer.core.config import Settings
from ai_youtube_knowledge_summarizer.core.exceptions import (
    TranscriptStorageError,
    TranscriptionError,
    ValidationError,
)
from ai_youtube_knowledge_summarizer.core.logging import log_failure, log_step
from ai_youtube_knowledge_summarizer.models.domain import TranscriptDocument


logger = logging.getLogger("ai_youtube_knowledge_summarizer.services.transcriber")


class WhisperTranscriptionService:
    """Transcribe local media files using Whisper."""

    _model_cache: dict[str, Any] = {}

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Initialize transcription service with settings and logger."""
        self.settings = settings
        self.logger = logger or logging.getLogger(
            "ai_youtube_knowledge_summarizer.services.transcriber"
        )

        log_step(
            self.logger,
            event="transcriber_initialized",
            operation="transcriber_init",
            status="success",
            message="Whisper transcription service initialized",
            model=self.settings.whisper_model,
            provider="whisper",
        )

    def transcribe(self, file_path: str | Path, *, video_id: str) -> TranscriptDocument:
        """Transcribe a media file and persist transcript text."""
        started_at = time.perf_counter()
        media_path = Path(file_path)

        log_step(
            self.logger,
            event="transcription_requested",
            operation="transcribe_media",
            status="started",
            message="Transcription requested",
            video_id=video_id,
            file_path=str(media_path),
            model=self.settings.whisper_model,
        )

        try:
            self._validate_input(media_path=media_path, video_id=video_id)

            model = self._load_model()

            result = self._run_transcription(
                model=model,
                media_path=media_path,
                video_id=video_id,
            )

            transcript_text = self._extract_transcript_text(result=result, video_id=video_id)
            language = self._extract_language(result=result)

            transcript_id = str(uuid4())
            transcript_path = self._save_transcript(
                transcript_id=transcript_id,
                video_id=video_id,
                transcript_text=transcript_text,
            )

            transcript = TranscriptDocument(
                transcript_id=transcript_id,
                video_id=video_id,
                text=transcript_text,
                language=language,
                transcript_path=transcript_path,
            )

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="transcription_completed",
                operation="transcribe_media",
                status="success",
                message="Transcription completed successfully",
                video_id=video_id,
                transcript_id=transcript.transcript_id,
                transcript_path=str(transcript_path),
                language=language,
                duration_ms=duration_ms,
                text_length=len(transcript_text),
            )

            return transcript

        except (ValidationError, TranscriptionError, TranscriptStorageError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="transcription_failed",
                operation="transcribe_media",
                message="Unexpected transcription failure",
                exc=exc,
                video_id=video_id,
                file_path=str(media_path),
            )
            raise TranscriptionError(
                "Unexpected failure while transcribing media file",
                details={
                    "video_id": video_id,
                    "file_path": str(media_path),
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_input(self, *, media_path: Path, video_id: str) -> None:
        """Validate media file and video identifier before transcription."""
        log_step(
            self.logger,
            event="transcription_input_validation_started",
            operation="validate_transcription_input",
            status="started",
            message="Validating transcription input",
            video_id=video_id,
            file_path=str(media_path),
        )

        if not video_id or not video_id.strip():
            raise ValidationError("video_id must not be empty")

        if not media_path.exists():
            raise ValidationError(
                "Media file does not exist",
                details={"file_path": str(media_path)},
            )

        if not media_path.is_file():
            raise ValidationError(
                "Media path is not a file",
                details={"file_path": str(media_path)},
            )

        if media_path.stat().st_size <= 0:
            raise ValidationError(
                "Media file is empty",
                details={"file_path": str(media_path)},
            )

        log_step(
            self.logger,
            event="transcription_input_validation_completed",
            operation="validate_transcription_input",
            status="success",
            message="Transcription input validation completed",
            video_id=video_id,
            file_path=str(media_path),
            file_size_bytes=media_path.stat().st_size,
        )

    def _load_model(self) -> Any:
        """Load Whisper model with in-process caching."""
        model_name = self.settings.whisper_model

        log_step(
            self.logger,
            event="whisper_model_load_requested",
            operation="load_whisper_model",
            status="started",
            message="Whisper model load requested",
            model=model_name,
        )

        if model_name in self._model_cache:
            log_step(
                self.logger,
                event="whisper_model_cache_hit",
                operation="load_whisper_model",
                status="success",
                message="Using cached Whisper model",
                model=model_name,
            )
            return self._model_cache[model_name]

        try:
            device = (
                "cpu"
                if self.settings.whisper_device == "cpu"
                else self.settings.whisper_device
            )

            model = whisper.load_model(model_name, device=device)
            
            self._model_cache[model_name] = model

            log_step(
                self.logger,
                event="whisper_model_loaded",
                operation="load_whisper_model",
                status="success",
                message="Whisper model loaded successfully",
                model=model_name,
            )

            return model

        except Exception as exc:
            log_failure(
                self.logger,
                event="whisper_model_load_failed",
                operation="load_whisper_model",
                message="Failed to load Whisper model",
                exc=exc,
                model=model_name,
            )
            raise TranscriptionError(
                "Failed to load Whisper model",
                details={
                    "model": model_name,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _run_transcription(
        self,
        *,
        model: Any,
        media_path: Path,
        video_id: str,
    ) -> dict[str, Any]:
        """Run Whisper transcription for a media file."""
        log_step(
            self.logger,
            event="whisper_transcription_started",
            operation="run_whisper_transcription",
            status="started",
            message="Running Whisper transcription",
            video_id=video_id,
            file_path=str(media_path),
            model=self.settings.whisper_model,
        )

        try:
            kwargs: dict[str, Any] = {}

            if self.settings.whisper_device != "auto":
                kwargs["fp16"] = self.settings.whisper_device == "cuda"

            result = model.transcribe(str(media_path), **kwargs)

            if not isinstance(result, dict):
                raise TranscriptionError(
                    "Whisper returned invalid transcription payload",
                    details={"payload_type": type(result).__name__},
                )

            log_step(
                self.logger,
                event="whisper_transcription_completed",
                operation="run_whisper_transcription",
                status="success",
                message="Whisper transcription completed",
                video_id=video_id,
                language=result.get("language"),
            )

            return result

        except TranscriptionError:
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="whisper_transcription_failed",
                operation="run_whisper_transcription",
                message="Whisper transcription failed",
                exc=exc,
                video_id=video_id,
                file_path=str(media_path),
            )
            raise TranscriptionError(
                "Whisper transcription failed",
                details={
                    "video_id": video_id,
                    "file_path": str(media_path),
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _extract_transcript_text(self, *, result: dict[str, Any], video_id: str) -> str:
        """Extract and validate transcript text from Whisper result."""
        log_step(
            self.logger,
            event="transcript_text_extraction_started",
            operation="extract_transcript_text",
            status="started",
            message="Extracting transcript text",
            video_id=video_id,
        )

        text = str(result.get("text") or "").strip()

        if not text:
            raise TranscriptionError(
                "Whisper returned empty transcript text",
                details={"video_id": video_id},
            )

        log_step(
            self.logger,
            event="transcript_text_extraction_completed",
            operation="extract_transcript_text",
            status="success",
            message="Transcript text extracted",
            video_id=video_id,
            text_length=len(text),
        )

        return text

    def _extract_language(self, *, result: dict[str, Any]) -> str | None:
        """Extract detected language from Whisper result."""
        language = result.get("language")

        if language is None:
            return None

        normalized = str(language).strip()

        return normalized or None

    def _save_transcript(
        self,
        *,
        transcript_id: str,
        video_id: str,
        transcript_text: str,
    ) -> Path:
        """Persist transcript text to local storage."""
        log_step(
            self.logger,
            event="transcript_save_started",
            operation="save_transcript",
            status="started",
            message="Saving transcript to disk",
            video_id=video_id,
            transcript_id=transcript_id,
        )

        try:
            self.settings.transcripts_dir.mkdir(parents=True, exist_ok=True)

            transcript_path = self.settings.transcripts_dir / f"{transcript_id}.txt"

            transcript_path.write_text(transcript_text, encoding="utf-8")

            log_step(
                self.logger,
                event="transcript_save_completed",
                operation="save_transcript",
                status="success",
                message="Transcript saved successfully",
                video_id=video_id,
                transcript_id=transcript_id,
                transcript_path=str(transcript_path),
                text_length=len(transcript_text),
            )

            return transcript_path

        except Exception as exc:
            log_failure(
                self.logger,
                event="transcript_save_failed",
                operation="save_transcript",
                message="Failed to save transcript",
                exc=exc,
                video_id=video_id,
                transcript_id=transcript_id,
            )
            raise TranscriptStorageError(
                "Failed to save transcript to disk",
                details={
                    "video_id": video_id,
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc
