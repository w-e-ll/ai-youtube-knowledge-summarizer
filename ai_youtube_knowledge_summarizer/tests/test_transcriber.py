from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_youtube_knowledge_summarizer.core.exceptions import (
    TranscriptStorageError,
    TranscriptionError,
    ValidationError,
)
from ai_youtube_knowledge_summarizer.models.domain import TranscriptDocument
from ai_youtube_knowledge_summarizer.services.transcriber import WhisperTranscriptionService


@pytest.fixture
def settings(tmp_path: Path) -> Mock:
    """Create minimal settings object for transcription tests."""
    return Mock(
        whisper_model="base",
        whisper_device="cpu",
        transcripts_dir=tmp_path / "transcripts",
    )


@pytest.fixture
def transcriber(settings: Mock) -> WhisperTranscriptionService:
    """Create transcription service instance."""
    WhisperTranscriptionService._model_cache.clear()
    return WhisperTranscriptionService(settings=settings)


def test_validate_input_accepts_existing_media_file(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify transcription input validation accepts a valid media file."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    transcriber._validate_input(media_path=media_path, video_id="video-1")


def test_validate_input_rejects_empty_video_id(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify transcription input validation rejects empty video ID."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    with pytest.raises(ValidationError, match="video_id must not be empty"):
        transcriber._validate_input(media_path=media_path, video_id="")


def test_validate_input_rejects_missing_file(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify transcription input validation rejects missing media file."""
    with pytest.raises(ValidationError, match="Media file does not exist"):
        transcriber._validate_input(
            media_path=tmp_path / "missing.webm",
            video_id="video-1",
        )


def test_validate_input_rejects_directory(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify transcription input validation rejects directory paths."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()

    with pytest.raises(ValidationError, match="Media path is not a file"):
        transcriber._validate_input(media_path=media_dir, video_id="video-1")


def test_validate_input_rejects_empty_file(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify transcription input validation rejects empty media files."""
    media_path = tmp_path / "empty.webm"
    media_path.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError, match="Media file is empty"):
        transcriber._validate_input(media_path=media_path, video_id="video-1")


def test_load_model_uses_whisper_and_caches_model(
    transcriber: WhisperTranscriptionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Whisper model is loaded once and cached."""
    mock_model = Mock(name="whisper_model")
    load_model_mock = Mock(return_value=mock_model)

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.transcriber.whisper.load_model",
        load_model_mock,
    )

    first = transcriber._load_model()
    second = transcriber._load_model()

    assert first is mock_model
    assert second is mock_model
    load_model_mock.assert_called_once_with("base")


def test_load_model_wraps_failure(
    transcriber: WhisperTranscriptionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Whisper model load failures are wrapped."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.transcriber.whisper.load_model",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(TranscriptionError, match="Failed to load Whisper model"):
        transcriber._load_model()


def test_run_transcription_returns_payload(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify Whisper transcription payload is returned."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    model = Mock()
    model.transcribe.return_value = {
        "text": "Transcript text",
        "language": "en",
    }

    result = transcriber._run_transcription(
        model=model,
        media_path=media_path,
        video_id="video-1",
    )

    assert result["text"] == "Transcript text"
    assert result["language"] == "en"
    model.transcribe.assert_called_once()


def test_run_transcription_rejects_invalid_payload(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify invalid Whisper payload is rejected."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    model = Mock()
    model.transcribe.return_value = "bad-payload"

    with pytest.raises(TranscriptionError, match="invalid transcription payload"):
        transcriber._run_transcription(
            model=model,
            media_path=media_path,
            video_id="video-1",
        )


def test_run_transcription_wraps_unexpected_failure(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify Whisper runtime failures are wrapped."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    model = Mock()
    model.transcribe.side_effect = RuntimeError("boom")

    with pytest.raises(TranscriptionError, match="Whisper transcription failed"):
        transcriber._run_transcription(
            model=model,
            media_path=media_path,
            video_id="video-1",
        )


def test_extract_transcript_text_returns_text(
    transcriber: WhisperTranscriptionService,
) -> None:
    """Verify transcript text extraction returns normalized text."""
    text = transcriber._extract_transcript_text(
        result={"text": "  Transcript text  "},
        video_id="video-1",
    )

    assert text == "Transcript text"


def test_extract_transcript_text_rejects_empty_text(
    transcriber: WhisperTranscriptionService,
) -> None:
    """Verify empty Whisper transcript text is rejected."""
    with pytest.raises(TranscriptionError, match="empty transcript text"):
        transcriber._extract_transcript_text(
            result={"text": "   "},
            video_id="video-1",
        )


def test_extract_language_returns_normalized_language(
    transcriber: WhisperTranscriptionService,
) -> None:
    """Verify detected language is normalized."""
    assert transcriber._extract_language(result={"language": " en "}) == "en"


def test_extract_language_returns_none_for_missing_or_empty_language(
    transcriber: WhisperTranscriptionService,
) -> None:
    """Verify missing or empty language returns None."""
    assert transcriber._extract_language(result={}) is None
    assert transcriber._extract_language(result={"language": "   "}) is None


def test_save_transcript_writes_file(
    transcriber: WhisperTranscriptionService,
) -> None:
    """Verify transcript text is persisted to disk."""
    transcript_path = transcriber._save_transcript(
        transcript_id="transcript-1",
        video_id="video-1",
        transcript_text="Transcript text",
    )

    assert transcript_path.exists()
    assert transcript_path.read_text(encoding="utf-8") == "Transcript text"


def test_save_transcript_wraps_file_write_failure(
    transcriber: WhisperTranscriptionService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify transcript storage failures are wrapped."""
    monkeypatch.setattr(
        Path,
        "write_text",
        Mock(side_effect=OSError("disk error")),
    )

    with pytest.raises(TranscriptStorageError, match="Failed to save transcript"):
        transcriber._save_transcript(
            transcript_id="transcript-1",
            video_id="video-1",
            transcript_text="Transcript text",
        )


def test_transcribe_success(
    transcriber: WhisperTranscriptionService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify full transcribe flow returns TranscriptDocument."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "text": "Transcript text",
        "language": "en",
    }

    monkeypatch.setattr(transcriber, "_load_model", Mock(return_value=mock_model))

    result = transcriber.transcribe(media_path, video_id="video-1")

    assert isinstance(result, TranscriptDocument)
    assert result.video_id == "video-1"
    assert result.text == "Transcript text"
    assert result.language == "en"
    assert result.transcript_path is not None
    assert result.transcript_path.exists()


def test_transcribe_propagates_validation_error(
    transcriber: WhisperTranscriptionService,
    tmp_path: Path,
) -> None:
    """Verify validation errors are propagated as typed errors."""
    with pytest.raises(ValidationError):
        transcriber.transcribe(tmp_path / "missing.webm", video_id="video-1")


def test_transcribe_wraps_unexpected_failure(
    transcriber: WhisperTranscriptionService,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify unexpected transcription failures are wrapped."""
    media_path = tmp_path / "video.webm"
    media_path.write_text("fake media", encoding="utf-8")

    monkeypatch.setattr(
        transcriber,
        "_load_model",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(TranscriptionError, match="Unexpected failure"):
        transcriber.transcribe(media_path, video_id="video-1")
