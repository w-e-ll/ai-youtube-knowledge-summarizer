from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_youtube_knowledge_summarizer.core.exceptions import (
    DownloadError,
    VideoMetadataError,
    VideoTooLongError,
    YouTubeUrlValidationError,
)
from ai_youtube_knowledge_summarizer.models.domain import VideoMetadata
from ai_youtube_knowledge_summarizer.services.downloader import YouTubeDownloaderService


@pytest.fixture
def settings(tmp_path: Path) -> Mock:
    """Create minimal settings object for downloader tests."""
    return Mock(
        videos_dir=tmp_path / "videos",
        youtube_download_timeout_seconds=30,
        max_video_duration_seconds=3600,
        max_retries=2,
    )


@pytest.fixture
def downloader(settings: Mock) -> YouTubeDownloaderService:
    """Create downloader service instance."""
    return YouTubeDownloaderService(settings=settings)


def test_validate_youtube_url_accepts_watch_url(downloader: YouTubeDownloaderService) -> None:
    """Verify standard YouTube watch URLs are accepted."""
    downloader._validate_youtube_url("https://www.youtube.com/watch?v=abc123")


def test_validate_youtube_url_accepts_short_url(downloader: YouTubeDownloaderService) -> None:
    """Verify youtu.be short URLs are accepted."""
    downloader._validate_youtube_url("https://youtu.be/abc123")


def test_validate_youtube_url_accepts_shorts_url(downloader: YouTubeDownloaderService) -> None:
    """Verify YouTube Shorts URLs are accepted."""
    downloader._validate_youtube_url("https://www.youtube.com/shorts/abc123")


def test_validate_youtube_url_rejects_empty_url(downloader: YouTubeDownloaderService) -> None:
    """Verify empty YouTube URL is rejected."""
    with pytest.raises(YouTubeUrlValidationError, match="must not be empty"):
        downloader._validate_youtube_url("")


def test_validate_youtube_url_rejects_non_youtube_url(
    downloader: YouTubeDownloaderService,
) -> None:
    """Verify non-YouTube URL is rejected."""
    with pytest.raises(YouTubeUrlValidationError, match="Unsupported YouTube URL"):
        downloader._validate_youtube_url("https://example.com/watch?v=abc123")


def test_validate_youtube_url_rejects_unsupported_youtube_path(
    downloader: YouTubeDownloaderService,
) -> None:
    """Verify unsupported YouTube path is rejected."""
    with pytest.raises(YouTubeUrlValidationError):
        downloader._validate_youtube_url("https://www.youtube.com/channel/abc123")


def test_safe_string_normalizes_values(downloader: YouTubeDownloaderService) -> None:
    """Verify optional metadata strings are normalized."""
    assert downloader._safe_string("  hello  ") == "hello"
    assert downloader._safe_string("") is None
    assert downloader._safe_string("   ") is None
    assert downloader._safe_string(None) is None
    assert downloader._safe_string(123) == "123"


def test_get_duration_returns_integer(downloader: YouTubeDownloaderService) -> None:
    """Verify duration metadata is converted to integer."""
    assert downloader._get_duration({"duration": "120"}) == 120
    assert downloader._get_duration({"duration": 120}) == 120


def test_get_duration_returns_none_for_missing_or_invalid_value(
    downloader: YouTubeDownloaderService,
) -> None:
    """Verify missing or invalid duration returns None."""
    assert downloader._get_duration({}) is None
    assert downloader._get_duration({"duration": "bad"}) is None


def test_validate_metadata_accepts_title(downloader: YouTubeDownloaderService) -> None:
    """Verify metadata validation accepts payload with title."""
    downloader._validate_metadata(
        metadata={"title": "Test Video", "duration": 100},
        url="https://www.youtube.com/watch?v=abc123",
    )


def test_validate_metadata_rejects_missing_title(
    downloader: YouTubeDownloaderService,
) -> None:
    """Verify metadata validation rejects missing title."""
    with pytest.raises(VideoMetadataError, match="title is missing"):
        downloader._validate_metadata(
            metadata={"duration": 100},
            url="https://www.youtube.com/watch?v=abc123",
        )


def test_resolve_downloaded_file_uses_requested_downloads(
    downloader: YouTubeDownloaderService,
    tmp_path: Path,
) -> None:
    """Verify downloaded file path is resolved from yt-dlp requested_downloads."""
    file_path = tmp_path / "video.webm"
    file_path.write_text("fake media", encoding="utf-8")

    resolved = downloader._resolve_downloaded_file(
        ydl_result={
            "requested_downloads": [
                {
                    "filepath": str(file_path),
                }
            ]
        },
        output_template=str(tmp_path / "video.%(ext)s"),
    )

    assert resolved == file_path


def test_resolve_downloaded_file_falls_back_to_output_pattern(
    downloader: YouTubeDownloaderService,
    tmp_path: Path,
) -> None:
    """Verify downloaded file path can be resolved from output template pattern."""
    file_path = tmp_path / "job-1.webm"
    file_path.write_text("fake media", encoding="utf-8")

    resolved = downloader._resolve_downloaded_file(
        ydl_result={},
        output_template=str(tmp_path / "job-1.%(ext)s"),
    )

    assert resolved == file_path


def test_resolve_downloaded_file_rejects_missing_file(
    downloader: YouTubeDownloaderService,
    tmp_path: Path,
) -> None:
    """Verify unresolved downloaded file raises DownloadError."""
    with pytest.raises(DownloadError, match="could not be located"):
        downloader._resolve_downloaded_file(
            ydl_result={},
            output_template=str(tmp_path / "missing.%(ext)s"),
        )


def test_resolve_downloaded_file_rejects_invalid_payload(
    downloader: YouTubeDownloaderService,
    tmp_path: Path,
) -> None:
    """Verify invalid yt-dlp result payload raises DownloadError."""
    with pytest.raises(DownloadError, match="valid result payload"):
        downloader._resolve_downloaded_file(
            ydl_result=None,
            output_template=str(tmp_path / "missing.%(ext)s"),
        )


def test_download_success(monkeypatch: pytest.MonkeyPatch, downloader: YouTubeDownloaderService, tmp_path: Path) -> None:
    """Verify download returns VideoMetadata when all pipeline steps succeed."""
    media_path = tmp_path / "videos" / "video.webm"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_text("fake media", encoding="utf-8")

    monkeypatch.setattr(
        downloader,
        "_extract_metadata",
        lambda url: {
            "title": "Test Video",
            "uploader": "Test Author",
            "duration": 120,
        },
    )
    monkeypatch.setattr(
        downloader,
        "_download_media",
        lambda url, output_template: media_path,
    )

    result = downloader.download("https://www.youtube.com/watch?v=abc123")

    assert isinstance(result, VideoMetadata)
    assert result.title == "Test Video"
    assert result.author == "Test Author"
    assert result.duration_seconds == 120
    assert result.file_path == media_path


def test_download_rejects_too_long_video(
    monkeypatch: pytest.MonkeyPatch,
    downloader: YouTubeDownloaderService,
) -> None:
    """Verify videos longer than configured limit are rejected."""
    monkeypatch.setattr(
        downloader,
        "_extract_metadata",
        lambda url: {
            "title": "Long Video",
            "uploader": "Test Author",
            "duration": 999999,
        },
    )

    with pytest.raises(VideoTooLongError, match="duration exceeds"):
        downloader.download("https://www.youtube.com/watch?v=abc123")


def test_download_propagates_validation_error(downloader: YouTubeDownloaderService) -> None:
    """Verify URL validation errors are propagated as typed errors."""
    with pytest.raises(YouTubeUrlValidationError):
        downloader.download("https://example.com/video")


def test_download_wraps_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    downloader: YouTubeDownloaderService,
) -> None:
    """Verify unexpected errors are wrapped as DownloadError."""
    monkeypatch.setattr(
        downloader,
        "_extract_metadata",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(VideoMetadataError):
        downloader.download("https://www.youtube.com/watch?v=abc123")
