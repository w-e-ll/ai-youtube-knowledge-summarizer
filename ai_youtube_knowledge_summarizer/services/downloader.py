from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import yt_dlp
from pydantic import HttpUrl

from ai_youtube_knowledge_summarizer.core.config import Settings
from ai_youtube_knowledge_summarizer.core.exceptions import (
    DownloadError,
    VideoMetadataError,
    VideoTooLongError,
    YouTubeUrlValidationError,
)
from ai_youtube_knowledge_summarizer.core.logging import log_failure, log_step
from ai_youtube_knowledge_summarizer.models.domain import VideoMetadata


logger = logging.getLogger("ai_youtube_knowledge_summarizer.services.downloader")


class YouTubeDownloaderService:
    """Download YouTube video/audio and extract safe metadata."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Initialize downloader service with application settings."""
        self.settings = settings
        self.logger = logger or logging.getLogger(
            "ai_youtube_knowledge_summarizer.services.downloader"
        )

        log_step(
            self.logger,
            event="downloader_initialized",
            operation="downloader_init",
            status="success",
            message="YouTube downloader service initialized",
            provider="yt-dlp",
        )

    def download(self, youtube_url: str | HttpUrl) -> VideoMetadata:
        """Download YouTube media and return normalized video metadata."""
        started_at = time.perf_counter()
        url = str(youtube_url)

        log_step(
            self.logger,
            event="download_requested",
            operation="download_youtube_video",
            status="started",
            message="YouTube download requested",
            youtube_url=url,
        )

        try:
            self._validate_youtube_url(url)

            self.settings.videos_dir.mkdir(parents=True, exist_ok=True)

            job_id = str(uuid4())
            output_template = str(self.settings.videos_dir / f"{job_id}.%(ext)s")

            metadata = self._extract_metadata(url=url)

            self._validate_metadata(metadata=metadata, url=url)

            duration = self._get_duration(metadata)
            if duration and duration > self.settings.max_video_duration_seconds:
                raise VideoTooLongError(
                    (
                        "Video duration exceeds configured maximum: "
                        f"duration_seconds={duration}, "
                        f"max_video_duration_seconds={self.settings.max_video_duration_seconds}"
                    ),
                    details={
                        "duration_seconds": duration,
                        "max_video_duration_seconds": self.settings.max_video_duration_seconds,
                    },
                )

            downloaded_file = self._download_media(
                url=url,
                output_template=output_template,
            )

            video = VideoMetadata(
                video_id=job_id,
                source_url=url,
                title=self._safe_string(metadata.get("title")),
                author=self._safe_string(metadata.get("uploader")),
                duration_seconds=duration,
                file_path=downloaded_file,
            )

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="download_completed",
                operation="download_youtube_video",
                status="success",
                message="YouTube download completed successfully",
                video_id=video.video_id,
                duration_ms=duration_ms,
                title=video.title,
                author=video.author,
                file_path=str(video.file_path),
            )

            return video

        except (
            YouTubeUrlValidationError,
            VideoMetadataError,
            VideoTooLongError,
            DownloadError,
        ):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="download_failed",
                operation="download_youtube_video",
                message="Unexpected YouTube download failure",
                exc=exc,
                youtube_url=url,
            )
            raise DownloadError(
                "Unexpected failure while downloading YouTube media",
                details={"youtube_url": url, "exception_type": type(exc).__name__},
            ) from exc

    def _validate_youtube_url(self, url: str) -> None:
        """Validate that URL belongs to a supported YouTube domain."""
        log_step(
            self.logger,
            event="youtube_url_validation_started",
            operation="validate_youtube_url",
            status="started",
            message="Validating YouTube URL",
            youtube_url=url,
        )

        normalized = url.strip()

        if not normalized:
            raise YouTubeUrlValidationError("YouTube URL must not be empty")

        allowed_patterns = (
            r"^https?://(www\.)?youtube\.com/watch\?",
            r"^https?://m\.youtube\.com/watch\?",
            r"^https?://youtu\.be/",
            r"^https?://(www\.)?youtube\.com/shorts/",
        )

        if not any(re.match(pattern, normalized) for pattern in allowed_patterns):
            raise YouTubeUrlValidationError(
                "Unsupported YouTube URL format",
                details={"youtube_url": url},
            )

        log_step(
            self.logger,
            event="youtube_url_validation_completed",
            operation="validate_youtube_url",
            status="success",
            message="YouTube URL validation completed",
            youtube_url=url,
        )

    def _extract_metadata(self, url: str) -> dict[str, Any]:
        """Extract YouTube metadata without downloading media."""
        log_step(
            self.logger,
            event="metadata_extraction_started",
            operation="extract_video_metadata",
            status="started",
            message="Extracting YouTube video metadata",
            youtube_url=url,
        )

        options = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": self.settings.youtube_download_timeout_seconds,
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                metadata = ydl.extract_info(url, download=False)

            if not isinstance(metadata, dict):
                raise VideoMetadataError(
                    "yt-dlp returned invalid metadata payload",
                    details={"payload_type": type(metadata).__name__},
                )

            log_step(
                self.logger,
                event="metadata_extraction_completed",
                operation="extract_video_metadata",
                status="success",
                message="YouTube video metadata extracted",
                title=self._safe_string(metadata.get("title")),
                duration_seconds=self._get_duration(metadata),
            )

            return metadata

        except VideoMetadataError:
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="metadata_extraction_failed",
                operation="extract_video_metadata",
                message="Failed to extract YouTube metadata",
                exc=exc,
                youtube_url=url,
            )
            raise VideoMetadataError(
                "Failed to extract YouTube metadata",
                details={"youtube_url": url, "exception_type": type(exc).__name__},
            ) from exc

    def _validate_metadata(self, metadata: dict[str, Any], url: str) -> None:
        """Validate required metadata fields before download."""
        log_step(
            self.logger,
            event="metadata_validation_started",
            operation="validate_video_metadata",
            status="started",
            message="Validating extracted video metadata",
            youtube_url=url,
        )

        title = self._safe_string(metadata.get("title"))

        if not title:
            raise VideoMetadataError(
                "Video title is missing in metadata",
                details={"youtube_url": url},
            )

        log_step(
            self.logger,
            event="metadata_validation_completed",
            operation="validate_video_metadata",
            status="success",
            message="Video metadata validation completed",
            title=title,
            duration_seconds=self._get_duration(metadata),
        )

    def _download_media(self, url: str, output_template: str) -> Path:
        """Download media file using yt-dlp and return local file path."""
        log_step(
            self.logger,
            event="media_download_started",
            operation="download_media_file",
            status="started",
            message="Downloading media file with yt-dlp",
            youtube_url=url,
            output_template=output_template,
        )

        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "noplaylist": True,
            "socket_timeout": self.settings.youtube_download_timeout_seconds,
            "retries": self.settings.max_retries,
            "continuedl": True,
            "ignoreerrors": False,
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                result = ydl.extract_info(url, download=True)

            downloaded_file = self._resolve_downloaded_file(
                ydl_result=result,
                output_template=output_template,
            )

            log_step(
                self.logger,
                event="media_download_completed",
                operation="download_media_file",
                status="success",
                message="Media file downloaded successfully",
                file_path=str(downloaded_file),
            )

            return downloaded_file

        except DownloadError:
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="media_download_failed",
                operation="download_media_file",
                message="Failed to download media file",
                exc=exc,
                youtube_url=url,
            )
            raise DownloadError(
                "Failed to download YouTube media file",
                details={"youtube_url": url, "exception_type": type(exc).__name__},
            ) from exc

    def _resolve_downloaded_file(
        self,
        ydl_result: dict[str, Any] | None,
        output_template: str,
    ) -> Path:
        """Resolve downloaded file path from yt-dlp result or output pattern."""
        log_step(
            self.logger,
            event="downloaded_file_resolution_started",
            operation="resolve_downloaded_file",
            status="started",
            message="Resolving downloaded media file path",
        )

        if not isinstance(ydl_result, dict):
            raise DownloadError("yt-dlp did not return a valid result payload")

        requested_downloads = ydl_result.get("requested_downloads") or []
        if requested_downloads:
            file_path = requested_downloads[0].get("filepath")
            if file_path:
                path = Path(file_path)
                if path.exists():
                    return path

        candidate_prefix = Path(output_template).with_suffix("")
        candidates = sorted(candidate_prefix.parent.glob(f"{candidate_prefix.name}.*"))

        if not candidates:
            raise DownloadError(
                "Downloaded media file could not be located",
                details={"output_template": output_template},
            )

        resolved = candidates[0]

        log_step(
            self.logger,
            event="downloaded_file_resolution_completed",
            operation="resolve_downloaded_file",
            status="success",
            message="Downloaded media file path resolved",
            file_path=str(resolved),
        )

        return resolved

    def _get_duration(self, metadata: dict[str, Any]) -> int | None:
        """Return video duration in seconds when available."""
        raw_duration = metadata.get("duration")

        if raw_duration is None:
            return None

        try:
            return int(raw_duration)
        except (TypeError, ValueError):
            logger.warning("Invalid duration metadata received: value=%s", raw_duration)
            return None

    def _safe_string(self, value: Any) -> str | None:
        """Normalize optional metadata strings."""
        if value is None:
            return None

        normalized = str(value).strip()

        return normalized or None
