from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_youtube_knowledge_summarizer.core.config import validate_settings_on_startup
from ai_youtube_knowledge_summarizer.core.exceptions import AppError, log_exception
from ai_youtube_knowledge_summarizer.core.logging import setup_logging
from ai_youtube_knowledge_summarizer.services.chunker import TextChunkingService
from ai_youtube_knowledge_summarizer.services.downloader import YouTubeDownloaderService
from ai_youtube_knowledge_summarizer.services.qa_service import QAService
from ai_youtube_knowledge_summarizer.services.summarizer import SummarizationService
from ai_youtube_knowledge_summarizer.services.transcriber import WhisperTranscriptionService
from ai_youtube_knowledge_summarizer.services.vector_store import VectorStoreService


logger = logging.getLogger("ai_youtube_knowledge_summarizer.scripts.run_local_pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run local YouTube summarization and RAG pipeline."
    )

    parser.add_argument(
        "--url",
        required=True,
        help="YouTube URL to process.",
    )

    parser.add_argument(
        "--summary-mode",
        default="map_reduce",
        choices=["stuff", "map_reduce", "refine"],
        help="Summarization strategy.",
    )

    parser.add_argument(
        "--question",
        default=None,
        help="Optional question to ask after indexing.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of transcript chunks to retrieve for QA.",
    )

    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip summary generation.",
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the full local AI pipeline from YouTube URL to summary and QA."""
    started_at = time.perf_counter()

    app_logger = setup_logging()

    app_logger.info(
        "Local pipeline started",
        extra={
            "event": "local_pipeline_started",
            "operation": "run_local_pipeline",
            "status": "started",
            "youtube_url": args.url,
            "summary_mode": args.summary_mode,
        },
    )

    try:
        settings = validate_settings_on_startup()

        downloader = YouTubeDownloaderService(settings=settings, logger=app_logger)
        transcriber = WhisperTranscriptionService(settings=settings, logger=app_logger)
        chunker = TextChunkingService(settings=settings, logger=app_logger)
        vector_store = VectorStoreService(settings=settings, logger=app_logger)
        summarizer = SummarizationService(settings=settings, logger=app_logger)
        qa_service = QAService(settings=settings, logger=app_logger)

        video = downloader.download(args.url)

        transcript = transcriber.transcribe(
            video.file_path,
            video_id=video.video_id,
        )

        chunks = chunker.chunk_text(
            text=transcript.text,
            metadata={
                "video_id": video.video_id,
                "transcript_id": transcript.transcript_id,
                "title": video.title,
                "author": video.author,
                "source_url": video.source_url,
            },
        )

        vector_store.add_documents(
            transcript_id=transcript.transcript_id,
            documents=chunks,
        )

        summary = None
        if not args.skip_summary:
            summary = summarizer.summarize(
                transcript.text,
                mode=args.summary_mode,
            )

        qa_result = None
        if args.question:
            qa_result = qa_service.answer_question(
                transcript_id=transcript.transcript_id,
                question=args.question,
                top_k=args.top_k,
            )

        duration_ms = int((time.perf_counter() - started_at) * 1000)

        app_logger.info(
            "Local pipeline completed successfully",
            extra={
                "event": "local_pipeline_completed",
                "operation": "run_local_pipeline",
                "status": "success",
                "video_id": video.video_id,
                "transcript_id": transcript.transcript_id,
                "chunk_count": len(chunks),
                "duration_ms": duration_ms,
            },
        )

        print("\n=== Video ===")
        print(f"Title: {video.title}")
        print(f"Author: {video.author}")
        print(f"Duration seconds: {video.duration_seconds}")
        print(f"Video ID: {video.video_id}")
        print(f"Transcript ID: {transcript.transcript_id}")
        print(f"Chunks: {len(chunks)}")

        if summary:
            print("\n=== Summary ===")
            print(summary)

        if qa_result:
            print("\n=== Question ===")
            print(qa_result.question)

            print("\n=== Answer ===")
            print(qa_result.answer)

            print("\n=== Sources ===")
            for index, source in enumerate(qa_result.sources, start=1):
                print(f"\nSource {index}:")
                print(source.get("content", "")[:700])

        return 0

    except AppError as exc:
        log_exception(
            exc,
            context={
                "operation": "run_local_pipeline",
                "youtube_url": args.url,
            },
        )
        print(f"\nApplication error: {exc.message}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        app_logger.warning(
            "Local pipeline interrupted by user",
            extra={
                "event": "local_pipeline_interrupted",
                "operation": "run_local_pipeline",
                "status": "cancelled",
            },
        )
        print("\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as exc:
        log_exception(
            exc,
            context={
                "operation": "run_local_pipeline",
                "youtube_url": args.url,
            },
        )
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    """Run local pipeline command-line entry point."""
    args = parse_args()
    exit_code = run_pipeline(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
