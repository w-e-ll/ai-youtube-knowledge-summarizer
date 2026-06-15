from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ai_youtube_knowledge_summarizer.main import create_app
from ai_youtube_knowledge_summarizer.models.domain import QuestionAnswerDocument, TranscriptDocument, VideoMetadata


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create FastAPI test client with startup validation mocked."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.main.validate_settings_on_startup",
        lambda: Mock(
            app_name="AI YouTube Knowledge Summarizer",
            environment="test",
            openai_api_key=None,
            vector_store_provider="chroma",
            safe_summary=lambda: {"environment": "test"},
        ),
    )

    return TestClient(create_app())


def test_health_endpoint_returns_status(client: TestClient) -> None:
    """Verify health endpoint returns application status."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["app_name"] == "AI YouTube Knowledge Summarizer"
    assert payload["environment"] == "test"
    assert payload["vector_store_provider"] == "chroma"


def test_process_video_rejects_non_youtube_url(client: TestClient) -> None:
    """Verify video processing rejects unsupported URL domains."""
    response = client.post(
        "/api/v1/videos/process",
        json={
            "youtube_url": "https://example.com/video",
            "generate_summary": True,
            "summary_mode": "map_reduce",
        },
    )

    assert response.status_code == 422


def test_process_video_success(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Verify video processing endpoint returns expected pipeline result."""
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake-video", encoding="utf-8")

    mock_downloader = Mock()
    mock_downloader.download.return_value = VideoMetadata(
        video_id="video-1",
        source_url="https://www.youtube.com/watch?v=abc123",
        title="Test Video",
        author="Test Author",
        duration_seconds=120,
        file_path=video_path,
    )

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = TranscriptDocument(
        transcript_id="transcript-1",
        video_id="video-1",
        text="This is a test transcript.",
    )

    mock_chunker = Mock()
    mock_chunker.chunk_text.return_value = [
        Mock(page_content="This is a test transcript.", metadata={"chunk_index": 0})
    ]

    mock_vector_store = Mock()
    mock_vector_store.add_documents.return_value = ["transcript-1:0"]

    mock_summarizer = Mock()
    mock_summarizer.summarize.return_value = "Test summary."

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.YouTubeDownloaderService",
        lambda settings, logger: mock_downloader,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.WhisperTranscriptionService",
        lambda settings, logger: mock_transcriber,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.TextChunkingService",
        lambda settings, logger: mock_chunker,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.VectorStoreService",
        lambda settings, logger: mock_vector_store,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.SummarizationService",
        lambda settings, logger: mock_summarizer,
    )

    response = client.post(
        "/api/v1/videos/process",
        json={
            "youtube_url": "https://www.youtube.com/watch?v=abc123",
            "generate_summary": True,
            "summary_mode": "map_reduce",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["video_id"] == "video-1"
    assert payload["transcript_id"] == "transcript-1"
    assert payload["title"] == "Test Video"
    assert payload["author"] == "Test Author"
    assert payload["duration_seconds"] == 120
    assert payload["chunk_count"] == 1
    assert payload["summary"] == "Test summary."

    mock_downloader.download.assert_called_once()
    mock_transcriber.transcribe.assert_called_once()
    mock_chunker.chunk_text.assert_called_once()
    mock_vector_store.add_documents.assert_called_once()
    mock_summarizer.summarize.assert_called_once()


def test_process_video_can_skip_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Verify video processing can skip summary generation."""
    video_path = tmp_path / "video.mp4"
    video_path.write_text("fake-video", encoding="utf-8")

    mock_downloader = Mock()
    mock_downloader.download.return_value = VideoMetadata(
        video_id="video-1",
        source_url="https://www.youtube.com/watch?v=abc123",
        title="Test Video",
        author="Test Author",
        duration_seconds=120,
        file_path=video_path,
    )

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = TranscriptDocument(
        transcript_id="transcript-1",
        video_id="video-1",
        text="This is a test transcript.",
    )

    mock_chunker = Mock()
    mock_chunker.chunk_text.return_value = [
        Mock(page_content="This is a test transcript.", metadata={"chunk_index": 0})
    ]

    mock_vector_store = Mock()
    mock_vector_store.add_documents.return_value = ["transcript-1:0"]

    mock_summarizer = Mock()

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.YouTubeDownloaderService",
        lambda settings, logger: mock_downloader,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.WhisperTranscriptionService",
        lambda settings, logger: mock_transcriber,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.TextChunkingService",
        lambda settings, logger: mock_chunker,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.VectorStoreService",
        lambda settings, logger: mock_vector_store,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.SummarizationService",
        lambda settings, logger: mock_summarizer,
    )

    response = client.post(
        "/api/v1/videos/process",
        json={
            "youtube_url": "https://www.youtube.com/watch?v=abc123",
            "generate_summary": False,
            "summary_mode": "map_reduce",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["summary"] is None
    mock_summarizer.summarize.assert_not_called()


def test_summarize_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify summary endpoint loads transcript text and returns summary."""
    mock_vector_store = Mock()
    mock_vector_store.get_transcript_text.return_value = "Transcript text"

    mock_summarizer = Mock()
    mock_summarizer.summarize.return_value = "Generated summary"

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.VectorStoreService",
        lambda settings, logger: mock_vector_store,
    )
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.SummarizationService",
        lambda settings, logger: mock_summarizer,
    )

    response = client.post(
        "/api/v1/summaries",
        json={
            "transcript_id": "transcript-1",
            "mode": "map_reduce",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["transcript_id"] == "transcript-1"
    assert payload["mode"] == "map_reduce"
    assert payload["summary"] == "Generated summary"

    mock_vector_store.get_transcript_text.assert_called_once_with("transcript-1")
    mock_summarizer.summarize.assert_called_once_with("Transcript text", mode="map_reduce")


def test_summarize_endpoint_rejects_empty_transcript_id(client: TestClient) -> None:
    """Verify summary endpoint rejects empty transcript ID."""
    response = client.post(
        "/api/v1/summaries",
        json={
            "transcript_id": "",
            "mode": "map_reduce",
        },
    )

    assert response.status_code == 422


def test_qa_endpoint_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify QA endpoint returns answer and sources."""
    qa_result = QuestionAnswerDocument(
        transcript_id="transcript-1",
        question="What is this about?",
        answer="It is about AI systems.",
        sources=[
            {
                "content": "AI systems transcript chunk",
                "score": None,
                "metadata": {"chunk_index": 0},
            }
        ],
    )

    mock_qa_service = Mock()
    mock_qa_service.answer_question.return_value = qa_result

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.api.routes.QAService",
        lambda settings, logger: mock_qa_service,
    )

    response = client.post(
        "/api/v1/qa",
        json={
            "transcript_id": "transcript-1",
            "question": "What is this about?",
            "top_k": 4,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["transcript_id"] == "transcript-1"
    assert payload["question"] == "What is this about?"
    assert payload["answer"] == "It is about AI systems."
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["content"] == "AI systems transcript chunk"

    mock_qa_service.answer_question.assert_called_once_with(
        transcript_id="transcript-1",
        question="What is this about?",
        top_k=4,
    )


def test_qa_endpoint_rejects_short_question(client: TestClient) -> None:
    """Verify QA endpoint rejects too-short questions."""
    response = client.post(
        "/api/v1/qa",
        json={
            "transcript_id": "transcript-1",
            "question": "a",
            "top_k": 4,
        },
    )

    assert response.status_code == 422


def test_qa_endpoint_rejects_invalid_top_k(client: TestClient) -> None:
    """Verify QA endpoint rejects invalid top_k values."""
    response = client.post(
        "/api/v1/qa",
        json={
            "transcript_id": "transcript-1",
            "question": "What is this about?",
            "top_k": 100,
        },
    )

    assert response.status_code == 422
