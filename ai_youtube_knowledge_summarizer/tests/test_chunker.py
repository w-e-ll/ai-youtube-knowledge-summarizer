from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ai_youtube_knowledge_summarizer.core.exceptions import ValidationError
from ai_youtube_knowledge_summarizer.services.chunker import TextChunkingService


@pytest.fixture
def settings() -> Mock:
    """Create minimal settings object for chunking tests."""
    return Mock(
        chunk_size=100,
        chunk_overlap=10,
    )


@pytest.fixture
def chunker(settings: Mock) -> TextChunkingService:
    """Create chunking service instance."""
    return TextChunkingService(settings=settings)


def test_chunk_text_returns_langchain_documents(chunker: TextChunkingService) -> None:
    """Verify chunk_text returns LangChain Document objects."""
    documents = chunker.chunk_text(
        text="This is a test transcript. " * 20,
        metadata={"video_id": "video-1"},
    )

    assert documents
    assert all(isinstance(document, Document) for document in documents)


def test_chunk_text_preserves_metadata(chunker: TextChunkingService) -> None:
    """Verify metadata is copied into every chunk."""
    documents = chunker.chunk_text(
        text="This is a test transcript. " * 20,
        metadata={
            "video_id": "video-1",
            "title": "Test Video",
            "source_url": "https://www.youtube.com/watch?v=abc123",
        },
    )

    assert documents

    for index, document in enumerate(documents):
        assert document.metadata["video_id"] == "video-1"
        assert document.metadata["title"] == "Test Video"
        assert document.metadata["source_url"] == "https://www.youtube.com/watch?v=abc123"
        assert document.metadata["chunk_index"] == index
        assert document.metadata["chunk_size"] == len(document.page_content)


def test_chunk_text_rejects_empty_text(chunker: TextChunkingService) -> None:
    """Verify empty text is rejected before chunking."""
    with pytest.raises(ValidationError, match="Text to chunk must not be empty"):
        chunker.chunk_text(text="", metadata={})


def test_chunk_text_rejects_whitespace_text(chunker: TextChunkingService) -> None:
    """Verify whitespace-only text is rejected before chunking."""
    with pytest.raises(ValidationError, match="Text to chunk must not be empty"):
        chunker.chunk_text(text="   \n\t  ", metadata={})


def test_chunk_text_rejects_non_dict_metadata(chunker: TextChunkingService) -> None:
    """Verify metadata must be a dictionary."""
    with pytest.raises(ValidationError, match="metadata must be a dictionary"):
        chunker.chunk_text(
            text="Valid text for chunking.",
            metadata="bad-metadata",  # type: ignore[arg-type]
        )


def test_chunk_text_rejects_invalid_chunk_configuration() -> None:
    """Verify invalid chunk overlap configuration is rejected."""
    bad_settings = Mock(
        chunk_size=100,
        chunk_overlap=100,
    )

    chunker = TextChunkingService(settings=bad_settings)

    with pytest.raises(ValidationError):
        chunker.chunk_text(
            text="This is valid text, but config is invalid.",
            metadata={},
        )


def test_chunk_text_creates_multiple_chunks_for_long_text(chunker: TextChunkingService) -> None:
    """Verify long text is split into multiple chunks."""
    text = "This is a long transcript sentence. " * 100

    documents = chunker.chunk_text(
        text=text,
        metadata={"video_id": "video-1"},
    )

    assert len(documents) > 1


def test_chunk_text_removes_empty_chunks(chunker: TextChunkingService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify empty chunks produced by splitter are removed."""
    monkeypatch.setattr(
        chunker._splitter,
        "split_text",
        lambda text: ["valid chunk", " ", "", "\n", "second valid chunk"],
    )

    documents = chunker.chunk_text(
        text="Input text that will be monkeypatched.",
        metadata={"video_id": "video-1"},
    )

    assert len(documents) == 2
    assert documents[0].page_content == "valid chunk"
    assert documents[1].page_content == "second valid chunk"


def test_chunk_to_domain_documents_returns_domain_chunks(chunker: TextChunkingService) -> None:
    """Verify chunk_to_domain_documents returns internal domain chunk models."""
    chunks = chunker.chunk_to_domain_documents(
        transcript_id="transcript-1",
        text="This is a test transcript. " * 20,
        metadata={"video_id": "video-1"},
    )

    assert chunks
    assert chunks[0].transcript_id == "transcript-1"
    assert chunks[0].chunk_index == 0
    assert chunks[0].text
    assert chunks[0].metadata["transcript_id"] == "transcript-1"


def test_chunk_to_domain_documents_rejects_empty_transcript_id(
    chunker: TextChunkingService,
) -> None:
    """Verify domain chunking rejects empty transcript ID."""
    with pytest.raises(ValidationError, match="transcript_id must not be empty"):
        chunker.chunk_to_domain_documents(
            transcript_id="",
            text="This is valid transcript text.",
            metadata={},
        )


def test_chunk_to_domain_documents_assigns_sequential_indexes(
    chunker: TextChunkingService,
) -> None:
    """Verify domain chunks receive sequential indexes."""
    chunks = chunker.chunk_to_domain_documents(
        transcript_id="transcript-1",
        text="This is a test transcript. " * 100,
        metadata={"video_id": "video-1"},
    )

    assert chunks

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.metadata["chunk_index"] == index
