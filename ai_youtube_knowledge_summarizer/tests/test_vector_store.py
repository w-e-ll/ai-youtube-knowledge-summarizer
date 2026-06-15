from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ai_youtube_knowledge_summarizer.core.exceptions import (
    EmbeddingError,
    ValidationError,
    VectorStoreDocumentNotFoundError,
    VectorStoreError,
)
from ai_youtube_knowledge_summarizer.services.vector_store import VectorStoreService


@pytest.fixture
def settings(tmp_path: Path) -> Mock:
    """Create minimal settings object for vector store tests."""
    return Mock(
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-3-small",
        vector_store_provider="chroma",
        vector_store_dir=tmp_path / "vector_store",
        transcripts_dir=tmp_path / "transcripts",
        retriever_top_k=4,
    )


@pytest.fixture
def vector_store(monkeypatch: pytest.MonkeyPatch, settings: Mock) -> VectorStoreService:
    """Create vector store service with embeddings mocked."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.vector_store.OpenAIEmbeddings",
        lambda **kwargs: Mock(name="OpenAIEmbeddings"),
    )

    return VectorStoreService(settings=settings)


def test_validate_configuration_accepts_chroma_provider(vector_store: VectorStoreService) -> None:
    """Verify vector store configuration accepts Chroma."""
    vector_store._validate_configuration()


def test_validate_configuration_rejects_unsupported_provider(settings: Mock) -> None:
    """Verify unsupported vector store provider is rejected."""
    settings.vector_store_provider = "pinecone"

    with pytest.raises(VectorStoreError, match="Unsupported vector store provider"):
        VectorStoreService(settings=settings)


def test_validate_configuration_requires_openai_api_key(settings: Mock) -> None:
    """Verify OpenAI API key is required for embeddings."""
    settings.openai_api_key = None

    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY is required"):
        VectorStoreService(settings=settings)


def test_create_embeddings_wraps_failure(
    monkeypatch: pytest.MonkeyPatch,
    settings: Mock,
) -> None:
    """Verify embedding client creation failures are wrapped."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.vector_store.OpenAIEmbeddings",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(EmbeddingError, match="Failed to create OpenAI embeddings client"):
        VectorStoreService(settings=settings)


def test_validate_transcript_id_accepts_valid_value(vector_store: VectorStoreService) -> None:
    """Verify transcript ID validation accepts non-empty values."""
    vector_store._validate_transcript_id("transcript-1")


def test_validate_transcript_id_rejects_empty_value(vector_store: VectorStoreService) -> None:
    """Verify transcript ID validation rejects empty values."""
    with pytest.raises(ValidationError, match="transcript_id must not be empty"):
        vector_store._validate_transcript_id("")


def test_validate_query_accepts_valid_value(vector_store: VectorStoreService) -> None:
    """Verify query validation accepts non-empty values."""
    vector_store._validate_query("What is this about?")


def test_validate_query_rejects_empty_value(vector_store: VectorStoreService) -> None:
    """Verify query validation rejects empty query."""
    with pytest.raises(ValidationError, match="query must not be empty"):
        vector_store._validate_query("")


def test_validate_documents_accepts_valid_documents(vector_store: VectorStoreService) -> None:
    """Verify document validation accepts valid LangChain documents."""
    vector_store._validate_documents(
        [
            Document(
                page_content="Valid content",
                metadata={"chunk_index": 0},
            )
        ]
    )


def test_validate_documents_rejects_empty_list(vector_store: VectorStoreService) -> None:
    """Verify document validation rejects empty list."""
    with pytest.raises(ValidationError, match="documents must not be empty"):
        vector_store._validate_documents([])


def test_validate_documents_rejects_empty_page_content(
    vector_store: VectorStoreService,
) -> None:
    """Verify document validation rejects empty page content."""
    with pytest.raises(ValidationError, match="page_content"):
        vector_store._validate_documents(
            [
                Document(
                    page_content="   ",
                    metadata={"chunk_index": 0},
                )
            ]
        )


def test_collection_name_is_stable_and_chroma_safe(vector_store: VectorStoreService) -> None:
    """Verify collection name is normalized for Chroma."""
    collection_name = vector_store._collection_name("abc-123:def")

    assert collection_name == "transcript_abc_123_def"


def test_collection_path_uses_vector_store_dir(vector_store: VectorStoreService, settings: Mock) -> None:
    """Verify collection path is built under configured vector store directory."""
    path = vector_store._collection_path("transcript-1")

    assert path == settings.vector_store_dir / "transcript-1"


def test_transcript_cache_path_uses_transcripts_dir(
    vector_store: VectorStoreService,
    settings: Mock,
) -> None:
    """Verify transcript cache path is built under transcript directory."""
    path = vector_store._transcript_cache_path("transcript-1")

    assert path == settings.transcripts_dir / "transcript-1.indexed.txt"


def test_normalize_documents_adds_transcript_metadata(vector_store: VectorStoreService) -> None:
    """Verify document normalization injects transcript metadata."""
    documents = [
        Document(
            page_content=" Chunk text ",
            metadata={"chunk_index": 7, "title": "Video"},
        )
    ]

    normalized = vector_store._normalize_documents(
        transcript_id="transcript-1",
        documents=documents,
    )

    assert len(normalized) == 1
    assert normalized[0].page_content == "Chunk text"
    assert normalized[0].metadata["transcript_id"] == "transcript-1"
    assert normalized[0].metadata["chunk_index"] == 7
    assert normalized[0].metadata["title"] == "Video"


def test_normalize_documents_uses_index_when_chunk_index_missing(
    vector_store: VectorStoreService,
) -> None:
    """Verify normalization assigns chunk index when metadata lacks one."""
    documents = [
        Document(page_content="First chunk", metadata={}),
        Document(page_content="Second chunk", metadata={}),
    ]

    normalized = vector_store._normalize_documents(
        transcript_id="transcript-1",
        documents=documents,
    )

    assert normalized[0].metadata["chunk_index"] == 0
    assert normalized[1].metadata["chunk_index"] == 1


def test_persist_transcript_text_writes_sorted_chunks(
    vector_store: VectorStoreService,
    settings: Mock,
) -> None:
    """Verify transcript cache persists chunks sorted by chunk_index."""
    documents = [
        Document(page_content="Second chunk", metadata={"chunk_index": 1}),
        Document(page_content="First chunk", metadata={"chunk_index": 0}),
    ]

    vector_store._persist_transcript_text(
        transcript_id="transcript-1",
        documents=documents,
    )

    cache_path = settings.transcripts_dir / "transcript-1.indexed.txt"

    assert cache_path.exists()
    assert cache_path.read_text(encoding="utf-8") == "First chunk\n\nSecond chunk"


def test_persist_transcript_text_wraps_write_failure(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify transcript cache write failures are wrapped."""
    monkeypatch.setattr(
        Path,
        "write_text",
        Mock(side_effect=OSError("disk error")),
    )

    with pytest.raises(VectorStoreError, match="Failed to persist transcript text cache"):
        vector_store._persist_transcript_text(
            transcript_id="transcript-1",
            documents=[
                Document(page_content="Chunk", metadata={"chunk_index": 0}),
            ],
        )


def test_get_transcript_text_reads_cached_text(
    vector_store: VectorStoreService,
    settings: Mock,
) -> None:
    """Verify cached transcript text can be loaded."""
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.transcripts_dir / "transcript-1.indexed.txt"
    cache_path.write_text("Cached transcript", encoding="utf-8")

    text = vector_store.get_transcript_text("transcript-1")

    assert text == "Cached transcript"


def test_get_transcript_text_rejects_missing_cache(vector_store: VectorStoreService) -> None:
    """Verify missing transcript cache raises not-found error."""
    with pytest.raises(VectorStoreDocumentNotFoundError, match="not found"):
        vector_store.get_transcript_text("missing-transcript")


def test_get_transcript_text_rejects_empty_cache(
    vector_store: VectorStoreService,
    settings: Mock,
) -> None:
    """Verify empty transcript cache raises not-found error."""
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.transcripts_dir / "transcript-1.indexed.txt"
    cache_path.write_text("   ", encoding="utf-8")

    with pytest.raises(VectorStoreDocumentNotFoundError, match="empty"):
        vector_store.get_transcript_text("transcript-1")


def test_add_documents_success(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify add_documents stores documents and returns generated IDs."""
    mock_collection = Mock()
    monkeypatch.setattr(
        vector_store,
        "_get_collection",
        Mock(return_value=mock_collection),
    )

    documents = [
        Document(page_content="First chunk", metadata={"chunk_index": 0}),
        Document(page_content="Second chunk", metadata={"chunk_index": 1}),
    ]

    ids = vector_store.add_documents(
        transcript_id="transcript-1",
        documents=documents,
    )

    assert ids == ["transcript-1:0", "transcript-1:1"]
    mock_collection.add_documents.assert_called_once()


def test_add_documents_propagates_validation_error(vector_store: VectorStoreService) -> None:
    """Verify add_documents propagates validation errors."""
    with pytest.raises(ValidationError):
        vector_store.add_documents(
            transcript_id="",
            documents=[Document(page_content="Chunk", metadata={})],
        )


def test_add_documents_wraps_unexpected_failure(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unexpected add_documents failures are wrapped."""
    monkeypatch.setattr(
        vector_store,
        "_get_collection",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(VectorStoreError, match="Failed to add documents"):
        vector_store.add_documents(
            transcript_id="transcript-1",
            documents=[Document(page_content="Chunk", metadata={})],
        )


def test_similarity_search_success(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify similarity_search returns documents from vector collection."""
    expected_docs = [
        Document(page_content="Relevant chunk", metadata={"chunk_index": 0}),
    ]

    mock_collection = Mock()
    mock_collection.similarity_search.return_value = expected_docs

    monkeypatch.setattr(
        vector_store,
        "_get_collection",
        Mock(return_value=mock_collection),
    )

    result = vector_store.similarity_search(
        transcript_id="transcript-1",
        query="What is this about?",
        top_k=3,
    )

    assert result == expected_docs
    mock_collection.similarity_search.assert_called_once_with(
        query="What is this about?",
        k=3,
    )


def test_similarity_search_uses_default_top_k(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify similarity_search uses configured top_k when not provided."""
    mock_collection = Mock()
    mock_collection.similarity_search.return_value = []

    monkeypatch.setattr(
        vector_store,
        "_get_collection",
        Mock(return_value=mock_collection),
    )

    vector_store.similarity_search(
        transcript_id="transcript-1",
        query="What is this about?",
        top_k=None,
    )

    mock_collection.similarity_search.assert_called_once_with(
        query="What is this about?",
        k=4,
    )


def test_similarity_search_propagates_validation_error(vector_store: VectorStoreService) -> None:
    """Verify similarity_search propagates validation errors."""
    with pytest.raises(ValidationError):
        vector_store.similarity_search(
            transcript_id="transcript-1",
            query="",
            top_k=4,
        )


def test_as_retriever_returns_retriever(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify as_retriever returns retriever from vector collection."""
    mock_retriever = Mock(name="retriever")

    mock_collection = Mock()
    mock_collection.as_retriever.return_value = mock_retriever

    monkeypatch.setattr(
        vector_store,
        "_get_collection",
        Mock(return_value=mock_collection),
    )

    retriever = vector_store.as_retriever(
        transcript_id="transcript-1",
        top_k=5,
    )

    assert retriever is mock_retriever
    mock_collection.as_retriever.assert_called_once_with(
        search_kwargs={"k": 5}
    )


def test_as_retriever_uses_default_top_k(
    vector_store: VectorStoreService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify as_retriever uses configured top_k when not provided."""
    mock_collection = Mock()
    mock_collection.as_retriever.return_value = Mock(name="retriever")

    monkeypatch.setattr(
        vector_store,
        "_get_collection",
        Mock(return_value=mock_collection),
    )

    vector_store.as_retriever(
        transcript_id="transcript-1",
        top_k=None,
    )

    mock_collection.as_retriever.assert_called_once_with(
        search_kwargs={"k": 4}
    )
