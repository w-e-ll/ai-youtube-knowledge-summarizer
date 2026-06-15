from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ai_youtube_knowledge_summarizer.core.exceptions import RetrievalError, ValidationError
from ai_youtube_knowledge_summarizer.models.domain import QuestionAnswerDocument
from ai_youtube_knowledge_summarizer.services.qa_service import QAService


@pytest.fixture
def settings() -> Mock:
    """Create minimal settings object for QA service tests."""
    return Mock(
        openai_api_key="sk-test",
        openai_chat_model="gpt-4o-mini",
        openai_temperature=0.0,
        request_timeout_seconds=60,
        max_retries=1,
        retriever_top_k=4,
        openai_embedding_model="text-embedding-3-small",
        vector_store_provider="chroma",
        vector_store_dir="tmp/vector_store",
        transcripts_dir="tmp/transcripts",
    )


@pytest.fixture
def qa_service(monkeypatch: pytest.MonkeyPatch, settings: Mock) -> QAService:
    """Create QA service with external dependencies mocked."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.qa_service.ChatOpenAI",
        lambda **kwargs: Mock(name="ChatOpenAI"),
    )

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.qa_service.VectorStoreService",
        lambda settings, logger: Mock(name="VectorStoreService"),
    )

    return QAService(settings=settings)


def test_validate_configuration_requires_openai_key(settings: Mock) -> None:
    """Verify QA service rejects missing OpenAI API key."""
    settings.openai_api_key = None

    with pytest.raises(RetrievalError, match="OPENAI_API_KEY is required"):
        QAService(settings=settings)


def test_validate_request_accepts_valid_values(qa_service: QAService) -> None:
    """Verify QA request validation accepts valid inputs."""
    qa_service._validate_request(
        transcript_id="transcript-1",
        question="What is this video about?",
        top_k=4,
    )


def test_validate_request_rejects_empty_transcript_id(qa_service: QAService) -> None:
    """Verify QA request validation rejects empty transcript ID."""
    with pytest.raises(ValidationError, match="transcript_id must not be empty"):
        qa_service._validate_request(
            transcript_id="",
            question="What is this video about?",
            top_k=4,
        )


def test_validate_request_rejects_empty_question(qa_service: QAService) -> None:
    """Verify QA request validation rejects empty question."""
    with pytest.raises(ValidationError, match="question must not be empty"):
        qa_service._validate_request(
            transcript_id="transcript-1",
            question="",
            top_k=4,
        )


def test_validate_request_rejects_short_question(qa_service: QAService) -> None:
    """Verify QA request validation rejects too-short question."""
    with pytest.raises(ValidationError, match="at least 3 characters"):
        qa_service._validate_request(
            transcript_id="transcript-1",
            question="a",
            top_k=4,
        )


def test_validate_request_rejects_invalid_top_k(qa_service: QAService) -> None:
    """Verify QA request validation rejects invalid top_k."""
    with pytest.raises(ValidationError, match="top_k"):
        qa_service._validate_request(
            transcript_id="transcript-1",
            question="What is this video about?",
            top_k=0,
        )

    with pytest.raises(ValidationError, match="top_k"):
        qa_service._validate_request(
            transcript_id="transcript-1",
            question="What is this video about?",
            top_k=21,
        )


def test_extract_answer_from_result_key(qa_service: QAService) -> None:
    """Verify answer extraction supports RetrievalQA result key."""
    answer = qa_service._extract_answer(result={"result": "Generated answer"})

    assert answer == "Generated answer"


def test_extract_answer_from_answer_key(qa_service: QAService) -> None:
    """Verify answer extraction supports alternative answer key."""
    answer = qa_service._extract_answer(result={"answer": "Generated answer"})

    assert answer == "Generated answer"


def test_extract_answer_rejects_empty_output(qa_service: QAService) -> None:
    """Verify empty QA output raises RetrievalError."""
    with pytest.raises(RetrievalError, match="empty answer"):
        qa_service._extract_answer(result={"result": ""})


def test_extract_sources_returns_documents(qa_service: QAService) -> None:
    """Verify source extraction converts LangChain documents into dictionaries."""
    document = Document(
        page_content="Relevant transcript chunk",
        metadata={
            "transcript_id": "transcript-1",
            "chunk_index": 0,
        },
    )

    sources = qa_service._extract_sources(
        result={
            "source_documents": [document],
        }
    )

    assert len(sources) == 1
    assert sources[0]["content"] == "Relevant transcript chunk"
    assert sources[0]["score"] is None
    assert sources[0]["metadata"]["transcript_id"] == "transcript-1"
    assert sources[0]["metadata"]["chunk_index"] == 0
    assert sources[0]["metadata"]["source_index"] == 0


def test_extract_sources_ignores_invalid_documents(qa_service: QAService) -> None:
    """Verify source extraction ignores non-Document values."""
    sources = qa_service._extract_sources(
        result={
            "source_documents": ["bad-document", object()],
        }
    )

    assert sources == []


def test_extract_sources_returns_empty_for_non_dict_result(qa_service: QAService) -> None:
    """Verify source extraction returns empty list for unexpected result type."""
    sources = qa_service._extract_sources(result="plain answer")

    assert sources == []


def test_answer_question_success(qa_service: QAService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify answer_question returns domain QA document."""
    mock_retriever = Mock(name="retriever")
    qa_service._vector_store.as_retriever.return_value = mock_retriever

    mock_chain = Mock()
    mock_chain.invoke.return_value = {
        "result": "This video explains RAG systems.",
        "source_documents": [
            Document(
                page_content="RAG systems use retrieval and generation.",
                metadata={"chunk_index": 0},
            )
        ],
    }

    monkeypatch.setattr(
        qa_service,
        "_build_chain",
        lambda retriever: mock_chain,
    )

    result = qa_service.answer_question(
        transcript_id="transcript-1",
        question="What does the video explain?",
        top_k=4,
    )

    assert isinstance(result, QuestionAnswerDocument)
    assert result.transcript_id == "transcript-1"
    assert result.question == "What does the video explain?"
    assert result.answer == "This video explains RAG systems."
    assert len(result.sources) == 1

    qa_service._vector_store.as_retriever.assert_called_once_with(
        transcript_id="transcript-1",
        top_k=4,
    )
    mock_chain.invoke.assert_called_once_with({"query": "What does the video explain?"})


def test_answer_question_uses_default_top_k(qa_service: QAService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify answer_question uses configured top_k when not provided."""
    mock_chain = Mock()
    mock_chain.invoke.return_value = {
        "result": "Default top_k answer.",
        "source_documents": [],
    }

    monkeypatch.setattr(
        qa_service,
        "_build_chain",
        lambda retriever: mock_chain,
    )

    qa_service.answer_question(
        transcript_id="transcript-1",
        question="What is this about?",
        top_k=None,
    )

    qa_service._vector_store.as_retriever.assert_called_once_with(
        transcript_id="transcript-1",
        top_k=4,
    )


def test_answer_question_propagates_validation_error(qa_service: QAService) -> None:
    """Verify validation errors are propagated as typed errors."""
    with pytest.raises(ValidationError):
        qa_service.answer_question(
            transcript_id="",
            question="What is this about?",
            top_k=4,
        )


def test_answer_question_wraps_unexpected_error(
    qa_service: QAService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unexpected QA failures are wrapped as RetrievalError."""
    monkeypatch.setattr(
        qa_service,
        "_build_chain",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(RetrievalError, match="Failed to answer question"):
        qa_service.answer_question(
            transcript_id="transcript-1",
            question="What is this about?",
            top_k=4,
        )
