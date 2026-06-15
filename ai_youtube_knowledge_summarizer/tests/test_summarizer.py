from __future__ import annotations

from unittest.mock import Mock

import pytest

from ai_youtube_knowledge_summarizer.core.exceptions import SummarizationError, ValidationError
from ai_youtube_knowledge_summarizer.services.summarizer import SummarizationService


@pytest.fixture
def settings() -> Mock:
    """Create minimal settings object for summarizer tests."""
    return Mock(
        openai_api_key="sk-test",
        openai_chat_model="gpt-4o-mini",
        openai_temperature=0.0,
        request_timeout_seconds=60,
        max_retries=1,
    )


@pytest.fixture
def summarizer(monkeypatch: pytest.MonkeyPatch, settings: Mock) -> SummarizationService:
    """Create summarization service with external LLM client mocked."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.summarizer.ChatOpenAI",
        lambda **kwargs: Mock(name="ChatOpenAI"),
    )

    return SummarizationService(settings=settings)


def test_validate_configuration_requires_openai_key(settings: Mock) -> None:
    """Verify summarizer rejects missing OpenAI API key."""
    settings.openai_api_key = None

    with pytest.raises(SummarizationError, match="OPENAI_API_KEY is required"):
        SummarizationService(settings=settings)


def test_validate_text_accepts_non_empty_text(summarizer: SummarizationService) -> None:
    """Verify text validation accepts valid transcript text."""
    summarizer._validate_text("This is a valid transcript.")


def test_validate_text_rejects_empty_text(summarizer: SummarizationService) -> None:
    """Verify text validation rejects empty transcript text."""
    with pytest.raises(ValidationError, match="Text to summarize must not be empty"):
        summarizer._validate_text("")


def test_validate_text_rejects_whitespace_text(summarizer: SummarizationService) -> None:
    """Verify text validation rejects whitespace-only transcript text."""
    with pytest.raises(ValidationError, match="Text to summarize must not be empty"):
        summarizer._validate_text("   \n\t  ")


def test_validate_mode_accepts_supported_modes(summarizer: SummarizationService) -> None:
    """Verify all supported summary modes are accepted."""
    summarizer._validate_mode("stuff")
    summarizer._validate_mode("map_reduce")
    summarizer._validate_mode("refine")


def test_validate_mode_rejects_unsupported_mode(summarizer: SummarizationService) -> None:
    """Verify unsupported summary mode is rejected."""
    with pytest.raises(ValidationError, match="Unsupported summary mode"):
        summarizer._validate_mode("bad-mode")


def test_build_documents_creates_single_transcript_document(
    summarizer: SummarizationService,
) -> None:
    """Verify transcript text is wrapped into a LangChain document."""
    documents = summarizer._build_documents("Transcript text")

    assert len(documents) == 1
    assert documents[0].page_content == "Transcript text"
    assert documents[0].metadata["source"] == "transcript"


def test_extract_summary_from_output_text(summarizer: SummarizationService) -> None:
    """Verify summary extraction supports output_text key."""
    summary = summarizer._extract_summary(result={"output_text": "Generated summary"})

    assert summary == "Generated summary"


def test_extract_summary_from_text_key(summarizer: SummarizationService) -> None:
    """Verify summary extraction supports text key."""
    summary = summarizer._extract_summary(result={"text": "Generated summary"})

    assert summary == "Generated summary"


def test_extract_summary_from_plain_result(summarizer: SummarizationService) -> None:
    """Verify summary extraction supports plain non-dict result."""
    summary = summarizer._extract_summary(result="Generated summary")

    assert summary == "Generated summary"


def test_extract_summary_rejects_empty_output(summarizer: SummarizationService) -> None:
    """Verify empty summarization output is rejected."""
    with pytest.raises(SummarizationError, match="empty output"):
        summarizer._extract_summary(result={"output_text": ""})


def test_prompt_factories_return_prompt_templates(summarizer: SummarizationService) -> None:
    """Verify prompt factory methods return usable prompt templates."""
    assert summarizer._stuff_prompt().format(text="Transcript")
    assert summarizer._map_prompt().format(text="Transcript part")
    assert summarizer._combine_prompt().format(text="Partial summaries")
    assert summarizer._refine_question_prompt().format(text="Transcript section")
    assert summarizer._refine_prompt().format(
        existing_answer="Existing summary",
        text="Additional transcript section",
    )


def test_build_chain_supports_all_modes(
    summarizer: SummarizationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify chain builder supports all configured summary modes."""
    mock_chain = Mock(name="summary_chain")

    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.summarizer.load_summarize_chain",
        Mock(return_value=mock_chain),
    )

    assert summarizer._build_chain(mode="stuff") is mock_chain
    assert summarizer._build_chain(mode="map_reduce") is mock_chain
    assert summarizer._build_chain(mode="refine") is mock_chain


def test_build_chain_wraps_unexpected_error(
    summarizer: SummarizationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify chain builder wraps unexpected failures."""
    monkeypatch.setattr(
        "ai_youtube_knowledge_summarizer.services.summarizer.load_summarize_chain",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(SummarizationError, match="Failed to build summarization chain"):
        summarizer._build_chain(mode="stuff")


def test_summarize_success(
    summarizer: SummarizationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify summarize returns chain output."""
    mock_chain = Mock()
    mock_chain.invoke.return_value = {"output_text": "Generated summary"}

    monkeypatch.setattr(
        summarizer,
        "_build_chain",
        Mock(return_value=mock_chain),
    )

    result = summarizer.summarize(
        "This is a transcript.",
        mode="map_reduce",
    )

    assert result == "Generated summary"
    mock_chain.invoke.assert_called_once()


def test_summarize_propagates_validation_error(summarizer: SummarizationService) -> None:
    """Verify summarize propagates validation errors."""
    with pytest.raises(ValidationError):
        summarizer.summarize("", mode="map_reduce")


def test_summarize_wraps_unexpected_error(
    summarizer: SummarizationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unexpected summarization failures are wrapped."""
    monkeypatch.setattr(
        summarizer,
        "_build_chain",
        Mock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(SummarizationError, match="Failed to summarize transcript text"):
        summarizer.summarize("This is valid transcript text.", mode="map_reduce")
