from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ai_youtube_knowledge_summarizer.core.config import Settings
from ai_youtube_knowledge_summarizer.core.exceptions import SummarizationError, ValidationError
from ai_youtube_knowledge_summarizer.core.logging import log_failure, log_step


logger = logging.getLogger("ai_youtube_knowledge_summarizer.services.summarizer")


SummaryMode = Literal["stuff", "map_reduce", "refine"]


class SummarizationService:
    """Generate summaries from transcript text using LLM chains."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Initialize summarization service with configured LLM."""
        self.settings = settings
        self.logger = logger or logging.getLogger(
            "ai_youtube_knowledge_summarizer.services.summarizer"
        )

        log_step(
            self.logger,
            event="summarizer_initialized",
            operation="summarizer_init",
            status="started",
            message="Summarization service initialization started",
            provider="openai",
            model=self.settings.openai_chat_model,
        )

        try:
            self._validate_configuration()
            self._llm = self._create_llm()

            log_step(
                self.logger,
                event="summarizer_ready",
                operation="summarizer_init",
                status="success",
                message="Summarization service initialized successfully",
                provider="openai",
                model=self.settings.openai_chat_model,
            )

        except (ValidationError, SummarizationError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="summarizer_init_failed",
                operation="summarizer_init",
                message="Failed to initialize summarization service",
                exc=exc,
                model=self.settings.openai_chat_model,
            )
            raise SummarizationError(
                "Failed to initialize summarization service",
                details={"exception_type": type(exc).__name__},
            ) from exc

    def summarize(self, text: str, *, mode: SummaryMode = "map_reduce") -> str:
        """Generate a summary for transcript text."""
        started_at = time.perf_counter()

        log_step(
            self.logger,
            event="summarization_requested",
            operation="summarize_text",
            status="started",
            message="Transcript summarization requested",
            mode=mode,
            text_length=len(text or ""),
            model=self.settings.openai_chat_model,
        )

        try:
            self._validate_text(text)
            self._validate_mode(mode)

            documents = self._build_documents(text=text)

            chain = self._build_chain(mode=mode)

            result = chain.invoke({"input_documents": documents})

            summary = self._extract_summary(result=result)

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="summarization_completed",
                operation="summarize_text",
                status="success",
                message="Transcript summarization completed successfully",
                mode=mode,
                document_count=len(documents),
                summary_length=len(summary),
                duration_ms=duration_ms,
                model=self.settings.openai_chat_model,
            )

            return summary

        except (ValidationError, SummarizationError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="summarization_failed",
                operation="summarize_text",
                message="Transcript summarization failed",
                exc=exc,
                mode=mode,
                text_length=len(text or ""),
            )
            raise SummarizationError(
                "Failed to summarize transcript text",
                details={
                    "mode": mode,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_configuration(self) -> None:
        """Validate summarization configuration."""
        log_step(
            self.logger,
            event="summarizer_config_validation_started",
            operation="validate_summarizer_config",
            status="started",
            message="Validating summarization configuration",
            model=self.settings.openai_chat_model,
        )

        if not self.settings.openai_api_key:
            raise SummarizationError(
                "OPENAI_API_KEY is required for summarization",
                details={"model": self.settings.openai_chat_model},
            )

        log_step(
            self.logger,
            event="summarizer_config_validation_completed",
            operation="validate_summarizer_config",
            status="success",
            message="Summarization configuration validation completed",
            model=self.settings.openai_chat_model,
        )

    def _create_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI client for summarization."""
        log_step(
            self.logger,
            event="summarizer_llm_creation_started",
            operation="create_summarizer_llm",
            status="started",
            message="Creating ChatOpenAI client for summarization",
            provider="openai",
            model=self.settings.openai_chat_model,
        )

        try:
            llm = ChatOpenAI(
                model=self.settings.openai_chat_model,
                temperature=self.settings.openai_temperature,
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
            )

            log_step(
                self.logger,
                event="summarizer_llm_created",
                operation="create_summarizer_llm",
                status="success",
                message="ChatOpenAI client created for summarization",
                provider="openai",
                model=self.settings.openai_chat_model,
            )

            return llm

        except Exception as exc:
            log_failure(
                self.logger,
                event="summarizer_llm_creation_failed",
                operation="create_summarizer_llm",
                message="Failed to create ChatOpenAI summarization client",
                exc=exc,
                model=self.settings.openai_chat_model,
            )
            raise SummarizationError(
                "Failed to create ChatOpenAI summarization client",
                details={
                    "model": self.settings.openai_chat_model,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_text(self, text: str) -> None:
        """Validate transcript text before summarization."""
        log_step(
            self.logger,
            event="summarization_text_validation_started",
            operation="validate_summarization_text",
            status="started",
            message="Validating transcript text for summarization",
            text_length=len(text or ""),
        )

        if not text or not text.strip():
            raise ValidationError("Text to summarize must not be empty")

        log_step(
            self.logger,
            event="summarization_text_validation_completed",
            operation="validate_summarization_text",
            status="success",
            message="Transcript text validation completed",
            text_length=len(text),
        )

    def _validate_mode(self, mode: str) -> None:
        """Validate summarization mode."""
        log_step(
            self.logger,
            event="summary_mode_validation_started",
            operation="validate_summary_mode",
            status="started",
            message="Validating summary mode",
            mode=mode,
        )

        allowed_modes = {"stuff", "map_reduce", "refine"}

        if mode not in allowed_modes:
            raise ValidationError(
                "Unsupported summary mode",
                details={"mode": mode, "allowed_modes": sorted(allowed_modes)},
            )

        log_step(
            self.logger,
            event="summary_mode_validation_completed",
            operation="validate_summary_mode",
            status="success",
            message="Summary mode validation completed",
            mode=mode,
        )

    def _build_documents(self, text: str) -> list[Document]:
        """Build LangChain documents for summarization."""
        log_step(
            self.logger,
            event="summary_document_build_started",
            operation="build_summary_documents",
            status="started",
            message="Building summarization documents",
            text_length=len(text),
        )

        documents = [
            Document(
                page_content=text.strip(),
                metadata={"source": "transcript"},
            )
        ]

        log_step(
            self.logger,
            event="summary_document_build_completed",
            operation="build_summary_documents",
            status="success",
            message="Summarization documents built",
            document_count=len(documents),
        )

        return documents

    def _build_chain(self, *, mode: SummaryMode):
        """Build summarization chain for the selected mode."""
        log_step(
            self.logger,
            event="summarization_chain_build_started",
            operation="build_summarization_chain",
            status="started",
            message="Building summarization chain",
            mode=mode,
        )

        try:
            if mode == "stuff":
                chain = load_summarize_chain(
                    llm=self._llm,
                    chain_type="stuff",
                    prompt=self._stuff_prompt(),
                )
            elif mode == "map_reduce":
                chain = load_summarize_chain(
                    llm=self._llm,
                    chain_type="map_reduce",
                    map_prompt=self._map_prompt(),
                    combine_prompt=self._combine_prompt(),
                )
            elif mode == "refine":
                chain = load_summarize_chain(
                    llm=self._llm,
                    chain_type="refine",
                    question_prompt=self._refine_question_prompt(),
                    refine_prompt=self._refine_prompt(),
                )
            else:
                raise ValidationError("Unsupported summary mode", details={"mode": mode})

            log_step(
                self.logger,
                event="summarization_chain_build_completed",
                operation="build_summarization_chain",
                status="success",
                message="Summarization chain built successfully",
                mode=mode,
            )

            return chain

        except ValidationError:
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="summarization_chain_build_failed",
                operation="build_summarization_chain",
                message="Failed to build summarization chain",
                exc=exc,
                mode=mode,
            )
            raise SummarizationError(
                "Failed to build summarization chain",
                details={"mode": mode, "exception_type": type(exc).__name__},
            ) from exc

    def _extract_summary(self, *, result: object) -> str:
        """Extract summary text from LangChain result."""
        log_step(
            self.logger,
            event="summary_extraction_started",
            operation="extract_summary",
            status="started",
            message="Extracting summary from chain result",
        )

        if isinstance(result, dict):
            raw_summary = result.get("output_text") or result.get("text") or ""
        else:
            raw_summary = str(result)

        summary = str(raw_summary).strip()

        if not summary:
            raise SummarizationError("Summarization chain returned empty output")

        log_step(
            self.logger,
            event="summary_extraction_completed",
            operation="extract_summary",
            status="success",
            message="Summary extracted successfully",
            summary_length=len(summary),
        )

        return summary

    def _stuff_prompt(self) -> PromptTemplate:
        """Return prompt for direct transcript summarization."""
        return PromptTemplate.from_template(
            """
You are an AI media intelligence assistant.

Summarize the following YouTube transcript clearly and professionally.

Include:
- main topic
- key ideas
- important technical or business points
- practical conclusions
- concise bullet summary

Transcript:
{text}

Summary:
"""
        )

    def _map_prompt(self) -> PromptTemplate:
        """Return map prompt for map-reduce summarization."""
        return PromptTemplate.from_template(
            """
You are summarizing part of a YouTube transcript.

Extract the most important ideas, facts, decisions, and technical details from this part.

Transcript part:
{text}

Partial summary:
"""
        )

    def _combine_prompt(self) -> PromptTemplate:
        """Return combine prompt for map-reduce summarization."""
        return PromptTemplate.from_template(
            """
You are combining partial summaries into one final professional summary.

Create a clean, non-repetitive summary with:
- short overview
- key points
- important details
- practical takeaways
- possible action items if applicable

Partial summaries:
{text}

Final summary:
"""
        )

    def _refine_question_prompt(self) -> PromptTemplate:
        """Return initial prompt for refine summarization."""
        return PromptTemplate.from_template(
            """
Write an initial professional summary of this transcript section.

Transcript section:
{text}

Initial summary:
"""
        )

    def _refine_prompt(self) -> PromptTemplate:
        """Return refinement prompt for iterative summarization."""
        return PromptTemplate.from_template(
            """
We have an existing summary:

{existing_answer}

Refine and improve it using the additional transcript section below.
Only add important new information. Avoid repetition.

Additional transcript section:
{text}

Refined summary:
"""
        )
