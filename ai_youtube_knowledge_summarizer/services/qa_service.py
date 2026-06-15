from __future__ import annotations

import logging
import time
from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ai_youtube_knowledge_summarizer.core.config import Settings
from ai_youtube_knowledge_summarizer.core.exceptions import RetrievalError, ValidationError
from ai_youtube_knowledge_summarizer.core.logging import log_failure, log_step
from ai_youtube_knowledge_summarizer.models.domain import QuestionAnswerDocument
from ai_youtube_knowledge_summarizer.services.vector_store import VectorStoreService


logger = logging.getLogger("ai_youtube_knowledge_summarizer.services.qa_service")


class QAService:
    """Answer questions using retrieval over indexed transcript chunks."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Initialize QA service with LLM and vector store dependencies."""
        self.settings = settings
        self.logger = logger or logging.getLogger(
            "ai_youtube_knowledge_summarizer.services.qa_service"
        )

        log_step(
            self.logger,
            event="qa_service_initialized",
            operation="qa_service_init",
            status="started",
            message="QA service initialization started",
            provider="openai",
            model=self.settings.openai_chat_model,
        )

        try:
            self._validate_configuration()
            self._llm = self._create_llm()
            self._vector_store = VectorStoreService(
                settings=self.settings,
                logger=self.logger,
            )

            log_step(
                self.logger,
                event="qa_service_ready",
                operation="qa_service_init",
                status="success",
                message="QA service initialized successfully",
                provider="openai",
                model=self.settings.openai_chat_model,
            )

        except (ValidationError, RetrievalError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="qa_service_init_failed",
                operation="qa_service_init",
                message="Failed to initialize QA service",
                exc=exc,
                model=self.settings.openai_chat_model,
            )
            raise RetrievalError(
                "Failed to initialize QA service",
                details={"exception_type": type(exc).__name__},
            ) from exc

    def answer_question(
        self,
        *,
        transcript_id: str,
        question: str,
        top_k: int | None = None,
    ) -> QuestionAnswerDocument:
        """Answer a question using transcript retrieval context."""
        started_at = time.perf_counter()
        resolved_top_k = top_k or self.settings.retriever_top_k

        log_step(
            self.logger,
            event="qa_requested",
            operation="answer_question",
            status="started",
            message="Retrieval QA requested",
            transcript_id=transcript_id,
            top_k=resolved_top_k,
            model=self.settings.openai_chat_model,
        )

        try:
            self._validate_request(
                transcript_id=transcript_id,
                question=question,
                top_k=resolved_top_k,
            )

            retriever = self._vector_store.as_retriever(
                transcript_id=transcript_id,
                top_k=resolved_top_k,
            )

            chain = self._build_chain(retriever=retriever)

            result = chain.invoke({"query": question})

            answer = self._extract_answer(result=result)
            sources = self._extract_sources(result=result)

            qa_result = QuestionAnswerDocument(
                transcript_id=transcript_id,
                question=question.strip(),
                answer=answer,
                sources=sources,
            )

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="qa_completed",
                operation="answer_question",
                status="success",
                message="Retrieval QA completed successfully",
                transcript_id=transcript_id,
                source_count=len(sources),
                answer_length=len(answer),
                duration_ms=duration_ms,
                model=self.settings.openai_chat_model,
            )

            return qa_result

        except (ValidationError, RetrievalError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="qa_failed",
                operation="answer_question",
                message="Retrieval QA failed",
                exc=exc,
                transcript_id=transcript_id,
                top_k=resolved_top_k,
            )
            raise RetrievalError(
                "Failed to answer question using transcript retrieval",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_configuration(self) -> None:
        """Validate QA service configuration."""
        log_step(
            self.logger,
            event="qa_config_validation_started",
            operation="validate_qa_config",
            status="started",
            message="Validating QA service configuration",
            model=self.settings.openai_chat_model,
        )

        if not self.settings.openai_api_key:
            raise RetrievalError(
                "OPENAI_API_KEY is required for retrieval QA",
                details={"model": self.settings.openai_chat_model},
            )

        log_step(
            self.logger,
            event="qa_config_validation_completed",
            operation="validate_qa_config",
            status="success",
            message="QA service configuration validation completed",
            model=self.settings.openai_chat_model,
        )

    def _create_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI client for retrieval QA."""
        log_step(
            self.logger,
            event="qa_llm_creation_started",
            operation="create_qa_llm",
            status="started",
            message="Creating ChatOpenAI client for QA",
            provider="openai",
            model=self.settings.openai_chat_model,
        )

        try:
            llm = ChatOpenAI(
                model=self.settings.openai_chat_model,
                temperature=0,
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout_seconds,
                max_retries=self.settings.max_retries,
            )

            log_step(
                self.logger,
                event="qa_llm_created",
                operation="create_qa_llm",
                status="success",
                message="ChatOpenAI client created for QA",
                provider="openai",
                model=self.settings.openai_chat_model,
            )

            return llm

        except Exception as exc:
            log_failure(
                self.logger,
                event="qa_llm_creation_failed",
                operation="create_qa_llm",
                message="Failed to create ChatOpenAI QA client",
                exc=exc,
                model=self.settings.openai_chat_model,
            )
            raise RetrievalError(
                "Failed to create ChatOpenAI QA client",
                details={
                    "model": self.settings.openai_chat_model,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_request(
        self,
        *,
        transcript_id: str,
        question: str,
        top_k: int,
    ) -> None:
        """Validate QA request values."""
        log_step(
            self.logger,
            event="qa_request_validation_started",
            operation="validate_qa_request",
            status="started",
            message="Validating QA request",
            transcript_id=transcript_id,
            top_k=top_k,
        )

        if not transcript_id or not transcript_id.strip():
            raise ValidationError("transcript_id must not be empty")

        if not question or not question.strip():
            raise ValidationError("question must not be empty")

        if len(question.strip()) < 3:
            raise ValidationError("question must contain at least 3 characters")

        if top_k < 1 or top_k > 20:
            raise ValidationError(
                "top_k must be between 1 and 20",
                details={"top_k": top_k},
            )

        log_step(
            self.logger,
            event="qa_request_validation_completed",
            operation="validate_qa_request",
            status="success",
            message="QA request validation completed",
            transcript_id=transcript_id,
            top_k=top_k,
        )

    def _build_chain(self, *, retriever: Any) -> RetrievalQA:
        """Build retrieval QA chain with source documents enabled."""
        log_step(
            self.logger,
            event="qa_chain_build_started",
            operation="build_qa_chain",
            status="started",
            message="Building RetrievalQA chain",
            model=self.settings.openai_chat_model,
        )

        try:
            chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self._qa_prompt(),
                },
            )

            log_step(
                self.logger,
                event="qa_chain_build_completed",
                operation="build_qa_chain",
                status="success",
                message="RetrievalQA chain built successfully",
                model=self.settings.openai_chat_model,
            )

            return chain

        except Exception as exc:
            log_failure(
                self.logger,
                event="qa_chain_build_failed",
                operation="build_qa_chain",
                message="Failed to build RetrievalQA chain",
                exc=exc,
            )
            raise RetrievalError(
                "Failed to build RetrievalQA chain",
                details={"exception_type": type(exc).__name__},
            ) from exc

    def _extract_answer(self, *, result: object) -> str:
        """Extract answer text from RetrievalQA result."""
        log_step(
            self.logger,
            event="qa_answer_extraction_started",
            operation="extract_qa_answer",
            status="started",
            message="Extracting QA answer",
        )

        if isinstance(result, dict):
            raw_answer = result.get("result") or result.get("answer") or ""
        else:
            raw_answer = str(result)

        answer = str(raw_answer).strip()

        if not answer:
            raise RetrievalError("RetrievalQA returned empty answer")

        log_step(
            self.logger,
            event="qa_answer_extraction_completed",
            operation="extract_qa_answer",
            status="success",
            message="QA answer extracted successfully",
            answer_length=len(answer),
        )

        return answer

    def _extract_sources(self, *, result: object) -> list[dict[str, Any]]:
        """Extract source documents from RetrievalQA result."""
        log_step(
            self.logger,
            event="qa_source_extraction_started",
            operation="extract_qa_sources",
            status="started",
            message="Extracting QA source documents",
        )

        if not isinstance(result, dict):
            return []

        source_documents = result.get("source_documents") or []

        sources: list[dict[str, Any]] = []

        for index, document in enumerate(source_documents):
            if not isinstance(document, Document):
                continue

            sources.append(
                {
                    "content": document.page_content,
                    "score": None,
                    "metadata": {
                        **document.metadata,
                        "source_index": index,
                    },
                }
            )

        log_step(
            self.logger,
            event="qa_source_extraction_completed",
            operation="extract_qa_sources",
            status="success",
            message="QA source documents extracted",
            source_count=len(sources),
        )

        return sources

    def _qa_prompt(self) -> PromptTemplate:
        """Return hallucination-resistant QA prompt."""
        return PromptTemplate.from_template(
            """
You are an AI assistant answering questions about a YouTube video transcript.

Use only the provided transcript context.
If the answer is not present in the context, say:
"I could not find this information in the transcript."

Rules:
- Do not invent facts.
- Do not use external knowledge.
- Answer clearly and concisely.
- Mention uncertainty when the transcript is incomplete.
- Prefer bullet points when helpful.

Transcript context:
{context}

Question:
{question}

Answer:
"""
        )
