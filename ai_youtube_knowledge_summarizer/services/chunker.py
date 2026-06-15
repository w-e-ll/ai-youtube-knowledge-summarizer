from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_youtube_knowledge_summarizer.core.config import Settings
from ai_youtube_knowledge_summarizer.core.exceptions import ChunkingError, ValidationError
from ai_youtube_knowledge_summarizer.core.logging import log_failure, log_step
from ai_youtube_knowledge_summarizer.models.domain import ChunkDocument


logger = logging.getLogger("ai_youtube_knowledge_summarizer.services.chunker")


class TextChunkingService:
    """Split transcript text into metadata-aware chunks."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Initialize chunking service with configured chunk strategy."""
        self.settings = settings
        self.logger = logger or logging.getLogger(
            "ai_youtube_knowledge_summarizer.services.chunker"
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        log_step(
            self.logger,
            event="chunker_initialized",
            operation="chunker_init",
            status="success",
            message="Text chunking service initialized",
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def chunk_text(
        self,
        *,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Split text into LangChain documents with normalized metadata."""
        started_at = time.perf_counter()
        metadata = metadata or {}

        log_step(
            self.logger,
            event="chunking_requested",
            operation="chunk_text",
            status="started",
            message="Transcript chunking requested",
            text_length=len(text or ""),
            metadata_keys=list(metadata.keys()),
        )

        try:
            self._validate_input(text=text, metadata=metadata)

            raw_chunks = self._split_text(text=text)

            documents = self._build_documents(
                raw_chunks=raw_chunks,
                metadata=metadata,
            )

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="chunking_completed",
                operation="chunk_text",
                status="success",
                message="Transcript chunking completed successfully",
                chunk_count=len(documents),
                duration_ms=duration_ms,
            )

            return documents

        except (ValidationError, ChunkingError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="chunking_failed",
                operation="chunk_text",
                message="Unexpected chunking failure",
                exc=exc,
                text_length=len(text or ""),
            )
            raise ChunkingError(
                "Unexpected failure while chunking transcript text",
                details={"exception_type": type(exc).__name__},
            ) from exc

    def chunk_to_domain_documents(
        self,
        *,
        transcript_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkDocument]:
        """Split text into internal domain chunk documents."""
        started_at = time.perf_counter()
        metadata = metadata or {}

        log_step(
            self.logger,
            event="domain_chunking_requested",
            operation="chunk_to_domain_documents",
            status="started",
            message="Domain chunk generation requested",
            transcript_id=transcript_id,
            text_length=len(text or ""),
        )

        try:
            if not transcript_id or not transcript_id.strip():
                raise ValidationError("transcript_id must not be empty")

            langchain_docs = self.chunk_text(text=text, metadata=metadata)

            chunks = [
                ChunkDocument(
                    transcript_id=transcript_id,
                    chunk_index=index,
                    text=document.page_content,
                    metadata={
                        **document.metadata,
                        "transcript_id": transcript_id,
                        "chunk_index": index,
                    },
                )
                for index, document in enumerate(langchain_docs)
            ]

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="domain_chunking_completed",
                operation="chunk_to_domain_documents",
                status="success",
                message="Domain chunk generation completed",
                transcript_id=transcript_id,
                chunk_count=len(chunks),
                duration_ms=duration_ms,
            )

            return chunks

        except (ValidationError, ChunkingError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="domain_chunking_failed",
                operation="chunk_to_domain_documents",
                message="Failed to generate domain chunks",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise ChunkingError(
                "Failed to generate domain chunk documents",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_input(self, *, text: str, metadata: dict[str, Any]) -> None:
        """Validate text and metadata before chunking."""
        log_step(
            self.logger,
            event="chunking_input_validation_started",
            operation="validate_chunking_input",
            status="started",
            message="Validating chunking input",
            text_length=len(text or ""),
        )

        if not text or not text.strip():
            raise ValidationError("Text to chunk must not be empty")

        if not isinstance(metadata, dict):
            raise ValidationError("metadata must be a dictionary")

        if self.settings.chunk_overlap >= self.settings.chunk_size:
            raise ChunkingError(
                "Invalid chunking configuration",
                details={
                    "chunk_size": self.settings.chunk_size,
                    "chunk_overlap": self.settings.chunk_overlap,
                },
            )

        log_step(
            self.logger,
            event="chunking_input_validation_completed",
            operation="validate_chunking_input",
            status="success",
            message="Chunking input validation completed",
            text_length=len(text),
        )

    def _split_text(self, *, text: str) -> list[str]:
        """Split text using configured recursive splitter."""
        log_step(
            self.logger,
            event="text_split_started",
            operation="split_text",
            status="started",
            message="Splitting transcript text",
            text_length=len(text),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        try:
            chunks = self._splitter.split_text(text)

            chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

            if not chunks:
                raise ChunkingError("Text splitter produced no chunks")

            log_step(
                self.logger,
                event="text_split_completed",
                operation="split_text",
                status="success",
                message="Transcript text split completed",
                chunk_count=len(chunks),
            )

            return chunks

        except ChunkingError:
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="text_split_failed",
                operation="split_text",
                message="Text splitting failed",
                exc=exc,
                text_length=len(text),
            )
            raise ChunkingError(
                "Text splitting failed",
                details={"exception_type": type(exc).__name__},
            ) from exc

    def _build_documents(
        self,
        *,
        raw_chunks: list[str],
        metadata: dict[str, Any],
    ) -> list[Document]:
        """Build LangChain documents from raw chunk strings."""
        log_step(
            self.logger,
            event="document_build_started",
            operation="build_chunk_documents",
            status="started",
            message="Building LangChain chunk documents",
            chunk_count=len(raw_chunks),
        )

        documents: list[Document] = []

        for index, chunk in enumerate(raw_chunks):
            document_metadata = {
                **metadata,
                "chunk_index": index,
                "chunk_size": len(chunk),
            }

            documents.append(
                Document(
                    page_content=chunk,
                    metadata=document_metadata,
                )
            )

        log_step(
            self.logger,
            event="document_build_completed",
            operation="build_chunk_documents",
            status="success",
            message="LangChain chunk documents built",
            chunk_count=len(documents),
        )

        return documents
