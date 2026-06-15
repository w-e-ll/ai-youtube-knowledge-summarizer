from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ai_youtube_knowledge_summarizer.core.config import Settings
from ai_youtube_knowledge_summarizer.core.exceptions import (
    EmbeddingError,
    ValidationError,
    VectorStoreDocumentNotFoundError,
    VectorStoreError,
)
from ai_youtube_knowledge_summarizer.core.logging import log_failure, log_step


logger = logging.getLogger("ai_youtube_knowledge_summarizer.services.vector_store")


class VectorStoreService:
    """Manage transcript embeddings and vector retrieval."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Initialize vector store service with configured provider."""
        self.settings = settings
        self.logger = logger or logging.getLogger(
            "ai_youtube_knowledge_summarizer.services.vector_store"
        )

        log_step(
            self.logger,
            event="vector_store_service_initialized",
            operation="vector_store_init",
            status="started",
            message="Vector store service initialization started",
            provider=self.settings.vector_store_provider,
            model=self.settings.openai_embedding_model,
        )

        try:
            self._validate_configuration()

            self.settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

            self._embeddings = self._create_embeddings()

            log_step(
                self.logger,
                event="vector_store_service_ready",
                operation="vector_store_init",
                status="success",
                message="Vector store service initialized successfully",
                provider=self.settings.vector_store_provider,
                vector_store_dir=str(self.settings.vector_store_dir),
                model=self.settings.openai_embedding_model,
            )

        except (ValidationError, EmbeddingError, VectorStoreError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="vector_store_init_failed",
                operation="vector_store_init",
                message="Unexpected vector store initialization failure",
                exc=exc,
                provider=self.settings.vector_store_provider,
            )
            raise VectorStoreError(
                "Failed to initialize vector store service",
                details={"exception_type": type(exc).__name__},
            ) from exc

    def add_documents(
        self,
        *,
        transcript_id: str,
        documents: list[Document],
    ) -> list[str]:
        """Add transcript chunks to the vector store."""
        started_at = time.perf_counter()

        log_step(
            self.logger,
            event="vector_store_add_requested",
            operation="add_documents",
            status="started",
            message="Adding documents to vector store",
            transcript_id=transcript_id,
            document_count=len(documents or []),
        )

        try:
            self._validate_transcript_id(transcript_id)
            self._validate_documents(documents)

            collection = self._get_collection(transcript_id=transcript_id)

            normalized_documents = self._normalize_documents(
                transcript_id=transcript_id,
                documents=documents,
            )

            ids = [
                f"{transcript_id}:{index}"
                for index, _ in enumerate(normalized_documents)
            ]

            collection.add_documents(
                documents=normalized_documents,
                ids=ids,
            )

            self._persist_transcript_text(
                transcript_id=transcript_id,
                documents=normalized_documents,
            )

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="vector_store_add_completed",
                operation="add_documents",
                status="success",
                message="Documents added to vector store successfully",
                transcript_id=transcript_id,
                document_count=len(normalized_documents),
                duration_ms=duration_ms,
            )

            return ids

        except (ValidationError, VectorStoreError, EmbeddingError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="vector_store_add_failed",
                operation="add_documents",
                message="Failed to add documents to vector store",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise VectorStoreError(
                "Failed to add documents to vector store",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def similarity_search(
        self,
        *,
        transcript_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """Search transcript chunks by semantic similarity."""
        started_at = time.perf_counter()

        resolved_top_k = top_k or self.settings.retriever_top_k

        log_step(
            self.logger,
            event="similarity_search_requested",
            operation="similarity_search",
            status="started",
            message="Similarity search requested",
            transcript_id=transcript_id,
            top_k=resolved_top_k,
        )

        try:
            self._validate_transcript_id(transcript_id)
            self._validate_query(query)

            collection = self._get_collection(transcript_id=transcript_id)

            results = collection.similarity_search(
                query=query,
                k=resolved_top_k,
            )

            duration_ms = int((time.perf_counter() - started_at) * 1000)

            log_step(
                self.logger,
                event="similarity_search_completed",
                operation="similarity_search",
                status="success",
                message="Similarity search completed",
                transcript_id=transcript_id,
                result_count=len(results),
                duration_ms=duration_ms,
            )

            return results

        except (ValidationError, VectorStoreError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="similarity_search_failed",
                operation="similarity_search",
                message="Similarity search failed",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise VectorStoreError(
                "Similarity search failed",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def as_retriever(
        self,
        *,
        transcript_id: str,
        top_k: int | None = None,
    ) -> Any:
        """Return a LangChain retriever for a transcript collection."""
        resolved_top_k = top_k or self.settings.retriever_top_k

        log_step(
            self.logger,
            event="retriever_requested",
            operation="create_retriever",
            status="started",
            message="Creating retriever from vector store",
            transcript_id=transcript_id,
            top_k=resolved_top_k,
        )

        try:
            self._validate_transcript_id(transcript_id)

            collection = self._get_collection(transcript_id=transcript_id)

            retriever = collection.as_retriever(
                search_kwargs={"k": resolved_top_k}
            )

            log_step(
                self.logger,
                event="retriever_created",
                operation="create_retriever",
                status="success",
                message="Retriever created successfully",
                transcript_id=transcript_id,
                top_k=resolved_top_k,
            )

            return retriever

        except ValidationError:
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="retriever_creation_failed",
                operation="create_retriever",
                message="Failed to create retriever",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise VectorStoreError(
                "Failed to create retriever",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def get_transcript_text(self, transcript_id: str) -> str:
        """Load full transcript text reconstructed from indexed chunks."""
        log_step(
            self.logger,
            event="transcript_text_load_requested",
            operation="get_transcript_text",
            status="started",
            message="Loading transcript text from local metadata store",
            transcript_id=transcript_id,
        )

        try:
            self._validate_transcript_id(transcript_id)

            transcript_path = self._transcript_cache_path(transcript_id)

            if not transcript_path.exists():
                raise VectorStoreDocumentNotFoundError(
                    "Transcript text cache not found",
                    details={
                        "transcript_id": transcript_id,
                        "path": str(transcript_path),
                    },
                )

            text = transcript_path.read_text(encoding="utf-8").strip()

            if not text:
                raise VectorStoreDocumentNotFoundError(
                    "Transcript text cache is empty",
                    details={
                        "transcript_id": transcript_id,
                        "path": str(transcript_path),
                    },
                )

            log_step(
                self.logger,
                event="transcript_text_loaded",
                operation="get_transcript_text",
                status="success",
                message="Transcript text loaded successfully",
                transcript_id=transcript_id,
                text_length=len(text),
            )

            return text

        except (ValidationError, VectorStoreDocumentNotFoundError):
            raise

        except Exception as exc:
            log_failure(
                self.logger,
                event="transcript_text_load_failed",
                operation="get_transcript_text",
                message="Failed to load transcript text",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise VectorStoreError(
                "Failed to load transcript text",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_configuration(self) -> None:
        """Validate vector store configuration."""
        log_step(
            self.logger,
            event="vector_store_config_validation_started",
            operation="validate_vector_store_config",
            status="started",
            message="Validating vector store configuration",
            provider=self.settings.vector_store_provider,
        )

        if self.settings.vector_store_provider != "chroma":
            raise VectorStoreError(
                "Unsupported vector store provider",
                details={"provider": self.settings.vector_store_provider},
            )

        if not self.settings.openai_api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is required for OpenAI embeddings",
                details={"embedding_model": self.settings.openai_embedding_model},
            )

        log_step(
            self.logger,
            event="vector_store_config_validation_completed",
            operation="validate_vector_store_config",
            status="success",
            message="Vector store configuration validation completed",
            provider=self.settings.vector_store_provider,
        )

    def _create_embeddings(self) -> OpenAIEmbeddings:
        """Create OpenAI embeddings client."""
        log_step(
            self.logger,
            event="embedding_client_creation_started",
            operation="create_embeddings_client",
            status="started",
            message="Creating OpenAI embeddings client",
            model=self.settings.openai_embedding_model,
            provider="openai",
        )

        try:
            embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
            )

            log_step(
                self.logger,
                event="embedding_client_created",
                operation="create_embeddings_client",
                status="success",
                message="OpenAI embeddings client created successfully",
                model=self.settings.openai_embedding_model,
                provider="openai",
            )

            return embeddings

        except Exception as exc:
            log_failure(
                self.logger,
                event="embedding_client_creation_failed",
                operation="create_embeddings_client",
                message="Failed to create OpenAI embeddings client",
                exc=exc,
                model=self.settings.openai_embedding_model,
            )
            raise EmbeddingError(
                "Failed to create OpenAI embeddings client",
                details={
                    "embedding_model": self.settings.openai_embedding_model,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _get_collection(self, *, transcript_id: str) -> Chroma:
        """Return Chroma collection for one transcript."""
        log_step(
            self.logger,
            event="vector_collection_requested",
            operation="get_vector_collection",
            status="started",
            message="Opening vector store collection",
            transcript_id=transcript_id,
        )

        try:
            collection_name = self._collection_name(transcript_id)
            persist_directory = self._collection_path(transcript_id)

            persist_directory.mkdir(parents=True, exist_ok=True)

            collection = Chroma(
                collection_name=collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(persist_directory),
            )

            log_step(
                self.logger,
                event="vector_collection_ready",
                operation="get_vector_collection",
                status="success",
                message="Vector store collection ready",
                transcript_id=transcript_id,
                collection_name=collection_name,
                persist_directory=str(persist_directory),
            )

            return collection

        except Exception as exc:
            log_failure(
                self.logger,
                event="vector_collection_failed",
                operation="get_vector_collection",
                message="Failed to open vector store collection",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise VectorStoreError(
                "Failed to open vector store collection",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _normalize_documents(
        self,
        *,
        transcript_id: str,
        documents: list[Document],
    ) -> list[Document]:
        """Normalize document metadata before vector insertion."""
        log_step(
            self.logger,
            event="document_normalization_started",
            operation="normalize_documents",
            status="started",
            message="Normalizing documents before vector insertion",
            transcript_id=transcript_id,
            document_count=len(documents),
        )

        normalized_documents: list[Document] = []

        for index, document in enumerate(documents):
            metadata = {
                **document.metadata,
                "transcript_id": transcript_id,
                "chunk_index": document.metadata.get("chunk_index", index),
            }

            normalized_documents.append(
                Document(
                    page_content=document.page_content.strip(),
                    metadata=metadata,
                )
            )

        log_step(
            self.logger,
            event="document_normalization_completed",
            operation="normalize_documents",
            status="success",
            message="Document normalization completed",
            transcript_id=transcript_id,
            document_count=len(normalized_documents),
        )

        return normalized_documents

    def _persist_transcript_text(
        self,
        *,
        transcript_id: str,
        documents: list[Document],
    ) -> None:
        """Persist reconstructed transcript text for later summarization."""
        log_step(
            self.logger,
            event="transcript_cache_save_started",
            operation="persist_transcript_text",
            status="started",
            message="Persisting transcript text cache",
            transcript_id=transcript_id,
        )

        try:
            transcript_cache_path = self._transcript_cache_path(transcript_id)
            transcript_cache_path.parent.mkdir(parents=True, exist_ok=True)

            sorted_docs = sorted(
                documents,
                key=lambda doc: int(doc.metadata.get("chunk_index", 0)),
            )

            full_text = "\n\n".join(doc.page_content for doc in sorted_docs)

            transcript_cache_path.write_text(full_text, encoding="utf-8")

            log_step(
                self.logger,
                event="transcript_cache_save_completed",
                operation="persist_transcript_text",
                status="success",
                message="Transcript text cache persisted",
                transcript_id=transcript_id,
                path=str(transcript_cache_path),
                text_length=len(full_text),
            )

        except Exception as exc:
            log_failure(
                self.logger,
                event="transcript_cache_save_failed",
                operation="persist_transcript_text",
                message="Failed to persist transcript text cache",
                exc=exc,
                transcript_id=transcript_id,
            )
            raise VectorStoreError(
                "Failed to persist transcript text cache",
                details={
                    "transcript_id": transcript_id,
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    def _validate_transcript_id(self, transcript_id: str) -> None:
        """Validate transcript identifier."""
        if not transcript_id or not transcript_id.strip():
            raise ValidationError("transcript_id must not be empty")

    def _validate_query(self, query: str) -> None:
        """Validate semantic search query."""
        if not query or not query.strip():
            raise ValidationError("query must not be empty")

    def _validate_documents(self, documents: list[Document]) -> None:
        """Validate documents before vector insertion."""
        if not documents:
            raise ValidationError("documents must not be empty")

        for index, document in enumerate(documents):
            if not document.page_content or not document.page_content.strip():
                raise ValidationError(
                    "document page_content must not be empty",
                    details={"document_index": index},
                )

    def _collection_name(self, transcript_id: str) -> str:
        """Build stable Chroma collection name."""
        safe_id = transcript_id.replace("-", "_").replace(":", "_")
        return f"transcript_{safe_id}"

    def _collection_path(self, transcript_id: str) -> Path:
        """Build local persistence path for transcript vector collection."""
        return self.settings.vector_store_dir / transcript_id

    def _transcript_cache_path(self, transcript_id: str) -> Path:
        """Build local path for cached reconstructed transcript text."""
        return self.settings.transcripts_dir / f"{transcript_id}.indexed.txt"
