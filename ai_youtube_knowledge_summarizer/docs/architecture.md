# Architecture

# AI YouTube Knowledge Summarizer

Production-style AI SaaS backend for:

* YouTube media ingestion
* Whisper transcription
* Retrieval-Augmented Generation (RAG)
* OpenAI summarization
* Semantic search
* Question answering over transcript knowledge

---

# High-Level System Architecture

```text
                ┌────────────────────┐
                │     Streamlit UI   │
                │  User Interaction  │
                └─────────┬──────────┘
                          │ HTTP
                          ▼
                ┌────────────────────┐
                │      FastAPI       │
                │   REST Backend     │
                └─────────┬──────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
 ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
 │ YouTube Loader │ │  Whisper ASR │ │  OpenAI LLM     │
 │ yt-dlp         │ │ Transcription│ │ Summarization   │
 └────────────────┘ └──────────────┘ └─────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Transcript Chunking│
                │ Recursive Splitter │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  Embedding Layer   │
                │ OpenAI Embeddings  │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │   Chroma Vector DB │
                │ Semantic Retrieval │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Retrieval QA Layer │
                │ Contextual Answers │
                └────────────────────┘
```

---

# Design Goals

The project was designed as a production-style AI backend system rather than a simple tutorial implementation.

Primary goals:

* Reliability
* Observability
* Modularity
* Extensibility
* Typed interfaces
* Service isolation
* Failure visibility
* AI pipeline transparency
* SaaS-oriented backend architecture

---

# Technology Stack

| Layer            | Technology         |
| ---------------- | ------------------ |
| API              | FastAPI            |
| Frontend         | Streamlit          |
| LLM              | OpenAI GPT         |
| ASR              | OpenAI Whisper     |
| Embeddings       | OpenAI Embeddings  |
| Vector Store     | ChromaDB           |
| AI Framework     | LangChain          |
| Validation       | Pydantic v2        |
| Logging          | python-json-logger |
| Testing          | pytest             |
| Containerization | Docker             |
| Orchestration    | docker-compose     |

---

# Backend Architecture

The backend follows a layered service-oriented architecture.

```text
API Layer
    ↓
Schemas / Validation
    ↓
Service Layer
    ↓
AI Providers / Vector Storage
    ↓
Persistence / Local Storage
```

---

# Project Structure

```text
ai_youtube_knowledge_summarizer/
│
├── api/
│   ├── routes.py
│   └── schemas.py
│
├── core/
│   ├── config.py
│   ├── exceptions.py
│   └── logging.py
│
├── models/
│   └── domain.py
│
├── services/
│   ├── downloader.py
│   ├── transcriber.py
│   ├── chunker.py
│   ├── vector_store.py
│   ├── summarizer.py
│   └── qa_service.py
│
├── ui/
│   └── streamlit_app.py
│
├── tests/
│
├── docs/
│
├── scripts/
│
└── main.py
```

---

# Core Components

# 1. Downloader Service

File:

```text
services/downloader.py
```

Responsibilities:

* Validate YouTube URLs
* Extract video metadata
* Download media
* Normalize metadata
* Handle retry-safe downloading

Key Features:

* yt-dlp integration
* Validation layer
* Metadata safety
* Duration protection
* Structured operational logging

---

# 2. Transcriber Service

File:

```text
services/transcriber.py
```

Responsibilities:

* Load Whisper model
* Transcribe audio/video
* Detect language
* Persist transcripts
* Handle ASR failures

Key Features:

* Whisper model caching
* Device-aware inference
* Transcript persistence
* Runtime validation
* Failure isolation

---

# 3. Chunking Service

File:

```text
services/chunker.py
```

Responsibilities:

* Split transcript into semantic chunks
* Preserve metadata
* Generate LangChain documents

Key Features:

* RecursiveCharacterTextSplitter
* Configurable chunk size
* Configurable overlap
* Metadata propagation
* Domain chunk modeling

---

# 4. Vector Store Service

File:

```text
services/vector_store.py
```

Responsibilities:

* Generate embeddings
* Persist vectors
* Semantic retrieval
* Retriever abstraction
* Transcript indexing

Key Features:

* Chroma integration
* OpenAI embeddings
* Local persistence
* Similarity search
* Retriever generation

---

# 5. Summarization Service

File:

```text
services/summarizer.py
```

Responsibilities:

* Generate transcript summaries
* Support multiple summarization strategies

Supported Strategies:

* stuff
* map_reduce
* refine

Key Features:

* Prompt engineering
* LangChain summarize chains
* LLM abstraction
* Typed summarization modes
* Hallucination reduction

---

# 6. QA Service

File:

```text
services/qa_service.py
```

Responsibilities:

* Retrieval-Augmented Generation
* Semantic question answering
* Context retrieval
* Source tracking

Key Features:

* RetrievalQA chain
* Retriever abstraction
* Source attribution
* Context-only answering
* Hallucination protection

---

# Configuration System

File:

```text
core/config.py
```

Architecture:

* Pydantic Settings
* Typed configuration
* Environment-driven configuration
* Startup validation

Capabilities:

* Runtime validation
* Directory initialization
* Secret protection
* Safe configuration summaries

---

# Logging Architecture

File:

```text
core/logging.py
```

Logging design follows production observability principles.

Features:

* Structured logging
* Request correlation
* Operational events
* Duration tracking
* Failure tracing
* Service-level visibility

Example Event Structure:

```json
{
  "event": "transcription_completed",
  "operation": "transcribe_media",
  "status": "success",
  "duration_ms": 15230
}
```

---

# Exception Architecture

File:

```text
core/exceptions.py
```

The project uses a centralized typed exception hierarchy.

Goals:

* Stable API errors
* Predictable error handling
* Consistent responses
* Better operational visibility

Examples:

* ValidationError
* DownloadError
* TranscriptionError
* RetrievalError
* VectorStoreError
* SummarizationError

---

# API Layer

File:

```text
api/routes.py
```

Responsibilities:

* HTTP request handling
* Input validation
* Response serialization
* Request lifecycle logging

Endpoints:

| Endpoint          | Purpose          |
| ----------------- | ---------------- |
| `/health`         | Service health   |
| `/videos/process` | Full AI pipeline |
| `/summaries`      | Generate summary |
| `/qa`             | Retrieval QA     |

---

# Streamlit UI

File:

```text
ui/streamlit_app.py
```

Responsibilities:

* User interaction
* Backend orchestration
* Visualization
* Retrieval QA interaction

Capabilities:

* Video processing
* Summary generation
* QA interaction
* Source exploration
* Session tracking

---

# AI Pipeline Flow

```text
YouTube URL
    ↓
Download Media
    ↓
Whisper Transcription
    ↓
Transcript Chunking
    ↓
Embedding Generation
    ↓
Vector Store Indexing
    ↓
Retrieval QA / Summarization
```

---

# Reliability Features

The project includes production-oriented reliability mechanisms.

Implemented:

* Startup validation
* Typed configuration
* Structured exception hierarchy
* Request correlation IDs
* Timeout handling
* Retry support
* Runtime validation
* Safe failure propagation
* Persistent transcript storage

---

# Testing Strategy

Framework:

```text
pytest
```

Coverage Areas:

* API endpoints
* Configuration validation
* Downloader service
* Whisper transcription
* Chunking logic
* Vector store operations
* Summarization service
* Retrieval QA service

Testing Goals:

* Isolation
* Service mocking
* Validation coverage
* Failure-path coverage
* Reliability validation

---

# Deployment Architecture

Dockerized services:

```text
FastAPI API
Streamlit UI
Chroma persistence
Local storage volumes
```

Deployment Targets:

* Local development
* Docker Compose
* Kubernetes (future)
* Cloud-native deployment

---

# Future Scalability

Planned scalability improvements:

## Infrastructure

* Redis
* Celery
* PostgreSQL
* Object storage
* Horizontal scaling

## Observability

* Prometheus
* Grafana
* OpenTelemetry
* Sentry

## AI

* Multi-provider LLM support
* Local LLM inference
* Reranking pipelines
* Semantic caching
* Agentic orchestration

---

# Engineering Principles

This project emphasizes:

* Production-style backend engineering
* Reliability-first architecture
* AI infrastructure awareness
* Strong operational visibility
* Clear service boundaries
* Typed interfaces
* Extensible AI pipelines
* Modern Python backend practices

---

# Summary

AI YouTube Knowledge Summarizer is a production-oriented AI SaaS backend platform demonstrating:

* Generative AI integration
* Retrieval-Augmented Generation
* AI infrastructure engineering
* Vector search systems
* LLM orchestration
* Backend reliability practices
* Observability architecture
* Modern Python service design
