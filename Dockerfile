FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./

RUN pip install --upgrade pip setuptools wheel

RUN pip install .

COPY . .

RUN mkdir -p \
    /app/storage/videos \
    /app/storage/transcripts \
    /app/storage/vector_store \
    /app/logs

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "ai_youtube_knowledge_summarizer.main:app", "--host", "0.0.0.0", "--port", "8000"]
