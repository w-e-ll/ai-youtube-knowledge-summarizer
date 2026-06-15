.PHONY: help install dev test lint format run-api run-ui run-pipeline docker-build docker-up docker-down clean

PYTHON := python3
PIP := pip
PYTEST := pytest

APP_MODULE := ai_youtube_knowledge_summarizer.main:app

help:
	@echo ""
	@echo "AI YouTube Knowledge Summarizer"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make install        Install project dependencies"
	@echo "  make dev            Install development dependencies"
	@echo "  make test           Run test suite"
	@echo "  make lint           Run lint checks"
	@echo "  make format         Format source code"
	@echo "  make run-api        Start FastAPI backend"
	@echo "  make run-ui         Start Streamlit UI"
	@echo "  make run-pipeline   Run local pipeline example"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start Docker services"
	@echo "  make docker-down    Stop Docker services"
	@echo "  make clean          Remove cache and temporary files"
	@echo ""

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

dev:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -vv

lint:
	ruff check .
	mypy ai_youtube_knowledge_summarizer

format:
	black .
	ruff check . --fix

run-api:
	uvicorn $(APP_MODULE) --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run ai_youtube_knowledge_summarizer/ui/streamlit_app.py

run-pipeline:
	$(PYTHON) scripts/run_local_pipeline.py \
		--url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
		--summary-mode map_reduce \
		--question "What is the main topic of the video?"

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
