.DEFAULT_GOAL := help
SHELL         := /bin/bash

PORT_UI  ?= 8501

.PHONY: help dev up down build logs test lint format clean

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Table Tennis Prediction"
	@echo "  ======================="
	@echo ""
	@echo "  Development"
	@echo "  make dev        Start locally without Docker (UI + MLflow)"
	@echo "  make install    Install dependencies via uv"
	@echo ""
	@echo "  Production (Docker)"
	@echo "  make build      Build Docker image"
	@echo "  make up         Start  →  http://localhost:$(PORT_UI)"
	@echo "  make down       Stop containers"
	@echo "  make logs       Tail container logs"
	@echo ""
	@echo "  CI / Quality"
	@echo "  make test       Run unit tests"
	@echo "  make lint       Lint with ruff"
	@echo "  make format     Auto-fix + format"
	@echo ""
	@echo "  make clean      Remove cache files"
	@echo ""
	@echo "  All training & predictions are managed through the UI."
	@echo ""

# ─── Development ──────────────────────────────────────────────────────────────
install:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --extra dev

dev: install
	@echo "Starting development server..."
	@echo "  Streamlit → http://localhost:$(PORT_UI)"
	@echo "  Press Ctrl+C to stop."
	uv run streamlit run app/streamlit_app.py \
		--server.port $(PORT_UI) \
		--server.address 0.0.0.0 \
		--server.headless true

# ─── Docker ───────────────────────────────────────────────────────────────────
build:
	docker build -t tabletennis-prediction:latest -f docker/Dockerfile .

up:
	docker compose -f docker/docker-compose.yml up --build -d
	@echo ""
	@echo "  Streamlit → http://localhost:$(PORT_UI)"
	@echo "  Stop with: make down"

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f

# ─── CI / Quality ─────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v --tb=short

test-fast:
	uv run pytest tests/ -v --tb=short --ignore=tests/test_model_helpers.py -x

lint:
	uv run ruff check src/ app/ tests/

format:
	uv run ruff check --fix src/ app/ tests/
	uv run ruff format src/ app/ tests/

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
