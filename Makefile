# ── Makefile ────────────────────────────────────────────────────────────
# NewsForge Multi-Agent Research System
# Works on Windows (PowerShell) and Linux/macOS
# ───────────────────────────────────────────────────────────────────────
.PHONY: help install run-backend run-frontend run test lint clean docker-up docker-down

PYTHON ?= python
PIP    ?= pip
PORT   ?= 8080

help:  ## Show this help
	@echo.
	@echo  NewsForge Make Targets
	@echo  =====================
	@echo  install        Install all Python dependencies
	@echo  run-backend    Start FastAPI backend on port $(PORT)
	@echo  run-frontend   Start Streamlit frontend on port 8501
	@echo  run            Start both backend and frontend
	@echo  test           Run pytest
	@echo  lint           Run ruff linter
	@echo  clean          Remove caches and temp files
	@echo  docker-up      Build and start Docker containers
	@echo  docker-down    Stop Docker containers
	@echo.

install:  ## Install dependencies
	$(PIP) install -r requirements.txt

run-backend:  ## Start FastAPI backend
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload

run-frontend:  ## Start Streamlit frontend
	$(PYTHON) -m streamlit run frontend/app.py --server.port 8501

run: ## Start both (backend in background)
	@echo Starting backend on port $(PORT)...
	start /B $(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port $(PORT)
	@echo Starting frontend on port 8501...
	$(PYTHON) -m streamlit run frontend/app.py --server.port 8501

test:  ## Run tests
	$(PYTHON) -m pytest tests/ -v

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

clean:  ## Remove caches and temp files
	@if exist __pycache__ rd /s /q __pycache__
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist data\checkpoints.db del data\checkpoints.db
	@echo Cleaned.

docker-up:  ## Build and start Docker containers
	docker compose up --build -d

docker-down:  ## Stop Docker containers
	docker compose down
