# ── Makefile ────────────────────────────────────────────────────────────
# NewsForge Multi-Agent Research System
# ───────────────────────────────────────────────────────────────────────
.PHONY: help setup run-backend run-frontend test benchmark \
        benchmark-summary clean-data docker-up docker-down lint

PYTHON ?= python
PIP    ?= pip
PORT   ?= 8080

help:  ## Show this help
	@echo ""
	@echo "  NewsForge Make Targets"
	@echo "  ====================="
	@echo "  setup              Create venv, install deps, copy .env"
	@echo "  run-backend        Start FastAPI backend on port $(PORT)"
	@echo "  run-frontend       Start Streamlit frontend on port 8501"
	@echo "  test               Run pytest"
	@echo "  benchmark          Run 10-topic benchmark"
	@echo "  benchmark-summary  Show last benchmark results"
	@echo "  clean-data         Remove stale reports and checkpoints"
	@echo "  docker-up          Build and start Docker containers"
	@echo "  docker-down        Stop Docker containers"
	@echo "  lint               Syntax-check all Python files"
	@echo ""

setup:  ## Create venv, install deps, copy .env template
	$(PYTHON) -m venv .multi_agent
	. .multi_agent/bin/activate && $(PIP) install -r requirements.txt
	cp -n .env.example .env 2>/dev/null || true
	@echo "Edit .env with your API keys"

run-backend:  ## Start FastAPI backend
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port $(PORT) --reload

run-frontend:  ## Start Streamlit frontend
	$(PYTHON) -m streamlit run frontend/app.py --server.port 8501

test:  ## Run tests
	$(PYTHON) -m pytest tests/ -v

benchmark:  ## Run full 10-topic benchmark
	$(PYTHON) evaluation/benchmark_runner.py --topics 10

benchmark-summary:  ## Show last benchmark results
	$(PYTHON) evaluation/benchmark_runner.py --summary-only

clean-data:  ## Remove stale reports, old benchmarks, checkpoint DB
	$(PYTHON) scripts/cleanup_data.py

docker-up:  ## Build and start Docker containers
	docker compose up --build -d

docker-down:  ## Stop Docker containers
	docker compose down

lint:  ## Syntax-check all Python source files
	$(PYTHON) -m py_compile agents/planner.py
	$(PYTHON) -m py_compile agents/search.py
	$(PYTHON) -m py_compile agents/scraper.py
	$(PYTHON) -m py_compile agents/analysis.py
	$(PYTHON) -m py_compile agents/writer.py
	$(PYTHON) -m py_compile agents/critic.py
	$(PYTHON) -m py_compile agents/publisher.py
	$(PYTHON) -m py_compile orchestrator/state.py
	$(PYTHON) -m py_compile orchestrator/graph.py
	$(PYTHON) -m py_compile backend/main.py
	$(PYTHON) -m py_compile backend/schemas.py
	$(PYTHON) -m py_compile evaluation/judge.py
	$(PYTHON) -m py_compile frontend/app.py
	@echo "All files OK"
