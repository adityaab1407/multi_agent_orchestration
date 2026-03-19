# NewsForge — Multi-Agent Research System

> Automated news research, analysis, and report generation powered by a 7-agent LangGraph pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## What Is This?

NewsForge is a **multi-agent orchestration system** that takes a research topic, breaks it into subtasks, searches the web, and (in Phase 2) scrapes, analyses, visualises, writes, and quality-checks a final research report — all coordinated by a LangGraph StateGraph.

**Phase 1 (Live):** Planner Agent + Search Agent fully operational with Langfuse observability.

**Phase 2 (Planned):** Scraper, Analysis, Visual, Writer, Critic, and Publisher agents; MCP server integration; RAG with Qdrant.

---

## Architecture

```
User Topic
    │
    ▼
┌──────────────────┐
│  Planner Agent   │  ← ReAct loop (Groq LLM)          ✅ LIVE
│  Subtask          │
└────────┬─────────┘
         │ subtasks
         ▼
┌──────────────────┐
│  Search Agent    │  ← Tavily web search               ✅ LIVE
│  5 results/query  │
└────────┬─────────┘
         │ search_results
         ▼
┌──────────────────┐
│  Scraper Agent   │  ← BeautifulSoup + Playwright      🔜 Phase 2
└────────┬─────────┘
         │ scraped_content
         ▼
┌──────────────────┐
│  Analysis Agent  │  ← Groq ReAct                      🔜 Phase 2
└────────┬─────────┘
         │ analysis
         ▼
┌──────────────────┐
│  Visual Agent    │  ← matplotlib / plotly              🔜 Phase 2
└────────┬─────────┘
         │ visuals
         ▼
┌──────────────────┐
│  Writer Agent    │  ← Structured Generation            🔜 Phase 2
└────────┬─────────┘
         │ draft_report
         ▼
┌──────────────────┐
│  Critic Agent    │  ← Quality Check + Revision Loop   🔜 Phase 2
└──────────────────┘
         │
         ▼
    Final Report
```

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph, 7 nodes) |
| LLM (Reasoning) | Groq — `meta-llama/llama-4-scout-17b-16e-instruct` |
| LLM (Execution) | Groq — `llama-3.1-8b-instant` |
| LLM (Judge) | Groq — `llama-3.1-8b-instant` |
| Web Search | Tavily API |
| Observability | Langfuse v3 |
| Backend | FastAPI (port 8080) |
| Frontend | Streamlit (port 8501) |
| State Persistence | SQLite via `SqliteSaver` |
| Schemas | Pydantic V2 |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/adityaab1407/newsforge-multi-agent.git
cd newsforge-multi-agent
python -m venv .multi_agent
# Windows:
.multi_agent\Scripts\activate
# Linux/macOS:
source .multi_agent/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

> **API Keys Required:** Create a dedicated Groq API key for NewsForge — do not reuse keys from other projects. This keeps token usage and rate limits isolated per project.

```env
GROQ_API_KEY=gsk_your_newsforge_dedicated_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
GROQ_REASONING_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_JUDGE_MODEL=llama-3.1-8b-instant
TAVILY_API_KEY=tvly-dev-your_key_here
LANGFUSE_PUBLIC_KEY=pk-lf-your_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_key_here
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3. Run

**Backend:**
```bash
python -m uvicorn backend.main:app --port 8080
```

**Frontend (new terminal):**
```bash
python -m streamlit run frontend/app.py --server.port 8501
```

**Or with Docker:**
```bash
docker compose up --build
```

### 4. Use

Open `http://localhost:8501` in your browser, enter a research topic, and click **Run Research**.

---

## Project Structure

```
multi_agent_orchestration/
├── agents/
│   ├── planner.py         # ✅ ReAct planner (Groq)
│   ├── search.py          # ✅ Tavily search with retry
│   ├── scraper.py         # 🔜 Web scraping skeleton
│   ├── analysis.py        # 🔜 Theme extraction skeleton
│   ├── visual.py          # 🔜 Chart generation skeleton
│   ├── writer.py          # 🔜 Report composition skeleton
│   ├── critic.py          # 🔜 Quality review skeleton
│   └── publisher.py       # 🔜 Report delivery skeleton
├── orchestrator/
│   ├── state.py           # ✅ Shared TypedDict state schema
│   ├── graph.py           # ✅ LangGraph StateGraph (7 nodes)
│   └── checkpointer.py   # ✅ SQLite checkpointer
├── backend/
│   ├── main.py            # ✅ FastAPI app (4 endpoints)
│   └── schemas.py         # ✅ Pydantic V2 request/response
├── frontend/
│   └── app.py             # ✅ Streamlit UI
├── config/
│   └── settings.py        # ✅ Centralised .env config
├── mcp_servers/
│   ├── search_server.py   # 🔜 MCP search tool server
│   └── storage_server.py  # 🔜 MCP storage tool server
├── tests/                 # 🔜 Test suite
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── Makefile
├── requirements.txt
├── .gitignore
├── LEARNINGS.md
└── README.md
```

---

## Design Decisions

1. **LangGraph over LangChain Chains** — Explicit node-based graph gives fine control over agent orchestration, state sharing, and conditional routing (needed for the Critic → Writer revision loop).

2. **Pydantic V2 everywhere** — Strict schema validation at every boundary (LLM output parsing, API request/response, state contracts) prevents silent data corruption.

3. **ReAct Pattern for Planner** — Multi-iteration planning with coverage scoring produces far better subtask decomposition than single-shot prompting.

4. **Langfuse v3 context manager API** — Trace → Span nesting with `start_as_current_observation()` for clean observability without cluttering agent code.

5. **SQLite Checkpointer** — Zero-dependency state persistence; swap for PostgreSQL in production.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/research` | Run full research pipeline |
| `GET` | `/health` | Health check (agents live/pending) |
| `GET` | `/pipeline/status` | Pipeline status overview |
| `GET` | `/` | Welcome message |

---

## Author

**Aditya** — [GitHub](https://github.com/adityaab1407)

---

## License

MIT
