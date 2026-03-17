"""FastAPI application entry point exposing the NewsForge research pipeline API.

Endpoints:
    POST /research       — Run the full planner → search pipeline for a topic.
    GET  /health         — Health check with live / pending agent lists.
    GET  /pipeline/status — Detailed status of all 7 agents.
    GET  /               — Root welcome with docs link.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AgentStatus,
    HealthResponse,
    PipelineStatusResponse,
    ResearchRequest,
    ResearchResponse,
    SearchResultResponse,
    SubtaskResponse,
)
from orchestrator.graph import pipeline
from orchestrator.state import NewsForgeState


# ═══════════════════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook for the FastAPI application."""
    print("NewsForge API started — docs at http://localhost:8000/docs")
    yield


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NewsForge Multi-Agent Research API",
    description="LangGraph pipeline with Planner + Search agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """Run the full NewsForge research pipeline for a given topic.

    Accepts a ``ResearchRequest`` with a topic string and an optional
    research_id.  Builds an initial ``NewsForgeState``, invokes the
    LangGraph pipeline in a background thread (to avoid blocking the
    event loop), and returns structured results.
    """
    print(f"[API] POST /research — topic: {request.topic!r}")

    research_id: str = request.resolve_research_id()
    created_at: str = datetime.now(timezone.utc).isoformat()

    initial_state: NewsForgeState = {
        "research_id": research_id,
        "topic": request.topic,
        "subtasks": [],
        "search_results": [],
        "scraped_content": [],
        "analysis": None,
        "visuals": [],
        "draft_report": None,
        "critic_feedback": None,
        "revision_count": 0,
        "published_url": None,
        "published_record_id": None,
        "pipeline_status": "starting",
        "errors": [],
        "created_at": created_at,
        "completed_at": None,
    }

    config: dict[str, Any] = {"configurable": {"thread_id": research_id}}

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            pipeline.invoke,
            initial_state,
            config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    completed_at: str = datetime.now(timezone.utc).isoformat()

    # -- Determine overall status --
    errors: list[str] = result.get("errors", [])
    subtasks_raw: list[dict[str, Any]] = result.get("subtasks", [])
    results_raw: list[dict[str, Any]] = result.get("search_results", [])

    if errors and not subtasks_raw:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "complete"

    return ResearchResponse(
        research_id=research_id,
        topic=request.topic,
        status=status,
        subtasks=[SubtaskResponse(**s) for s in subtasks_raw],
        search_results=[SearchResultResponse(**r) for r in results_raw],
        subtask_count=len(subtasks_raw),
        result_count=len(results_raw),
        errors=errors,
        created_at=created_at,
        completed_at=completed_at,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and agent availability summary."""
    return HealthResponse()


@app.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status() -> PipelineStatusResponse:
    """Return detailed status of all 7 agents in the pipeline."""
    agents = [
        AgentStatus(
            name="planner",
            status="live",
            description="Decomposes research topics into subtasks using ReAct loop",
            phase="Phase 1",
        ),
        AgentStatus(
            name="search",
            status="live",
            description="Executes Tavily web searches for each subtask",
            phase="Phase 1",
        ),
        AgentStatus(
            name="scraper",
            status="coming_soon",
            description="Fetches and cleans full-page content from URLs",
            phase="Phase 2",
        ),
        AgentStatus(
            name="analysis",
            status="coming_soon",
            description="Extracts themes, facts, and contradictions",
            phase="Phase 2",
        ),
        AgentStatus(
            name="visual",
            status="coming_soon",
            description="Generates charts and diagrams from analysis",
            phase="Phase 2",
        ),
        AgentStatus(
            name="writer",
            status="coming_soon",
            description="Composes structured research report",
            phase="Phase 2",
        ),
        AgentStatus(
            name="critic",
            status="coming_soon",
            description="Reviews and scores report quality",
            phase="Phase 2",
        ),
    ]
    return PipelineStatusResponse(agents=agents)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with links to docs and health check."""
    return {
        "message": "NewsForge API",
        "docs": "/docs",
        "health": "/health",
    }
