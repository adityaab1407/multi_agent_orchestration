"""FastAPI application entry point exposing the NewsForge research pipeline API.

Endpoints:
    POST /research                          — Start a research pipeline (pauses for HITL review).
    POST /research/{research_id}/approve    — Resume a paused pipeline with approval.
    POST /research/{research_id}/reject     — Resume a paused pipeline with rejection.
    GET  /research/{research_id}/status     — Poll current status of a pipeline run.
    GET  /health                            — Health check with live / pending agent lists.
    GET  /pipeline/status                   — Detailed status of all 7 agents.
    GET  /                                  — Root welcome with docs link.

Human-in-the-Loop (HITL) flow:
    1. POST /research starts the pipeline.  It runs through all agents until
       the human_review_node, which calls LangGraph's ``interrupt()`` to PAUSE.
    2. The API returns status="awaiting_approval" with a report preview.
    3. The frontend calls POST /approve or /reject to resume the pipeline.
    4. On approve: publisher runs → report saved → status="complete".
       On reject:  pipeline ends immediately → status="rejected".

    The pipeline state is persisted in SQLite via LangGraph's checkpointer,
    so it can be resumed from exactly the interrupt point even after a server
    restart.
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
from langgraph.types import Command

from backend.schemas import (
    AgentStatus,
    CriticFeedbackResponse,
    HealthResponse,
    PipelineStatusResponse,
    ResearchRequest,
    ResearchResponse,
    ReviewStatusResponse,
    SearchResultResponse,
    SubtaskResponse,
)
from orchestrator.graph import pipeline
from orchestrator.state import NewsForgeState

active_runs: dict[str, dict[str, Any]] = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook for the FastAPI application."""
    print("NewsForge API started — docs at http://localhost:8000/docs")
    yield


app = FastAPI(
    title="NewsForge Multi-Agent Research API",
    description=(
        "LangGraph pipeline with 7 agents and Human-in-the-Loop review. "
        "POST /research starts a run that pauses for approval before publishing."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """Start the full NewsForge research pipeline for a given topic.

    The pipeline runs through all 7 agents until it reaches the
    ``human_review_node``, where LangGraph's ``interrupt()`` pauses
    execution.  The response includes ``status="awaiting_approval"``
    with a preview of the report so the human can decide.

    To continue, call ``POST /research/{research_id}/approve`` or
    ``POST /research/{research_id}/reject``.
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
        "draft_report": None,
        "critic_feedback": None,
        "revision_count": 0,
        "human_decision": None,
        "published_url": None,
        "published_record_id": None,
        "pipeline_status": "starting",
        "errors": [],
        "created_at": created_at,
        "completed_at": None,
    }

    config: dict[str, Any] = {"configurable": {"thread_id": research_id}}

    # Track this run
    active_runs[research_id] = {
        "thread_id": research_id,
        "topic": request.topic,
        "status": "running",
        "report_preview": None,
        "quality_score": None,
        "word_count": None,
        "revision_count": None,
        "created_at": created_at,
    }

    try:
        # Pipeline will run until human_review_node calls interrupt(),
        # then return control here with the graph in a paused state.
        result: dict[str, Any] = await asyncio.to_thread(
            pipeline.invoke,
            initial_state,
            config,
        )
    except Exception as e:
        active_runs[research_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))

    # ── Check if the pipeline paused at an interrupt ───────────────────
    # When interrupt() fires, pipeline.invoke() returns the state at the
    # point of interruption.  We detect this by checking pipeline status
    # or the presence of interrupt metadata via get_state().
    graph_state = pipeline.get_state(config)
    is_interrupted = bool(graph_state.next)  # non-empty = paused at a node

    if is_interrupted:
        # Extract the interrupt payload from the graph state
        # The interrupt value is stored in state tasks
        interrupt_data = {}
        if graph_state.tasks:
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_data = task.interrupts[0].value
                    break

        active_runs[research_id].update({
            "status": "awaiting_approval",
            "report_preview": interrupt_data.get("report_preview"),
            "quality_score": interrupt_data.get("quality_score"),
            "word_count": interrupt_data.get("word_count"),
            "revision_count": interrupt_data.get("revision_count"),
        })

        # Return a response indicating the pipeline is paused
        subtasks_raw = result.get("subtasks", [])
        results_raw = result.get("search_results", [])

        return _build_response_from_result(
            result, research_id, request.topic, created_at,
            status_override="awaiting_approval", completed_at_override="",
        )

    # If pipeline completed without interrupt (shouldn't happen in normal
    # flow, but handle gracefully)
    return _build_response_from_result(result, research_id, request.topic, created_at)


@app.post("/research/{research_id}/approve", response_model=ResearchResponse)
async def approve_research(research_id: str) -> ResearchResponse:
    """Resume a paused pipeline with human approval.

    The pipeline continues from the ``human_review_node`` interrupt,
    runs the Publisher agent, and returns the final result.
    """
    print(f"[API] POST /research/{research_id}/approve")

    run = _get_active_run(research_id)
    if run["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Run {research_id} is not awaiting approval (status: {run['status']})",
        )

    config: dict[str, Any] = {"configurable": {"thread_id": research_id}}

    try:
        # Resume the paused pipeline with the approval decision.
        # Command(resume=...) continues execution from the interrupt() call
        # in human_review_node.  The resume value becomes the return value
        # of interrupt() inside the node function.
        result: dict[str, Any] = await asyncio.to_thread(
            pipeline.invoke,
            Command(resume={"decision": "approve"}),
            config,
        )
    except Exception as e:
        active_runs[research_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))

    active_runs[research_id]["status"] = "complete"

    return _build_response_from_result(
        result, research_id, run["topic"], run["created_at"]
    )


@app.post("/research/{research_id}/reject", response_model=ResearchResponse)
async def reject_research(research_id: str) -> ResearchResponse:
    """Resume a paused pipeline with human rejection.

    The pipeline ends without publishing.  No report is saved.
    """
    print(f"[API] POST /research/{research_id}/reject")

    run = _get_active_run(research_id)
    if run["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Run {research_id} is not awaiting approval (status: {run['status']})",
        )

    config: dict[str, Any] = {"configurable": {"thread_id": research_id}}

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            pipeline.invoke,
            Command(resume={"decision": "reject"}),
            config,
        )
    except Exception as e:
        active_runs[research_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))

    active_runs[research_id]["status"] = "rejected"
    completed_at = datetime.now(timezone.utc).isoformat()

    subtasks_raw = result.get("subtasks", [])
    results_raw = result.get("search_results", [])

    return _build_response_from_result(
        result, research_id, run["topic"], run["created_at"],
        status_override="rejected",
    )


@app.get("/research/{research_id}/status", response_model=ReviewStatusResponse)
async def get_research_status(research_id: str) -> ReviewStatusResponse:
    """Poll the current status of a pipeline run.

    Useful for frontends that need to know when the pipeline has
    reached the human review point.
    """
    run = _get_active_run(research_id)

    return ReviewStatusResponse(
        research_id=research_id,
        status=run["status"],
        topic=run["topic"],
        report_preview=run.get("report_preview"),
        quality_score=run.get("quality_score"),
        word_count=run.get("word_count"),
        revision_count=run.get("revision_count"),
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
            description="Executes Tavily web searches for each subtask (parallel)",
            phase="Phase 1",
        ),
        AgentStatus(
            name="scraper",
            status="live",
            description="Fetches and cleans full-page content from URLs",
            phase="Phase 2",
        ),
        AgentStatus(
            name="analysis",
            status="live",
            description="Extracts themes, facts, and contradictions via ReAct loop",
            phase="Phase 2",
        ),
        AgentStatus(
            name="writer",
            status="live",
            description="Composes structured research report",
            phase="Phase 2",
        ),
        AgentStatus(
            name="critic",
            status="live",
            description="Reviews and scores report quality with revision loop",
            phase="Phase 2",
        ),
        AgentStatus(
            name="publisher",
            status="live",
            description="Saves reports locally and optionally to AWS S3 + DynamoDB",
            phase="Phase 2",
        ),
    ]
    return PipelineStatusResponse(agents=agents)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with links to docs and health check."""
    return {
        "message": "NewsForge API — Human-in-the-Loop Research Pipeline",
        "docs": "/docs",
        "health": "/health",
    }



def _get_active_run(research_id: str) -> dict[str, Any]:
    """Look up an active run or raise 404."""
    if research_id not in active_runs:
        raise HTTPException(
            status_code=404,
            detail=f"Research run {research_id} not found",
        )
    return active_runs[research_id]


def _build_response_from_result(
    result: dict[str, Any],
    research_id: str,
    topic: str,
    created_at: str,
    status_override: str | None = None,
    completed_at_override: str | None = None,
) -> ResearchResponse:
    """Build a ResearchResponse from a pipeline result dict.

    Populates all rich fields (analysis, draft_report, critic_feedback, etc.)
    so the frontend has full visibility into the pipeline output.
    """
    completed_at = completed_at_override if completed_at_override is not None else datetime.now(timezone.utc).isoformat()

    errors = result.get("errors", [])
    subtasks_raw = result.get("subtasks", [])
    results_raw = result.get("search_results", [])

    if status_override:
        status = status_override
    elif errors and not subtasks_raw:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "complete"

    # Build critic feedback response if present
    critic_raw = result.get("critic_feedback")
    critic_resp = None
    if critic_raw and isinstance(critic_raw, dict):
        critic_resp = CriticFeedbackResponse(
            passed=critic_raw.get("passed", False),
            quality_score=critic_raw.get("quality_score", 0.0),
            feedback_notes=critic_raw.get("feedback_notes", []),
        )

    return ResearchResponse(
        research_id=research_id,
        topic=topic,
        status=status,
        subtasks=[SubtaskResponse(**s) for s in subtasks_raw],
        search_results=[SearchResultResponse(**r) for r in results_raw],
        subtask_count=len(subtasks_raw),
        result_count=len(results_raw),
        scraped_content=result.get("scraped_content"),
        analysis=result.get("analysis"),
        draft_report=result.get("draft_report"),
        critic_feedback=critic_resp,
        revision_count=result.get("revision_count", 0),
        published_url=result.get("published_url"),
        published_record_id=result.get("published_record_id"),
        errors=errors,
        created_at=created_at,
        completed_at=completed_at,
    )
