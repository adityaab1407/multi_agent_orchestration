"""
orchestrator/graph.py

Full LangGraph StateGraph wiring all 7 NewsForge agents into a linear pipeline.

Pipeline order:
    planner → search → scraper → analysis → visual → writer → critic

planner_node calls PlannerAgent (Groq LLM) with Langfuse tracing.
search_node is a placeholder that logs received subtasks.
The remaining five nodes are pass-through stubs for Phase 2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid
from datetime import datetime, timezone
from typing import Any

from langfuse import Langfuse
from langgraph.graph import END, StateGraph

from agents.planner import PlannerAgent
from config.settings import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from orchestrator.checkpointer import get_checkpointer
from orchestrator.state import NewsForgeState


# ═══════════════════════════════════════════════════════════════════════════
# Langfuse client (module-level singleton)
# ═══════════════════════════════════════════════════════════════════════════

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)


# ═══════════════════════════════════════════════════════════════════════════
# Node functions — executed in order by the LangGraph runner
# ═══════════════════════════════════════════════════════════════════════════


def planner_node(state: NewsForgeState) -> dict[str, Any]:
    """Decompose the user's research topic into prioritised sub-tasks.

    This is the **real implementation** — it calls ``PlannerAgent().run()``
    which invokes Groq (llama-3.3-70b-versatile) in a ReAct loop.

    Reads from state:
        - research_id: unique pipeline run identifier (used as Langfuse trace id).
        - topic: the user-supplied research topic string.

    Writes to state:
        - subtasks: a list of Subtask dicts produced by the LLM.
        - pipeline_status: set to ``"planner_complete"``.
        - errors: appended to on failure.

    Langfuse instrumentation:
        - Creates a trace (via ``start_as_current_observation``) for the node.
        - Wraps the full PlannerAgent.run() call in a nested span.
        - Logs a ``planner_complete`` event with subtask count.
    """
    research_id: str = state["research_id"]
    topic: str = state["topic"]

    try:
        # -- Outer trace for this pipeline node --
        with langfuse.start_as_current_observation(
            name="planner_node",
            metadata={"topic": topic, "research_id": research_id},
        ) as trace:

            # -- Nested span covering the full ReAct loop --
            with trace.start_as_current_observation(
                name="react_loop",
                input={"topic": topic},
            ) as span:
                print(f"[planner_node] Calling PlannerAgent for topic: {topic!r}")
                agent = PlannerAgent()
                subtasks: list[dict[str, Any]] = agent.run(
                    topic=topic, research_id=research_id
                )

                span.update(output={"subtasks_generated": len(subtasks)})

            # -- Log completion event --
            trace.create_event(
                name="planner_complete",
                metadata={"subtask_count": len(subtasks)},
            )

        langfuse.flush()

        print(
            f"[planner_node] Done — {len(subtasks)} subtasks generated. "
            f"Trace id: {trace.trace_id}"
        )

        return {
            "subtasks": subtasks,
            "pipeline_status": "planner_complete",
        }

    except Exception as e:
        print(f"[planner_node] ERROR: {e}")
        return {"errors": [f"planner_node failed: {str(e)}"]}


def search_node(state: NewsForgeState) -> dict[str, Any]:
    """Execute web searches for each sub-task produced by the Planner.

    Reads from state:
        - subtasks: list of Subtask dicts to generate search queries from.

    Writes to state:
        - pipeline_status: set to ``"search_node_placeholder"``.
        - search_results: (Day 3) accumulated SearchResult dicts from Tavily.

    Current behaviour (placeholder):
        Logs the subtask count received from planner_node.
    """
    subtasks = state.get("subtasks", [])
    print(
        f"[search_node] Received {len(subtasks)} subtasks from Planner. "
        f"Real implementation in Day 3."
    )
    return {"pipeline_status": "search_node_placeholder"}


def scraper_node(state: NewsForgeState) -> dict[str, Any]:
    """Fetch and clean full-page content for each search result URL.

    Reads from state:
        - search_results: list of SearchResult dicts containing URLs to scrape.

    Writes to state:
        - pipeline_status: set to ``"scraper_node running"``.
        - scraped_content: (Phase 2) list of ScrapedContent dicts.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[scraper_node] STUB — pass-through, no scraping yet")
    return {"pipeline_status": "scraper_node running"}


def analysis_node(state: NewsForgeState) -> dict[str, Any]:
    """Run sentiment analysis, topic clustering, and fact extraction.

    Reads from state:
        - scraped_content: cleaned text chunks from the Scraper.
        - search_results: original search metadata for cross-referencing.

    Writes to state:
        - pipeline_status: set to ``"analysis_node running"``.
        - analysis: (Phase 2) an AnalysisOutput dict.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[analysis_node] STUB — pass-through, no analysis yet")
    return {"pipeline_status": "analysis_node running"}


def visual_node(state: NewsForgeState) -> dict[str, Any]:
    """Generate charts, diagrams, and visual summaries from the analysis.

    Reads from state:
        - analysis: AnalysisOutput dict containing themes and key facts.

    Writes to state:
        - pipeline_status: set to ``"visual_node running"``.
        - visuals: (Phase 2) list of VisualOutput dicts.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[visual_node] STUB — pass-through, no visuals yet")
    return {"pipeline_status": "visual_node running"}


def writer_node(state: NewsForgeState) -> dict[str, Any]:
    """Compose a polished research report from the analysis and visuals.

    Reads from state:
        - topic, analysis, visuals, critic_feedback.

    Writes to state:
        - pipeline_status: set to ``"writer_node running"``.
        - draft_report: (Phase 2) Markdown string of the full report.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[writer_node] STUB — pass-through, no writing yet")
    return {"pipeline_status": "writer_node running"}


def critic_node(state: NewsForgeState) -> dict[str, Any]:
    """Review and score the draft report for quality, accuracy, and coverage.

    Reads from state:
        - draft_report, analysis.

    Writes to state:
        - pipeline_status: set to ``"critic_node running"``.
        - critic_feedback: (Phase 2) CriticFeedback dict.
        - revision_count: (Phase 2) incremented on revision.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[critic_node] STUB — pass-through, no critique yet")
    return {"pipeline_status": "critic_node running"}


# ═══════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════

def build_pipeline():
    """Construct, compile, and return the full NewsForge LangGraph pipeline.

    The graph is a linear chain:

        planner → search → scraper → analysis → visual → writer → critic

    A SqliteSaver checkpointer is attached so that every node's output is
    persisted to ``data/newsforge_checkpoints.db``.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready to be invoked.
    """
    graph = StateGraph(NewsForgeState)

    # -- Add nodes --
    graph.add_node("planner_node", planner_node)
    graph.add_node("search_node", search_node)
    graph.add_node("scraper_node", scraper_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("visual_node", visual_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("critic_node", critic_node)

    # -- Wire linear edges --
    graph.add_edge("planner_node", "search_node")
    graph.add_edge("search_node", "scraper_node")
    graph.add_edge("scraper_node", "analysis_node")
    graph.add_edge("analysis_node", "visual_node")
    graph.add_edge("visual_node", "writer_node")
    graph.add_edge("writer_node", "critic_node")

    # -- Entry / finish --
    graph.set_entry_point("planner_node")
    graph.set_finish_point("critic_node")

    # -- Compile with SQLite checkpointer --
    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


# Module-level compiled pipeline (importable as `from orchestrator.graph import pipeline`)
pipeline = build_pipeline()


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    research_id = str(uuid.uuid4())

    print("=" * 60)
    print("NewsForge pipeline — smoke test")
    print(f"Research ID: {research_id}")
    print("=" * 60)

    initial_state: NewsForgeState = {
        "research_id": research_id,
        "topic": "impact of AI on healthcare in 2025",
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    config = {"configurable": {"thread_id": research_id}}

    print(f"\n▶  Topic: {initial_state['topic']}")
    print(f"▶  Thread: {config['configurable']['thread_id']}\n")

    result = pipeline.invoke(initial_state, config=config)

    print(f"\n{'=' * 60}")
    print(f"✔  Pipeline finished.")
    print(f"   pipeline_status : {result['pipeline_status']}")
    print(f"   subtasks        : {len(result.get('subtasks', []))}")
    print(f"   errors          : {result['errors']}")
    print(f"{'=' * 60}")
    print(f"\n📊 Check Langfuse dashboard for traces → {LANGFUSE_HOST}")
