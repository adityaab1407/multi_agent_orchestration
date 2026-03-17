"""
orchestrator/graph.py

Full LangGraph StateGraph wiring all 7 NewsForge agents into a linear pipeline.

Pipeline order:
    planner → search → scraper → analysis → visual → writer → critic

Currently only planner_node and search_node contain placeholder logic;
the remaining five nodes are pass-through stubs that will be implemented in
Phase 2.  Every node prints a status line so graph execution is visible in
the terminal during development.
"""

from datetime import datetime, timezone

from langgraph.graph import StateGraph

from orchestrator.state import NewsForgeState
from orchestrator.checkpointer import get_checkpointer


# ═══════════════════════════════════════════════════════════════════════════
# Node functions — executed in order by the LangGraph runner
# ═══════════════════════════════════════════════════════════════════════════


def planner_node(state: NewsForgeState) -> dict:
    """Decompose the user's research topic into prioritised sub-tasks.

    Reads from state:
        - topic: the user-supplied research topic string.

    Writes to state:
        - pipeline_status: set to ``"planner_node running"``.
        - subtasks: (Phase 2) a list of Subtask dicts produced by the LLM.

    Current behaviour (Day 1 stub):
        Logs a message and updates pipeline_status only.
    """
    print(f"[planner_node] Running — real implementation in Day 2")
    return {"pipeline_status": "planner_node running"}


def search_node(state: NewsForgeState) -> dict:
    """Execute web searches for each sub-task produced by the Planner.

    Reads from state:
        - subtasks: list of Subtask dicts to generate search queries from.

    Writes to state:
        - pipeline_status: set to ``"search_node running"``.
        - search_results: (Phase 2) accumulated SearchResult dicts from Tavily.

    Current behaviour (Day 1 stub):
        Logs a message and updates pipeline_status only.
    """
    print(f"[search_node] Running — real implementation in Day 2")
    return {"pipeline_status": "search_node running"}


def scraper_node(state: NewsForgeState) -> dict:
    """Fetch and clean full-page content for each search result URL.

    Reads from state:
        - search_results: list of SearchResult dicts containing URLs to scrape.

    Writes to state:
        - pipeline_status: set to ``"scraper_node running"``.
        - scraped_content: (Phase 2) list of ScrapedContent dicts with raw text
          and chunked segments.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[scraper_node] STUB — pass-through, no scraping yet")
    return {"pipeline_status": "scraper_node running"}


def analysis_node(state: NewsForgeState) -> dict:
    """Run sentiment analysis, topic clustering, and fact extraction.

    Reads from state:
        - scraped_content: cleaned text chunks from the Scraper.
        - search_results: original search metadata for cross-referencing.

    Writes to state:
        - pipeline_status: set to ``"analysis_node running"``.
        - analysis: (Phase 2) an AnalysisOutput dict with themes, key facts,
          contradictions, and a confidence score.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[analysis_node] STUB — pass-through, no analysis yet")
    return {"pipeline_status": "analysis_node running"}


def visual_node(state: NewsForgeState) -> dict:
    """Generate charts, diagrams, and visual summaries from the analysis.

    Reads from state:
        - analysis: AnalysisOutput dict containing themes and key facts.

    Writes to state:
        - pipeline_status: set to ``"visual_node running"``.
        - visuals: (Phase 2) list of VisualOutput dicts (base64 images or
          chart JSON).

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[visual_node] STUB — pass-through, no visuals yet")
    return {"pipeline_status": "visual_node running"}


def writer_node(state: NewsForgeState) -> dict:
    """Compose a polished research report from the analysis and visuals.

    Reads from state:
        - topic: original research topic for the report title.
        - analysis: structured analysis output.
        - visuals: list of generated visuals to embed.
        - critic_feedback: (revision loop) feedback from a prior Critic pass.

    Writes to state:
        - pipeline_status: set to ``"writer_node running"``.
        - draft_report: (Phase 2) Markdown string of the full report.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[writer_node] STUB — pass-through, no writing yet")
    return {"pipeline_status": "writer_node running"}


def critic_node(state: NewsForgeState) -> dict:
    """Review and score the draft report for quality, accuracy, and coverage.

    Reads from state:
        - draft_report: the Writer's Markdown report.
        - analysis: original analysis for fact-checking against the report.

    Writes to state:
        - pipeline_status: set to ``"critic_node running"``.
        - critic_feedback: (Phase 2) a CriticFeedback dict with pass/fail,
          quality score, and revision notes.
        - revision_count: (Phase 2) incremented if the report needs revision.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[critic_node] STUB — pass-through, no critique yet")
    return {"pipeline_status": "critic_node running"}


# ═══════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════

def build_pipeline() -> object:
    """Construct, compile, and return the full NewsForge LangGraph pipeline.

    The graph is a simple linear chain for Day 1:

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
    print("=" * 60)
    print("NewsForge pipeline — smoke test")
    print("=" * 60)

    initial_state: NewsForgeState = {
        "research_id": "test_001",
        "topic": "impact of AI on healthcare",
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
        "pipeline_status": "initialized",
        "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    config = {"configurable": {"thread_id": "test_001"}}

    print(f"\n▶  Topic: {initial_state['topic']}")
    print(f"▶  Thread: {config['configurable']['thread_id']}\n")

    result = pipeline.invoke(initial_state, config=config)

    print(f"\n{'=' * 60}")
    print(f"✔  Pipeline finished.")
    print(f"   pipeline_status : {result['pipeline_status']}")
    print(f"   errors          : {result['errors']}")
    print(f"{'=' * 60}")
