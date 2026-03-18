"""
orchestrator/graph.py

Full LangGraph StateGraph wiring all 7 NewsForge agents into a linear pipeline.

Pipeline order:
    planner → search → scraper → analysis → visual → writer → critic

planner_node calls PlannerAgent (Groq LLM) with Langfuse tracing.
search_node calls SearchAgent (Tavily) with Langfuse tracing.
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

from agents.analysis import AnalysisAgent
from agents.planner import PlannerAgent
from agents.scraper import ScraperAgent
from agents.search import SearchAgent
from agents.writer import WriterAgent
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

    This is the real implementation — it calls PlannerAgent().run()
    which invokes Groq (llama-3.3-70b-versatile) in a ReAct loop.

    Reads from state:
        - research_id: unique pipeline run identifier (used as Langfuse trace id).
        - topic: the user-supplied research topic string.

    Writes to state:
        - subtasks: a list of Subtask dicts produced by the LLM.
        - pipeline_status: set to "planner_complete".
        - errors: appended to on failure.
    """
    research_id: str = state["research_id"]
    topic: str = state["topic"]

    try:
        with langfuse.start_as_current_observation(
            name="planner_node",
            metadata={"topic": topic, "research_id": research_id},
        ) as trace:

            with trace.start_as_current_observation(
                name="react_loop",
                input={"topic": topic},
            ) as span:
                print(f"[planner_node] Calling PlannerAgent for topic: {topic!r}")
                agent = PlannerAgent()
                subtasks: list[dict[str, Any]] = agent.run(
                    topic=topic, research_id=research_id
                )
                # FIX: use update() then end() — Langfuse v3 API
                span.update(output={
                    "subtasks_generated": len(subtasks),
                    "subtasks": subtasks,
                })
                span.end()

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
    """Execute Tavily web searches for each sub-task produced by the Planner.

    This is the real implementation — it calls SearchAgent().run()
    which fires one Tavily search per subtask with retry logic.

    Reads from state:
        - subtasks: list of Subtask dicts containing search_query strings.

    Writes to state:
        - search_results: flat list of SearchResult dicts from Tavily.
        - pipeline_status: set to "search_complete".
        - errors: appended to on failure.
    """
    subtasks: list[dict[str, Any]] = state.get("subtasks", [])
    print(f"[search_node] Received {len(subtasks)} subtasks")

    try:
        with langfuse.start_as_current_observation(
            name="search_node",
            metadata={
                "research_id": state.get("research_id", ""),
                "subtask_count": len(subtasks),
            },
        ) as trace:

            agent = SearchAgent()
            results: list[dict[str, Any]] = agent.run(subtasks)

            # One span per subtask for granular observability
            failed_subtask_ids: list[str] = []
            for subtask in subtasks:
                sid = subtask.get("subtask_id", "unknown")
                subtask_results = [
                    r for r in results if r.get("subtask_id") == sid
                ]
                with trace.start_as_current_observation(
                    name=f"search_{sid}",
                    metadata={
                        "query": subtask.get("search_query", ""),
                        "priority": subtask.get("priority", 0),
                    },
                ) as sub_span:
                    # FIX: use update() then end() — Langfuse v3 API
                    sub_span.update(output={"results_count": len(subtask_results)})
                    sub_span.end()

                if not subtask_results:
                    failed_subtask_ids.append(sid)

            trace.create_event(
                name="search_complete",
                metadata={
                    "total_results": len(results),
                    "failed_subtasks": failed_subtask_ids,
                    "results_preview": [
                        {
                            "subtask_id": r["subtask_id"],
                            "title": r["title"],
                            "url": r["url"],
                            "score": r["relevance_score"],
                        }
                        for r in results[:3]
                    ],
                },
            )

        langfuse.flush()

        print(
            f"[search_node] Done — {len(results)} results across "
            f"{len(subtasks)} subtasks. Trace id: {trace.trace_id}"
        )

        return {
            "search_results": results,
            "pipeline_status": "search_complete",
        }

    except Exception as e:
        print(f"[search_node] ERROR: {e}")
        return {"errors": [f"search_node failed: {str(e)}"]}


def scraper_node(state: NewsForgeState) -> dict[str, Any]:
    """Fetch and clean full-page content for each search result URL.

    Reads from state:
        - search_results: list of SearchResult dicts containing URLs to scrape.

    Writes to state:
        - scraped_content: list of ScrapedContent dicts.
        - pipeline_status: set to "scraper_complete".
        - errors: appended to on failure.
    """
    search_results: list[dict[str, Any]] = state.get("search_results", [])
    print(f"[scraper_node] Received {len(search_results)} search results to scrape")

    try:
        with langfuse.start_as_current_observation(
            name="scraper_node",
            metadata={
                "research_id": state.get("research_id", ""),
                "result_count": len(search_results),
            },
        ) as trace:

            agent = ScraperAgent()
            scraped: list[dict[str, Any]] = agent.run(search_results)

            # One span per scraped item for granular observability
            for item in scraped:
                with trace.start_as_current_observation(
                    name=f"scrape_{item['result_id']}",
                    metadata={"url": item["url"]},
                ) as span:
                    span.update(output={
                        "status": item["scrape_status"],
                        "method": item["scrape_method"],
                        "word_count": item["word_count"],
                    })
                    span.end()

            success_count = sum(1 for s in scraped if s["scrape_status"] == "success")
            trace.create_event(
                name="scraper_complete",
                metadata={
                    "total_scraped": len(scraped),
                    "success_count": success_count,
                    "failed_count": len(scraped) - success_count,
                },
            )

        langfuse.flush()

        print(
            f"[scraper_node] Done — {len(scraped)} pages scraped "
            f"({success_count} success). Trace id: {trace.trace_id}"
        )

        return {
            "scraped_content": scraped,
            "pipeline_status": "scraper_complete",
        }

    except Exception as e:
        print(f"[scraper_node] ERROR: {e}")
        return {"errors": [f"scraper_node failed: {str(e)}"]}


def analysis_node(state: NewsForgeState) -> dict[str, Any]:
    """Extract themes, key facts, and contradictions from scraped content.

    Reads from state:
        - scraped_content: list of ScrapedContent dicts from the Scraper.
        - subtasks: original Planner subtasks for research-topic context.

    Writes to state:
        - analysis: AnalysisOutput dict with themes, key_facts, contradictions,
            confidence_score, coverage_notes, and iteration.
        - pipeline_status: set to "analysis_complete".
        - errors: appended to on failure.
    """
    scraped_content: list[dict[str, Any]] = state.get("scraped_content", [])
    subtasks: list[dict[str, Any]] = state.get("subtasks", [])
    print(
        f"[analysis_node] Received {len(scraped_content)} scraped items, "
        f"{len(subtasks)} subtasks"
    )

    try:
        with langfuse.start_as_current_observation(
            name="analysis_node",
            metadata={
                "research_id": state.get("research_id", ""),
                "scraped_count": len(scraped_content),
                "subtask_count": len(subtasks),
            },
        ) as trace:

            with trace.start_as_current_observation(
                name="react_loop",
                input={
                    "scraped_count": len(scraped_content),
                    "topic": state.get("topic", ""),
                },
            ) as span:
                agent = AnalysisAgent()
                analysis: dict[str, Any] = agent.run(
                    scraped_content=scraped_content,
                    subtasks=subtasks,
                )
                span.update(output={
                    "themes_count": len(analysis.get("themes", [])),
                    "facts_count": len(analysis.get("key_facts", [])),
                    "contradictions_count": len(analysis.get("contradictions", [])),
                    "confidence_score": analysis.get("confidence_score", 0.0),
                    "iteration": analysis.get("iteration", 1),
                })
                span.end()

            trace.create_event(
                name="analysis_complete",
                metadata={
                    "confidence_score": analysis.get("confidence_score", 0.0),
                    "themes_count": len(analysis.get("themes", [])),
                    "facts_count": len(analysis.get("key_facts", [])),
                },
            )

        langfuse.flush()

        print(
            f"[analysis_node] Done — "
            f"{len(analysis.get('themes', []))} themes, "
            f"{len(analysis.get('key_facts', []))} facts, "
            f"confidence: {analysis.get('confidence_score', 0.0):.2f}. "
            f"Trace id: {trace.trace_id}"
        )

        return {
            "analysis": analysis,
            "pipeline_status": "analysis_complete",
        }

    except Exception as e:
        print(f"[analysis_node] ERROR: {e}")
        return {"errors": [f"analysis_node failed: {str(e)}"]}


def visual_node(state: NewsForgeState) -> dict[str, Any]:
    """Generate charts, diagrams, and visual summaries from the analysis.

    Reads from state:
        - analysis: AnalysisOutput dict containing themes and key facts.

    Writes to state:
        - visuals: (Phase 2) list of VisualOutput dicts.

    Current behaviour:
        Pass-through stub — returns state unchanged.
    """
    # Phase 2 — not yet implemented
    print("[visual_node] STUB — pass-through, no visuals yet")
    return {"pipeline_status": "visual_node running"}


def writer_node(state: NewsForgeState) -> dict[str, Any]:
    """Compose a polished research report from the analysis and search results.

    Reads from state:
        - topic: the user-supplied research topic string.
        - analysis: AnalysisOutput dict from the Analysis Agent.
        - search_results: list of SearchResult dicts (for citation building).
        - subtasks: list of Subtask dicts (for research angle context).

    Writes to state:
        - draft_report: full Markdown research report string.
        - pipeline_status: set to "writer_complete".
        - errors: appended to on failure.
    """
    topic = state.get("topic", "")
    analysis = state.get("analysis")
    search_results = state.get("search_results", [])
    subtasks = state.get("subtasks", [])

    if not analysis:
        print("[writer_node] No analysis available — skipping")
        return {"pipeline_status": "writer_skipped"}

    print(f"[writer_node] Writing report for: {topic!r}")

    try:
        with langfuse.start_as_current_observation(
            name="writer_node",
            metadata={
                "research_id": state.get("research_id", ""),
                "topic": topic,
                "themes_count": len(analysis.get("themes", [])),
                "sources_count": analysis.get("sources_analysed", 0),
            },
        ) as trace:

            agent = WriterAgent()
            report = agent.run(
                topic=topic,
                analysis=analysis,
                search_results=search_results,
                subtasks=subtasks,
            )

            trace.create_event(
                name="writer_complete",
                metadata={
                    "word_count": report["word_count"],
                    "section_count": report["section_count"],
                    "title": report["title"],
                },
            )

        langfuse.flush()

        print(
            f"[writer_node] Done — {report['word_count']} words, "
            f"{report['section_count']} sections. "
            f"Trace id: {trace.trace_id}"
        )

        return {
            "draft_report": report["full_report"],
            "pipeline_status": "writer_complete",
        }

    except Exception as e:
        print(f"[writer_node] ERROR: {e}")
        return {"errors": [f"writer_node failed: {str(e)}"]}


def critic_node(state: NewsForgeState) -> dict[str, Any]:
    """Review and score the draft report for quality, accuracy, and coverage.

    Reads from state:
        - draft_report, analysis.

    Writes to state:
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
    persisted to data/newsforge_checkpoints.db.

    Returns:
        A compiled LangGraph CompiledStateGraph ready to be invoked.
    """
    graph = StateGraph(NewsForgeState)

    # Add nodes
    graph.add_node("planner_node", planner_node)
    graph.add_node("search_node", search_node)
    graph.add_node("scraper_node", scraper_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("visual_node", visual_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("critic_node", critic_node)

    # Wire linear edges
    graph.add_edge("planner_node", "search_node")
    graph.add_edge("search_node", "scraper_node")
    graph.add_edge("scraper_node", "analysis_node")
    graph.add_edge("analysis_node", "visual_node")
    graph.add_edge("visual_node", "writer_node")
    graph.add_edge("writer_node", "critic_node")

    # Entry / finish
    graph.set_entry_point("planner_node")
    graph.set_finish_point("critic_node")

    # Compile with SQLite checkpointer
    checkpointer = get_checkpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


# Module-level compiled pipeline
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

    # FIX: capture result (was final_state — undefined variable)
    result = pipeline.invoke(initial_state, config=config)

    print(f"\n{'=' * 60}")
    print(f"✔  Pipeline finished.")
    print(f"   pipeline_status  : {result['pipeline_status']}")
    print(f"   subtasks         : {len(result.get('subtasks', []))}")
    print(f"   search_results   : {len(result.get('search_results', []))}")
    print(f"   errors           : {result['errors']}")
    print(f"{'=' * 60}")
    print(f"\n📊 Check Langfuse dashboard for traces → {LANGFUSE_HOST}")

    # Print actual subtasks
    print("\n📋 SUBTASKS GENERATED:")
    print("-" * 60)
    for subtask in result["subtasks"]:
        print(f"  [{subtask['subtask_id']}] {subtask['title']}")
        print(f"  Query    : {subtask['search_query']}")
        print(f"  Priority : {subtask['priority']}")
        print(f"  Reasoning: {subtask['reasoning']}")
        print()

    # Print actual search results
    print("\n🔍 SEARCH RESULTS:")
    print("-" * 60)
    for r in result["search_results"]:
        print(f"  [{r['result_id']}] linked to {r['subtask_id']}")
        print(f"  Title    : {r['title']}")
        print(f"  URL      : {r['url']}")
        print(f"  Score    : {r['relevance_score']:.2f}")
        print(f"  Snippet  : {r['snippet'][:120]}...")
        print()

    # Print draft report preview
    if result.get("draft_report"):
        report = result["draft_report"]
        print(f"\n📝 DRAFT REPORT PREVIEW:")
        print(f"   Words  : {len(report.split())}")
        print(f"   Preview: {report[:500]}...")