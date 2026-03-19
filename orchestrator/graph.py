"""LangGraph StateGraph wiring all 7 NewsForge agents into a pipeline.

Pipeline: planner → search → scraper → analysis → writer → critic
          (critic ↔ writer revision loop) → human_review → publisher
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid
from datetime import datetime, timezone
from typing import Any

from langfuse import Langfuse
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agents.analysis import AnalysisAgent
from agents.critic import CriticAgent, MAX_REVISIONS
from agents.planner import PlannerAgent
from agents.scraper import ScraperAgent
from agents.search import SearchAgent
from agents.publisher import PublisherAgent
from agents.writer import WriterAgent
from config.settings import (
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)
from orchestrator.checkpointer import get_checkpointer
from orchestrator.state import NewsForgeState

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)


def planner_node(state: NewsForgeState) -> dict[str, Any]:
    """Decompose the user's research topic into prioritised sub-tasks."""
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
                span.update(output={
                    "subtasks_generated": len(subtasks),
                    "subtasks": subtasks,
                })

            trace.create_event(
                name="planner_complete",
                metadata={"subtask_count": len(subtasks)},
            )

        langfuse.flush()
        print(f"[planner_node] Done — {len(subtasks)} subtasks generated.")

        return {
            "subtasks": subtasks,
            "pipeline_status": "planner_complete",
        }

    except Exception as e:
        print(f"[planner_node] ERROR: {e}")
        return {"errors": [f"planner_node failed: {str(e)}"]}


def search_node(state: NewsForgeState) -> dict[str, Any]:
    """Execute Tavily web searches for each sub-task."""
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
                    sub_span.update(output={"results_count": len(subtask_results)})

                if not subtask_results:
                    failed_subtask_ids.append(sid)

            trace.create_event(
                name="search_complete",
                metadata={
                    "total_results": len(results),
                    "failed_subtasks": failed_subtask_ids,
                },
            )

        langfuse.flush()
        print(f"[search_node] Done — {len(results)} results across {len(subtasks)} subtasks.")

        return {
            "search_results": results,
            "pipeline_status": "search_complete",
        }

    except Exception as e:
        print(f"[search_node] ERROR: {e}")
        return {"errors": [f"search_node failed: {str(e)}"]}


def scraper_node(state: NewsForgeState) -> dict[str, Any]:
    """Fetch and clean full-page content for each search result URL."""
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
        print(f"[scraper_node] Done — {len(scraped)} pages scraped ({success_count} success).")

        return {
            "scraped_content": scraped,
            "pipeline_status": "scraper_complete",
        }

    except Exception as e:
        print(f"[scraper_node] ERROR: {e}")
        return {"errors": [f"scraper_node failed: {str(e)}"]}


def analysis_node(state: NewsForgeState) -> dict[str, Any]:
    """Extract themes, key facts, and contradictions from scraped content."""
    scraped_content: list[dict[str, Any]] = state.get("scraped_content", [])
    subtasks: list[dict[str, Any]] = state.get("subtasks", [])
    print(f"[analysis_node] Received {len(scraped_content)} scraped items, {len(subtasks)} subtasks")

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
                })

            trace.create_event(
                name="analysis_complete",
                metadata={
                    "confidence_score": analysis.get("confidence_score", 0.0),
                    "themes_count": len(analysis.get("themes", [])),
                },
            )

        langfuse.flush()
        print(
            f"[analysis_node] Done — "
            f"{len(analysis.get('themes', []))} themes, "
            f"{len(analysis.get('key_facts', []))} facts, "
            f"confidence: {analysis.get('confidence_score', 0.0):.2f}"
        )

        return {
            "analysis": analysis,
            "pipeline_status": "analysis_complete",
        }

    except Exception as e:
        print(f"[analysis_node] ERROR: {e}")
        return {"errors": [f"analysis_node failed: {str(e)}"]}


def writer_node(state: NewsForgeState) -> dict[str, Any]:
    """Compose a research report from analysis and search results."""
    topic = state.get("topic", "")
    analysis = state.get("analysis")
    search_results = state.get("search_results", [])
    subtasks = state.get("subtasks", [])
    revision_count = state.get("revision_count", 0)

    if not analysis:
        print("[writer_node] No analysis available — skipping")
        return {"pipeline_status": "writer_skipped"}

    critic_feedback = state.get("critic_feedback")
    feedback_notes: list[str] | None = None
    if critic_feedback and not critic_feedback.get("passed", True):
        feedback_notes = critic_feedback.get("feedback_notes", [])
        print(f"[writer_node] Revision {revision_count} — {len(feedback_notes)} feedback notes")
    else:
        print(f"[writer_node] Writing report for: {topic!r}")

    try:
        with langfuse.start_as_current_observation(
            name="writer_node",
            metadata={
                "research_id": state.get("research_id", ""),
                "topic": topic,
                "revision_count": revision_count,
                "is_revision": feedback_notes is not None,
            },
        ) as trace:
            agent = WriterAgent()
            report = agent.run(
                topic=topic,
                analysis=analysis,
                search_results=search_results,
                subtasks=subtasks,
                feedback_notes=feedback_notes,
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
        print(f"[writer_node] Done — {report['word_count']} words, {report['section_count']} sections.")

        return {
            "draft_report": report["full_report"],
            "pipeline_status": "writer_complete",
        }

    except Exception as e:
        print(f"[writer_node] ERROR: {e}")
        return {"errors": [f"writer_node failed: {str(e)}"]}


def critic_node(state: NewsForgeState) -> dict[str, Any]:
    """Review and score the draft report for quality and accuracy."""
    draft_report = state.get("draft_report", "")
    analysis = state.get("analysis") or {}
    subtasks = state.get("subtasks", [])
    revision_count = state.get("revision_count", 0)

    if not draft_report:
        print("[critic_node] No draft report available — skipping")
        return {
            "critic_feedback": {
                "passed": False,
                "quality_score": 0.0,
                "feedback_notes": ["No draft report was produced by the Writer"],
            },
            "pipeline_status": "critic_skipped",
        }

    print(f"[critic_node] Reviewing report (revision {revision_count})...")

    try:
        with langfuse.start_as_current_observation(
            name="critic_node",
            metadata={
                "research_id": state.get("research_id", ""),
                "revision_count": revision_count,
                "report_words": len(draft_report.split()),
            },
        ) as trace:
            agent = CriticAgent()
            result = agent.run(
                draft_report=draft_report,
                analysis=analysis,
                subtasks=subtasks,
                revision_count=revision_count,
            )

            trace.create_event(
                name="critic_complete",
                metadata={
                    "passed": result["passed"],
                    "quality_score": result["quality_score"],
                    "feedback_count": len(result.get("feedback_notes", [])),
                },
            )

        langfuse.flush()

        status = "PASSED" if result["passed"] else "NEEDS REVISION"
        print(f"[critic_node] {status} — score: {result['quality_score']:.2f}")

        critic_feedback = {
            "passed": result["passed"],
            "quality_score": result["quality_score"],
            "feedback_notes": result.get("feedback_notes", []),
        }

        return {
            "critic_feedback": critic_feedback,
            "revision_count": revision_count + 1,
            "pipeline_status": "critic_complete",
        }

    except Exception as e:
        print(f"[critic_node] ERROR: {e}")
        return {
            "critic_feedback": {
                "passed": True,
                "quality_score": 0.0,
                "feedback_notes": [],
            },
            "revision_count": revision_count + 1,
            "pipeline_status": "critic_error",
            "errors": [f"critic_node failed: {str(e)}"],
        }


def publisher_node(state: NewsForgeState) -> dict[str, Any]:
    """Save the final report locally and optionally to AWS S3 + DynamoDB."""
    research_id = state.get("research_id", "unknown")
    topic = state.get("topic", "")
    draft_report = state.get("draft_report", "")
    critic_feedback = state.get("critic_feedback") or {}

    if not draft_report:
        print("[publisher_node] No draft report — skipping")
        return {"pipeline_status": "publisher_skipped"}

    print(f"[publisher_node] Publishing report for: {topic!r}")

    try:
        with langfuse.start_as_current_observation(
            name="publisher_node",
            metadata={
                "research_id": research_id,
                "topic": topic,
                "report_words": len(draft_report.split()),
            },
        ) as trace:
            agent = PublisherAgent()
            result = agent.run(
                research_id=research_id,
                topic=topic,
                draft_report=draft_report,
                critic_feedback=critic_feedback,
            )

            trace.create_event(
                name="publisher_complete",
                metadata={
                    "local_report_path": result["local_report_path"],
                    "s3_url": result["s3_url"],
                    "aws_enabled": result["aws_enabled"],
                },
            )

        langfuse.flush()

        published_url = result["s3_url"] or result["local_report_path"]
        published_record_id = result["dynamodb_record_id"] or result["local_metadata_path"]

        print(f"[publisher_node] Done — saved to {result['local_report_path']}")

        return {
            "published_url": published_url,
            "published_record_id": published_record_id,
            "pipeline_status": "publisher_complete",
            "completed_at": result["published_at"],
        }

    except Exception as e:
        print(f"[publisher_node] ERROR: {e}")
        return {"errors": [f"publisher_node failed: {str(e)}"]}


def human_review_node(state: NewsForgeState) -> dict[str, Any]:
    """Pause the pipeline for human approval before publishing.

    Uses LangGraph's interrupt() to suspend execution. The caller resumes
    with Command(resume={"decision": "approve"}) or "reject".
    """
    research_id = state.get("research_id", "unknown")
    topic = state.get("topic", "")
    draft_report = state.get("draft_report", "")
    critic_feedback = state.get("critic_feedback") or {}
    revision_count = state.get("revision_count", 0)
    quality_score = critic_feedback.get("quality_score", 0.0)

    print(f"[human_review_node] Pipeline paused — topic: {topic}, score: {quality_score:.2f}")

    review_payload = interrupt({
        "research_id": research_id,
        "topic": topic,
        "quality_score": quality_score,
        "report_preview": draft_report[:1000],
        "word_count": len(draft_report.split()),
        "revision_count": revision_count,
    })

    decision = review_payload.get("decision", "reject")
    print(f"[human_review_node] Decision received: {decision}")

    return {
        "human_decision": decision,
        "pipeline_status": f"human_{decision}",
    }


def _human_review_router(state: NewsForgeState) -> str:
    """Route based on human reviewer's decision."""
    decision = state.get("human_decision", "rejected")
    if decision == "approve":
        return "approved"
    return "rejected"


def _critic_router(state: NewsForgeState) -> str:
    """Decide whether the report passes or needs another revision.

    Returns "done" (→ human review) or "revise" (→ writer for another pass).
    """
    critic_feedback = state.get("critic_feedback")
    revision_count = state.get("revision_count", 0)

    if not critic_feedback:
        return "done"

    passed = critic_feedback.get("passed", False)
    score = critic_feedback.get("quality_score", 0.0)

    if passed:
        print(f"[Router] Score {score:.2f} passed.")
        return "done"

    if revision_count > MAX_REVISIONS:
        print(f"[Router] Max revisions ({MAX_REVISIONS}) reached. Finishing.")
        return "done"

    notes_count = len(critic_feedback.get("feedback_notes", []))
    print(f"[Router] Score {score:.2f} — revision {revision_count}/{MAX_REVISIONS} ({notes_count} notes)")
    return "revise"


def build_pipeline():
    """Construct and compile the full NewsForge LangGraph pipeline.

    Topology:
        planner → search → scraper → analysis → writer → critic
                                                   ↑         |
                                                   └─ revise ─┘
                                                      (conditional)
                                                         |
                                                  human_review_node
                                                (interrupt — pauses)
                                                 /              \\
                                           approved          rejected
                                               |                |
                                          publisher → END      END
    """
    graph = StateGraph(NewsForgeState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("search_node", search_node)
    graph.add_node("scraper_node", scraper_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("human_review_node", human_review_node)
    graph.add_node("publisher_node", publisher_node)

    graph.add_edge("planner_node", "search_node")
    graph.add_edge("search_node", "scraper_node")
    graph.add_edge("scraper_node", "analysis_node")
    graph.add_edge("analysis_node", "writer_node")
    graph.add_edge("writer_node", "critic_node")

    graph.add_conditional_edges(
        "critic_node",
        _critic_router,
        {"done": "human_review_node", "revise": "writer_node"},
    )

    graph.add_conditional_edges(
        "human_review_node",
        _human_review_router,
        {"approved": "publisher_node", "rejected": END},
    )

    graph.add_edge("publisher_node", END)
    graph.set_entry_point("planner_node")

    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)


pipeline = build_pipeline()


if __name__ == "__main__":
    research_id = str(uuid.uuid4())

    print(f"NewsForge pipeline — smoke test (ID: {research_id})")

    initial_state: NewsForgeState = {
        "research_id": research_id,
        "topic": "impact of AI on healthcare in 2025",
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    config = {"configurable": {"thread_id": research_id}}

    # First invoke — pauses at human_review_node
    result = pipeline.invoke(initial_state, config=config)

    # Auto-approve for smoke test
    print("Pipeline paused — auto-approving...")
    result = pipeline.invoke(
        Command(resume={"decision": "approve"}),
        config=config,
    )

    print(f"Pipeline finished. Status: {result['pipeline_status']}")
    print(f"Subtasks: {len(result.get('subtasks', []))}")
    print(f"Results: {len(result.get('search_results', []))}")
    print(f"Errors: {result['errors']}")
