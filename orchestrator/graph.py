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

    On first invocation, generates a fresh report.  On subsequent invocations
    (after Critic feedback), passes ``feedback_notes`` to the Writer so it
    can revise the report to address specific quality issues.

    Reads from state:
        - topic, analysis, search_results, subtasks.
        - critic_feedback: (optional) CriticFeedback dict from a prior Critic pass.
        - revision_count: current revision number.

    Writes to state:
        - draft_report: full Markdown research report string.
        - pipeline_status: set to "writer_complete".
        - errors: appended to on failure.
    """
    topic = state.get("topic", "")
    analysis = state.get("analysis")
    search_results = state.get("search_results", [])
    subtasks = state.get("subtasks", [])
    revision_count = state.get("revision_count", 0)

    if not analysis:
        print("[writer_node] No analysis available — skipping")
        return {"pipeline_status": "writer_skipped"}

    # Check for revision feedback from a prior Critic pass
    critic_feedback = state.get("critic_feedback")
    feedback_notes: list[str] | None = None
    if critic_feedback and not critic_feedback.get("passed", True):
        feedback_notes = critic_feedback.get("feedback_notes", [])
        print(
            f"[writer_node] Revision {revision_count} — "
            f"incorporating {len(feedback_notes)} feedback notes"
        )
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
                feedback_notes=feedback_notes,
            )

            trace.create_event(
                name="writer_complete",
                metadata={
                    "word_count": report["word_count"],
                    "section_count": report["section_count"],
                    "title": report["title"],
                    "revision_count": revision_count,
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

    Evaluates the draft report against the original analysis using a structured
    rubric.  Returns a CriticFeedback dict with pass/fail verdict, quality score,
    and specific revision instructions if the report needs improvement.

    Reads from state:
        - draft_report: the Writer's markdown report string.
        - analysis: AnalysisOutput dict to fact-check against.
        - subtasks: original Planner subtasks for coverage checking.
        - revision_count: current revision number.

    Writes to state:
        - critic_feedback: CriticFeedback dict (passed, quality_score, feedback_notes).
        - revision_count: incremented by 1 (so the router can enforce max revisions).
        - pipeline_status: set to "critic_complete".
        - errors: appended to on failure.
    """
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
                    "revision_count": revision_count,
                },
            )

        langfuse.flush()

        status = "PASSED" if result["passed"] else "NEEDS REVISION"
        print(
            f"[critic_node] {status} — score: {result['quality_score']:.2f}, "
            f"revision: {revision_count}. Trace id: {trace.trace_id}"
        )

        # Build CriticFeedback dict matching the state schema TypedDict
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
        # On failure, pass the report through to avoid blocking the pipeline
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
    """Save the final report locally and optionally to AWS S3 + DynamoDB.

    Reads from state:
        - research_id, topic, draft_report, critic_feedback.

    Writes to state:
        - published_url: local file path or S3 URL.
        - published_record_id: DynamoDB record id or local metadata path.
        - pipeline_status: set to "publisher_complete".
        - completed_at: ISO timestamp.
        - errors: appended to on failure.
    """
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

        print(
            f"[publisher_node] Done — saved to {result['local_report_path']}"
            f"{' + S3' if result['s3_url'] else ''}. "
            f"Trace id: {trace.trace_id}"
        )

        return {
            "published_url": published_url,
            "published_record_id": published_record_id,
            "pipeline_status": "publisher_complete",
            "completed_at": result["published_at"],
        }

    except Exception as e:
        print(f"[publisher_node] ERROR: {e}")
        return {"errors": [f"publisher_node failed: {str(e)}"]}


# ═══════════════════════════════════════════════════════════════════════════
# Human-in-the-Loop (HITL) review node
# ═══════════════════════════════════════════════════════════════════════════
#
# This node uses LangGraph's `interrupt` primitive to PAUSE pipeline
# execution before publishing.  The full graph state is persisted to the
# SQLite checkpointer so the pipeline can be resumed from exactly this
# point after a human approves or rejects the report.
#
# Interview talking point:
#   "I used LangGraph's interrupt primitive to pause execution before
#    publishing. The pipeline state is persisted in SQLite so it can be
#    resumed from exactly the same point after human approval."
#
# Flow:
#   1. critic passes → human_review_node runs
#   2. interrupt() pauses execution, returns review metadata to caller
#   3. Caller (FastAPI /approve or /reject) resumes with:
#        Command(resume={"decision": "approve"})
#   4. interrupt() returns the resume value → node continues
#   5. _human_review_router sends to publisher_node or END
# ═══════════════════════════════════════════════════════════════════════════


def human_review_node(state: NewsForgeState) -> dict[str, Any]:
    """Pause the pipeline for human approval before publishing.

    Uses LangGraph's ``interrupt()`` to suspend execution.  The interrupt
    payload contains a preview of the report so the reviewer can make an
    informed decision without reading the full state.

    When the caller resumes with ``Command(resume={"decision": "approve"})``
    or ``Command(resume={"decision": "reject"})``, the ``interrupt()`` call
    returns that dict and execution continues.

    Reads from state:
        - research_id, topic, draft_report, critic_feedback, revision_count.

    Writes to state:
        - human_decision: ``"approved"`` or ``"rejected"``.
        - pipeline_status: ``"awaiting_approval"`` then updated on resume.
    """
    research_id = state.get("research_id", "unknown")
    topic = state.get("topic", "")
    draft_report = state.get("draft_report", "")
    critic_feedback = state.get("critic_feedback") or {}
    revision_count = state.get("revision_count", 0)
    quality_score = critic_feedback.get("quality_score", 0.0)

    print(f"[human_review_node] Pipeline paused for human review")
    print(f"[human_review_node] Topic: {topic}")
    print(f"[human_review_node] Quality score: {quality_score:.2f}")
    print(f"[human_review_node] Report preview: {draft_report[:200]}...")
    print(f"[human_review_node] Waiting for approval...")

    # ── INTERRUPT — execution pauses here ──────────────────────────────
    # The dict passed to interrupt() is returned to the caller as the
    # interrupt payload.  The caller inspects it, then resumes with a
    # Command(resume={...}) to continue execution.
    review_payload = interrupt({
        "research_id": research_id,
        "topic": topic,
        "quality_score": quality_score,
        "report_preview": draft_report[:1000],
        "word_count": len(draft_report.split()),
        "revision_count": revision_count,
    })
    # ── Execution resumes here after Command(resume=...) ───────────────

    decision = review_payload.get("decision", "reject")
    print(f"[human_review_node] Decision received: {decision}")

    return {
        "human_decision": decision,
        "pipeline_status": f"human_{decision}",
    }


def _human_review_router(state: NewsForgeState) -> str:
    """Route based on the human reviewer's decision.

    Returns:
        ``"approved"``  → publisher_node
        ``"rejected"``  → END (skip publishing)
    """
    decision = state.get("human_decision", "rejected")
    if decision == "approve":
        print("[Router] Human approved — proceeding to publisher.")
        return "approved"
    else:
        print(f"[Router] Human rejected — ending pipeline without publishing.")
        return "rejected"


# ═══════════════════════════════════════════════════════════════════════════
# Routing function for Writer ↔ Critic revision loop
# ═══════════════════════════════════════════════════════════════════════════


def _critic_router(state: NewsForgeState) -> str:
    """Decide whether the pipeline is done or needs another Writer revision.

    Called by LangGraph's conditional edge after critic_node completes.
    Returns a string key that maps to either:
      - ``"done"``   → pipeline finishes (report passed or max revisions reached).
      - ``"revise"`` → loops back to writer_node for another pass.

    Guard rails:
    1. If the Critic says ``passed=True``, the report is good — finish.
    2. If ``revision_count >= MAX_REVISIONS + 1``, stop even if still failing
       (revision_count is already incremented by critic_node before this runs,
       so after the first critic pass it's 1, after second it's 2, etc.).
    3. Otherwise, route back to the Writer for revision.

    Args:
        state: The current NewsForgeState after critic_node.

    Returns:
        ``"done"`` or ``"revise"``.
    """
    critic_feedback = state.get("critic_feedback")
    revision_count = state.get("revision_count", 0)

    # No feedback means critic was skipped or errored — finish
    if not critic_feedback:
        print("[Router] No critic feedback — finishing pipeline.")
        return "done"

    passed = critic_feedback.get("passed", False)
    score = critic_feedback.get("quality_score", 0.0)

    if passed:
        print(f"[Router] Score {score:.2f} passed. Pipeline complete.")
        return "done"

    # revision_count has already been incremented by critic_node.
    # After first review: revision_count=1, after second: revision_count=2, etc.
    # We allow MAX_REVISIONS revision passes, so we stop when
    # revision_count > MAX_REVISIONS.
    if revision_count > MAX_REVISIONS:
        print(
            f"[Router] Score {score:.2f} did not pass, but max revisions "
            f"({MAX_REVISIONS}) reached. Finishing with current report."
        )
        return "done"

    notes_count = len(critic_feedback.get("feedback_notes", []))
    print(
        f"[Router] Score {score:.2f} — routing back to Writer for "
        f"revision {revision_count}/{MAX_REVISIONS} "
        f"({notes_count} feedback notes)"
    )
    return "revise"


# ═══════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════


def build_pipeline():
    """Construct, compile, and return the full NewsForge LangGraph pipeline.

    Pipeline topology:
        planner → search → scraper → analysis → visual → writer → critic
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

    The critic_node uses a conditional edge:
      - "done"   → human_review_node (report passed or max revisions reached)
      - "revise" → writer_node (for another revision pass)

    The human_review_node uses LangGraph's ``interrupt()`` to pause execution.
    The caller resumes with ``Command(resume={"decision": "approve"})`` or
    ``Command(resume={"decision": "reject"})``.  A conditional edge then
    routes to publisher_node (approved) or END (rejected).

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
    graph.add_node("human_review_node", human_review_node)
    graph.add_node("publisher_node", publisher_node)

    # Wire linear edges (planner through writer)
    graph.add_edge("planner_node", "search_node")
    graph.add_edge("search_node", "scraper_node")
    graph.add_edge("scraper_node", "analysis_node")
    graph.add_edge("analysis_node", "visual_node")
    graph.add_edge("visual_node", "writer_node")
    graph.add_edge("writer_node", "critic_node")

    # Conditional edge: Critic → Human Review (done) or Critic → Writer (revise)
    graph.add_conditional_edges(
        "critic_node",
        _critic_router,
        {
            "done": "human_review_node",
            "revise": "writer_node",
        },
    )

    # Conditional edge: Human Review → Publisher (approved) or END (rejected)
    graph.add_conditional_edges(
        "human_review_node",
        _human_review_router,
        {
            "approved": "publisher_node",
            "rejected": END,
        },
    )

    # Publisher → END
    graph.add_edge("publisher_node", END)

    # Entry point
    graph.set_entry_point("planner_node")

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
        "human_decision": None,
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

    # First invoke — pipeline will PAUSE at human_review_node (interrupt)
    result = pipeline.invoke(initial_state, config=config)

    # The pipeline is now paused at the interrupt.
    # Auto-approve for the smoke test so it completes end-to-end.
    print(f"\n{'=' * 60}")
    print("⏸  Pipeline paused at human_review_node — auto-approving...")
    print(f"{'=' * 60}\n")

    # Resume with approval — this continues from the interrupt point
    result = pipeline.invoke(
        Command(resume={"decision": "approve"}),
        config=config,
    )

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

    # Print critic feedback
    if result.get("critic_feedback"):
        feedback = result["critic_feedback"]
        status = "PASSED" if feedback.get("passed") else "NEEDS REVISION"
        print(f"\n🎯 CRITIC FEEDBACK:")
        print(f"   Verdict        : {status}")
        print(f"   Quality score  : {feedback.get('quality_score', 0):.2f}")
        print(f"   Revision count : {result.get('revision_count', 0)}")
        notes = feedback.get("feedback_notes", [])
        if notes:
            print(f"   Feedback notes : {len(notes)}")
            for note in notes:
                print(f"     - {note}")

    # Print publisher results
    if result.get("published_url"):
        print(f"\n📦 PUBLISHER:")
        print(f"   Published URL  : {result['published_url']}")
        print(f"   Record ID      : {result.get('published_record_id', 'N/A')}")
        print(f"   Completed at   : {result.get('completed_at', 'N/A')}")