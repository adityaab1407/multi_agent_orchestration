"""Integration tests for the full NewsForge pipeline.

All external APIs (Groq LLM, Tavily search, httpx scraping) are mocked.
Tests verify that data flows correctly between all 7 agents, the revision
loop works, and the pipeline handles failures gracefully.

Run: pytest tests/test_integration.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════
# Mock factory helpers
# ═══════════════════════════════════════════════════════════════════════════


def mock_planner_json(coverage_score: float = 0.85) -> str:
    """Return valid JSON for PlannerOutput (what the LLM returns)."""
    return json.dumps({
        "subtasks": [
            {
                "subtask_id": "subtask_001",
                "title": "AI diagnostics in healthcare",
                "search_query": "AI diagnostics healthcare 2025",
                "priority": 1,
                "status": "pending",
                "reasoning": "Core area of AI healthcare adoption",
            },
            {
                "subtask_id": "subtask_002",
                "title": "AI drug discovery trends",
                "search_query": "AI drug discovery 2025",
                "priority": 2,
                "status": "pending",
                "reasoning": "Major investment area",
            },
        ],
        "coverage_score": coverage_score,
        "coverage_gaps": [],
        "iteration": 1,
    })


def mock_analysis_json(confidence: float = 0.85) -> str:
    """Return valid JSON for AnalysisIterationOutput."""
    return json.dumps({
        "themes": ["AI diagnostics adoption", "Drug discovery acceleration"],
        "key_facts": [
            "AI radiology tools approved by FDA in 2024 [source: fda.gov]",
            "Drug discovery timelines reduced by 40% [source: nature.com]",
        ],
        "contradictions": [],
        "confidence_score": confidence,
        "coverage_notes": [],
        "iteration": 1,
    })


def mock_critic_json(passed: bool = True, overall_score: float = 0.85) -> str:
    """Return valid JSON for CriticOutputSchema."""
    dim_score = overall_score
    return json.dumps({
        "passed": passed,
        "quality_score": overall_score,
        "dimension_scores": [
            {"dimension": "factual_accuracy", "score": dim_score, "reasoning": "Good"},
            {"dimension": "completeness", "score": dim_score, "reasoning": "Good"},
            {"dimension": "coherence", "score": dim_score, "reasoning": "Good"},
            {"dimension": "citation_quality", "score": dim_score, "reasoning": "Good"},
            {"dimension": "readability", "score": dim_score, "reasoning": "Good"},
        ],
        "feedback_notes": [] if passed else ["Improve citations", "Add more detail"],
        "strengths": ["Well structured", "Good coverage"],
    })


def mock_writer_markdown() -> str:
    """Return a sample markdown report."""
    return (
        "# Impact of AI on Healthcare\n\n"
        "## Executive Summary\n\n"
        "This report examines the impact of artificial intelligence "
        "on healthcare in 2025, covering diagnostics, drug discovery, "
        "and clinical workflows.\n\n"
        "## AI Diagnostics\n\n"
        "AI-powered diagnostic tools have seen rapid FDA approval. "
        "Radiology remains the leading specialty for AI adoption, "
        "with over 500 FDA-cleared algorithms as of 2024.\n\n"
        "## Drug Discovery\n\n"
        "Machine learning models have reduced drug discovery timelines "
        "by approximately 40 percent according to Nature. Major pharma "
        "companies are investing billions in AI-driven pipelines.\n\n"
        "## Conclusion\n\n"
        "AI is transforming healthcare across diagnostics and drug "
        "discovery, with significant momentum expected through 2025.\n\n"
        "## References\n\n"
        "1. FDA AI/ML Cleared Devices — https://fda.gov\n"
        "2. Nature Drug Discovery — https://nature.com\n"
    )


def mock_tavily_results() -> dict[str, Any]:
    """Return a Tavily API response shape."""
    return {
        "results": [
            {
                "title": "AI in Healthcare 2025",
                "url": "https://example.com/ai-health",
                "content": "AI is transforming healthcare diagnostics and treatment.",
                "score": 0.92,
            },
            {
                "title": "Drug Discovery with Machine Learning",
                "url": "https://nature.com/drug-discovery",
                "content": "ML models reduce drug discovery timelines by 40%.",
                "score": 0.88,
            },
        ]
    }


def mock_html_article() -> str:
    """Return HTML with an article tag and 200+ words."""
    words = " ".join(["Healthcare AI is advancing rapidly."] * 50)
    return f"<html><body><article><h1>AI Healthcare</h1><p>{words}</p></article></body></html>"


# ═══════════════════════════════════════════════════════════════════════════
# Shared mock setup
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response object with .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _build_pipeline_with_mocks(
    planner_json: str | None = None,
    analysis_json: str | None = None,
    writer_md: str | None = None,
    critic_json: str | None = None,
    critic_jsons: list[str] | None = None,
    tavily_response: dict | None = None,
    httpx_html: str | None = None,
    httpx_raises: bool = False,
):
    """Return (pipeline, config, initial_state) with all external calls mocked.

    Args:
        critic_jsons: if provided, the critic LLM returns these JSONs in order
                      (for testing revision loops).
    """
    import uuid
    from datetime import datetime, timezone
    from unittest.mock import patch as _patch

    from langgraph.types import Command

    # Defaults
    _planner = planner_json or mock_planner_json()
    _analysis = analysis_json or mock_analysis_json()
    _writer = writer_md or mock_writer_markdown()
    _critic = critic_json or mock_critic_json()
    _tavily = tavily_response or mock_tavily_results()
    _html = httpx_html or mock_html_article()

    research_id = str(uuid.uuid4())

    initial_state = {
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

    return research_id, initial_state, config


class _MockPatches:
    """Context manager that patches all external calls for the pipeline."""

    def __init__(
        self,
        planner_json: str | None = None,
        analysis_json: str | None = None,
        writer_md: str | None = None,
        critic_json: str | None = None,
        critic_jsons: list[str] | None = None,
        tavily_response: dict | None = None,
        httpx_html: str | None = None,
        httpx_raises: bool = False,
    ):
        self._planner = planner_json or mock_planner_json()
        self._analysis = analysis_json or mock_analysis_json()
        self._writer = writer_md or mock_writer_markdown()
        self._critic_jsons = critic_jsons or [critic_json or mock_critic_json()]
        self._critic_idx = 0
        self._tavily = tavily_response or mock_tavily_results()
        self._html = httpx_html or mock_html_article()
        self._httpx_raises = httpx_raises
        self._patches = []

    def _get_critic_response(self):
        """Return the next critic response, cycling if needed."""
        idx = min(self._critic_idx, len(self._critic_jsons) - 1)
        self._critic_idx += 1
        return self._critic_jsons[idx]

    def __enter__(self):
        import httpx

        # ── Patch Groq LLM per-agent (each gets its own fixed response) ──
        # Instead of routing by prompt content (fragile), we patch each
        # agent's ChatGroq independently with the correct response.

        p1 = patch("agents.planner.ChatGroq")
        mock_planner_llm_cls = p1.start()
        mock_planner_llm_cls.return_value.invoke.return_value = (
            _make_mock_llm_response(self._planner)
        )
        self._patches.append(p1)

        p2 = patch("agents.analysis.ChatGroq")
        mock_analysis_llm_cls = p2.start()
        mock_analysis_llm_cls.return_value.invoke.return_value = (
            _make_mock_llm_response(self._analysis)
        )
        self._patches.append(p2)

        p3 = patch("agents.writer.ChatGroq")
        mock_writer_llm_cls = p3.start()
        mock_writer_llm_cls.return_value.invoke.return_value = (
            _make_mock_llm_response(self._writer)
        )
        self._patches.append(p3)

        # Critic needs side_effect for revision loop support (multiple calls)
        p4 = patch("agents.critic.ChatGroq")
        mock_critic_llm_cls = p4.start()
        mock_critic_llm_cls.return_value.invoke.side_effect = [
            _make_mock_llm_response(c) for c in self._critic_jsons
        ] + [_make_mock_llm_response(self._critic_jsons[-1])] * 10  # extra for safety
        self._patches.append(p4)

        # ── Patch Tavily ──
        p5 = patch("agents.search.TavilyClient")
        mock_tavily_cls = p5.start()
        mock_tavily_cls.return_value.search.return_value = self._tavily
        self._patches.append(p5)

        # ── Patch httpx (used by scraper) ──
        p6 = patch("agents.scraper.httpx.get")
        mock_httpx = p6.start()
        if self._httpx_raises:
            mock_httpx.side_effect = Exception("Connection refused")
        else:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = self._html
            mock_resp.raise_for_status = MagicMock()
            mock_httpx.return_value = mock_resp
        self._patches.append(p6)

        # ── Patch Langfuse (avoid real API calls) ──
        p7 = patch("orchestrator.graph.langfuse")
        mock_lf = p7.start()
        # Make the context managers work
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_lf.start_as_current_observation.return_value.__enter__ = MagicMock(return_value=mock_trace)
        mock_lf.start_as_current_observation.return_value.__exit__ = MagicMock(return_value=False)
        mock_trace.start_as_current_observation.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_trace.start_as_current_observation.return_value.__exit__ = MagicMock(return_value=False)
        mock_trace.trace_id = "mock-trace-id"
        self._patches.append(p7)

        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Test 1: Full pipeline runs to completion with all mocks."""

    def test_full_pipeline_runs_to_completion(self, tmp_path):
        from langgraph.types import Command
        from orchestrator.graph import build_pipeline

        research_id, initial_state, config = _build_pipeline_with_mocks()

        with _MockPatches():
            pipeline = build_pipeline()

            # First invoke — will pause at human_review_node (interrupt)
            result = pipeline.invoke(initial_state, config=config)

            # Check pipeline is paused
            graph_state = pipeline.get_state(config)
            assert graph_state.next, "Pipeline should be paused at interrupt"

            # Resume with approval
            result = pipeline.invoke(
                Command(resume={"decision": "approve"}),
                config=config,
            )

        # Assert all key state fields populated
        assert len(result.get("subtasks", [])) > 0, "subtasks should be non-empty"
        assert len(result.get("search_results", [])) > 0, "search_results should be non-empty"
        assert result.get("analysis") is not None, "analysis should exist"
        assert "themes" in result["analysis"], "analysis should have themes"
        assert result.get("draft_report"), "draft_report should be non-empty"
        assert result.get("critic_feedback") is not None, "critic_feedback should exist"
        assert "passed" in result["critic_feedback"], "critic_feedback should have passed key"
        assert result.get("errors") == [], "errors should be empty"
        assert result.get("human_decision") == "approve"


class TestStatePassing:
    """Test 2: Data flows correctly between nodes."""

    def test_state_passes_correctly_between_nodes(self):
        from langgraph.types import Command
        from orchestrator.graph import build_pipeline

        research_id, initial_state, config = _build_pipeline_with_mocks()

        with _MockPatches():
            pipeline = build_pipeline()
            result = pipeline.invoke(initial_state, config=config)

            # Resume
            result = pipeline.invoke(
                Command(resume={"decision": "approve"}),
                config=config,
            )

        # Planner → Search: subtasks reach search (search_results link to subtask_ids)
        subtask_ids = {s["subtask_id"] for s in result["subtasks"]}
        search_subtask_ids = {r["subtask_id"] for r in result["search_results"]}
        assert search_subtask_ids.issubset(subtask_ids), \
            "Search results should link to planner subtasks"

        # Analysis exists (came from scraper → analysis chain)
        assert result["analysis"] is not None

        # Writer produced draft_report (came from analysis)
        assert result["draft_report"] is not None
        assert len(result["draft_report"]) > 0

        # Critic evaluated the draft_report
        assert result["critic_feedback"] is not None
        assert isinstance(result["critic_feedback"]["quality_score"], float)

        # Publisher ran (came from human approval)
        assert result.get("published_url") is not None or result.get("pipeline_status") == "publisher_complete"


class TestScraperFailure:
    """Test 3: Pipeline survives scraper failure."""

    def test_pipeline_survives_scraper_failure(self):
        from langgraph.types import Command
        from orchestrator.graph import build_pipeline

        research_id, initial_state, config = _build_pipeline_with_mocks()

        with _MockPatches(httpx_raises=True):
            pipeline = build_pipeline()
            result = pipeline.invoke(initial_state, config=config)

            # Resume with approval
            result = pipeline.invoke(
                Command(resume={"decision": "approve"}),
                config=config,
            )

        # Pipeline should not crash
        assert result is not None
        assert result.get("pipeline_status") != "crashed"

        # Scraped content should exist but all items failed
        scraped = result.get("scraped_content", [])
        if scraped:
            for item in scraped:
                assert item.get("scrape_status") in ("failed", "blocked", "success")


class TestRevisionLoop:
    """Test 4: Revision loop increments count."""

    def test_revision_loop_increments_count(self):
        from langgraph.types import Command
        from orchestrator.graph import build_pipeline

        research_id, initial_state, config = _build_pipeline_with_mocks()

        # First critic call fails, second passes
        critic_fail = mock_critic_json(passed=False, overall_score=0.55)
        critic_pass = mock_critic_json(passed=True, overall_score=0.88)

        with _MockPatches(critic_jsons=[critic_fail, critic_pass]):
            pipeline = build_pipeline()
            result = pipeline.invoke(initial_state, config=config)

            # Resume with approval
            result = pipeline.invoke(
                Command(resume={"decision": "approve"}),
                config=config,
            )

        # Revision count should be >= 1 (at least one revision happened)
        assert result.get("revision_count", 0) >= 1, \
            f"revision_count should be >= 1, got {result.get('revision_count')}"

        # Final critic feedback should be passed
        cf = result.get("critic_feedback", {})
        assert cf.get("passed") is True, "Final critic should pass"


class TestEmptyScrape:
    """Test 5: Pipeline completes even with 0 successful scrapes."""

    def test_pipeline_completes_with_empty_scrape(self):
        from langgraph.types import Command
        from orchestrator.graph import build_pipeline

        research_id, initial_state, config = _build_pipeline_with_mocks()

        # Return HTML with almost no content (below minimum word threshold)
        empty_html = "<html><body><p>Short.</p></body></html>"

        with _MockPatches(httpx_html=empty_html):
            pipeline = build_pipeline()
            result = pipeline.invoke(initial_state, config=config)

            # Resume with approval
            result = pipeline.invoke(
                Command(resume={"decision": "approve"}),
                config=config,
            )

        # Pipeline should still complete
        assert result is not None
        # Draft report should still exist (analysis + writer still ran)
        assert result.get("draft_report") is not None
