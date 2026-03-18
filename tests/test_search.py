"""Tests for agents/search.py — SearchAgent async parallel search.

All tests mock the TavilyClient so no real API calls are made.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.search import SearchAgent, SearchAgentOutput, SearchResultSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TAVILY_RESULT = {
    "title": "AI in Healthcare",
    "url": "https://example.com/article",
    "content": "A snippet about AI.",
    "score": 0.85,
}


def _make_tavily_response(n: int = 2) -> dict[str, Any]:
    """Build a fake Tavily response with *n* results."""
    return {"results": [_TAVILY_RESULT.copy() for _ in range(n)]}


def _make_subtasks(n: int) -> list[dict[str, Any]]:
    return [
        {
            "subtask_id": f"subtask_{i + 1:03d}",
            "search_query": f"query {i + 1}",
            "priority": i + 1,
            "status": "pending",
            "reasoning": "test",
        }
        for i in range(n)
    ]


def _make_agent_with_mock_tavily(
    tavily_response: dict[str, Any] | None = None,
    side_effect: Any = None,
) -> SearchAgent:
    """Create a SearchAgent with a mocked TavilyClient."""
    agent = SearchAgent.__new__(SearchAgent)
    agent.max_results = 5
    agent.max_retries = 1
    agent.retry_delay = 0.0
    agent.tavily = MagicMock()
    if side_effect is not None:
        agent.tavily.search.side_effect = side_effect
    else:
        agent.tavily.search.return_value = tavily_response or _make_tavily_response()
    return agent


# ═══════════════════════════════════════════════════════════════════════════
# SearchResultSchema
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchResultSchema:
    def test_model_dump_returns_all_keys(self):
        r = SearchResultSchema(
            result_id="result_001",
            subtask_id="subtask_001",
            title="Test",
            url="https://example.com",
            snippet="text",
            relevance_score=0.5,
            source_domain="example.com",
        )
        d = r.model_dump()
        assert set(d.keys()) == {
            "result_id", "subtask_id", "title", "url",
            "snippet", "relevance_score", "source_domain",
        }

    def test_score_bounds(self):
        with pytest.raises(Exception):
            SearchResultSchema(
                result_id="r", subtask_id="s", title="t", url="u",
                snippet="sn", relevance_score=1.5, source_domain="d",
            )


# ═══════════════════════════════════════════════════════════════════════════
# SearchAgentOutput
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchAgentOutput:
    def test_complete_status(self):
        out = SearchAgentOutput(
            results=[], total_results=0, search_status="complete",
        )
        assert out.search_status == "complete"
        assert out.failed_subtasks == []


# ═══════════════════════════════════════════════════════════════════════════
# Sync run()
# ═══════════════════════════════════════════════════════════════════════════


class TestSyncRun:
    def test_run_returns_list_of_dicts(self):
        agent = _make_agent_with_mock_tavily()
        results = agent.run(_make_subtasks(2))
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_run_empty_subtasks(self):
        agent = _make_agent_with_mock_tavily()
        results = agent.run([])
        assert results == []

    def test_run_result_ids_unique(self):
        agent = _make_agent_with_mock_tavily(_make_tavily_response(3))
        results = agent.run(_make_subtasks(3))
        ids = [r["result_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_run_subtask_ids_linked(self):
        agent = _make_agent_with_mock_tavily(_make_tavily_response(1))
        subtasks = _make_subtasks(2)
        results = agent.run(subtasks)
        linked_ids = {r["subtask_id"] for r in results}
        expected_ids = {s["subtask_id"] for s in subtasks}
        assert linked_ids == expected_ids

    def test_run_handles_all_failures(self):
        agent = _make_agent_with_mock_tavily(side_effect=RuntimeError("boom"))
        results = agent.run(_make_subtasks(2))
        assert results == []

    def test_run_empty_query_skipped(self):
        agent = _make_agent_with_mock_tavily()
        subtasks = [{"subtask_id": "s1", "search_query": ""}]
        results = agent.run(subtasks)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# Async run_async()
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAsync:
    def test_run_async_returns_results(self):
        agent = _make_agent_with_mock_tavily(_make_tavily_response(2))
        results = asyncio.run(agent.run_async(_make_subtasks(3)))
        assert isinstance(results, list)
        assert len(results) == 6  # 3 subtasks * 2 results each

    def test_run_async_handles_one_failure(self):
        """One subtask raises, the other two succeed."""

        def _side_effect(**kwargs):
            query = kwargs.get("query", "")
            if "query 2" in query:
                raise RuntimeError("subtask 2 exploded")
            return _make_tavily_response(2)

        agent = _make_agent_with_mock_tavily(side_effect=_side_effect)
        results = asyncio.run(agent.run_async(_make_subtasks(3)))
        # 2 successful subtasks * 2 results = 4
        assert len(results) == 4
        ids = {r["subtask_id"] for r in results}
        assert "subtask_002" not in ids

    def test_run_async_empty_subtasks(self):
        agent = _make_agent_with_mock_tavily()
        results = asyncio.run(agent.run_async([]))
        assert results == []

    def test_run_async_result_ids_globally_unique(self):
        agent = _make_agent_with_mock_tavily(_make_tavily_response(3))
        results = asyncio.run(agent.run_async(_make_subtasks(4)))
        ids = [r["result_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_run_async_all_fail(self):
        agent = _make_agent_with_mock_tavily(side_effect=RuntimeError("nope"))
        results = asyncio.run(agent.run_async(_make_subtasks(3)))
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# Parallel speedup
# ═══════════════════════════════════════════════════════════════════════════


class TestParallelSpeedup:
    def test_parallel_faster_than_sequential(self):
        """Mock Tavily with 0.1s sleep. Parallel should be at least 2x faster."""

        def _slow_search(**kwargs):
            time.sleep(0.1)
            return _make_tavily_response(1)

        agent = _make_agent_with_mock_tavily(side_effect=_slow_search)
        subtasks = _make_subtasks(5)

        # Sequential timing
        t0 = time.time()
        for subtask in subtasks:
            agent._search_subtask(subtask, 0)
        sequential_time = time.time() - t0

        # Parallel timing
        t0 = time.time()
        asyncio.run(agent.run_async(subtasks))
        parallel_time = time.time() - t0

        # Parallel should be at least 2x faster
        assert parallel_time < sequential_time / 2, (
            f"Parallel ({parallel_time:.2f}s) not 2x faster than "
            f"sequential ({sequential_time:.2f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# _search_subtask (unit)
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchSubtask:
    def test_returns_list_of_schemas(self):
        agent = _make_agent_with_mock_tavily()
        results = agent._search_subtask(
            {"subtask_id": "s1", "search_query": "test"}, 0
        )
        assert all(isinstance(r, SearchResultSchema) for r in results)

    def test_result_offset_applied(self):
        agent = _make_agent_with_mock_tavily(_make_tavily_response(2))
        results = agent._search_subtask(
            {"subtask_id": "s1", "search_query": "test"}, 10
        )
        assert results[0].result_id == "result_011"
        assert results[1].result_id == "result_012"

    def test_domain_extraction(self):
        resp = {"results": [{"title": "T", "url": "https://www.nature.com/articles/1", "content": "c", "score": 0.9}]}
        agent = _make_agent_with_mock_tavily(resp)
        results = agent._search_subtask(
            {"subtask_id": "s1", "search_query": "test"}, 0
        )
        assert results[0].source_domain == "nature.com"

    def test_score_clamped(self):
        resp = {"results": [{"title": "T", "url": "https://x.com", "content": "c", "score": 5.0}]}
        agent = _make_agent_with_mock_tavily(resp)
        results = agent._search_subtask(
            {"subtask_id": "s1", "search_query": "test"}, 0
        )
        assert results[0].relevance_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# _call_tavily_with_retry
# ═══════════════════════════════════════════════════════════════════════════


class TestRetry:
    def test_success_on_first_try(self):
        agent = _make_agent_with_mock_tavily()
        results = agent._call_tavily_with_retry("test query")
        assert len(results) == 2

    def test_raises_after_max_retries(self):
        agent = _make_agent_with_mock_tavily(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError, match="fail"):
            agent._call_tavily_with_retry("test query")
