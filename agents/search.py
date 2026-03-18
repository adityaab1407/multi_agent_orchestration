"""Search agent that queries Tavily for relevant news and information.

This module contains:

* **SearchResultSchema / SearchAgentOutput** — Pydantic V2 models that
  define the structured contract for search results flowing into the pipeline.
* **SearchAgent** — A simple tool-use agent (no ReAct loop) that executes
  one Tavily search per subtask with retry logic and exponential backoff.

Searches run in parallel via ``asyncio.gather`` + ``asyncio.to_thread``
(the Tavily SDK is synchronous, so each call runs in a thread-pool worker).

# Sequential (old): ~10s for 5 subtasks
# Parallel  (new):  ~2s for 5 subtasks (5x speedup)

The agent is invoked by ``search_node`` in ``orchestrator/graph.py``
but has **zero LangGraph imports** — it is pure agent logic.
"""

import asyncio
import concurrent.futures
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from tavily import TavilyClient

from config.settings import SEARCH_RESULTS_PER_QUERY, TAVILY_API_KEY


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 schemas
# ═══════════════════════════════════════════════════════════════════════════


class SearchResultSchema(BaseModel):
    """A single search result returned by Tavily for one subtask.

    ``result_id`` is globally unique across all subtask results within a
    single pipeline run so downstream agents can reference individual sources.
    """

    result_id: str = Field(
        ...,
        description='Globally unique id, e.g. "result_001"',
    )
    subtask_id: str = Field(
        ...,
        description="Links back to the subtask that generated this result",
    )
    title: str = Field(
        ...,
        description="Article / page title returned by Tavily",
    )
    url: str = Field(
        ...,
        description="Source URL",
    )
    snippet: str = Field(
        ...,
        description="Clean text content / snippet from Tavily",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from Tavily, 0.0 to 1.0",
    )
    source_domain: str = Field(
        ...,
        description='Domain extracted from URL, e.g. "nature.com"',
    )


class SearchAgentOutput(BaseModel):
    """Aggregated output of the SearchAgent across all subtasks.

    ``search_status`` indicates overall health:
    - ``"complete"`` — every subtask returned results.
    - ``"partial"``  — some subtasks failed but at least one succeeded.
    - ``"failed"``   — no subtask returned any results.
    """

    results: list[SearchResultSchema]
    total_results: int
    failed_subtasks: list[str] = Field(
        default_factory=list,
        description="subtask_ids that failed to return results",
    )
    search_status: str = Field(
        ...,
        description='"complete" | "partial" | "failed"',
    )


# ═══════════════════════════════════════════════════════════════════════════
# SearchAgent
# ═══════════════════════════════════════════════════════════════════════════


class SearchAgent:
    """Simple tool-use agent that runs one Tavily search per subtask.

    No reasoning loop — each subtask's ``search_query`` is sent directly to
    the Tavily API.  Retry logic with exponential backoff handles transient
    failures and rate limits.

    Searches run in parallel via ``run_async`` (asyncio.gather).
    The synchronous ``run()`` method delegates to ``run_async()`` so that
    callers (like ``search_node``) need zero changes.
    """

    def __init__(self) -> None:
        """Initialise the SearchAgent with a TavilyClient and tuning params.

        Reads ``TAVILY_API_KEY`` and ``SEARCH_RESULTS_PER_QUERY`` from
        ``config.settings`` (which loads them from ``.env``).
        """
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.max_results: int = SEARCH_RESULTS_PER_QUERY
        self.max_retries: int = 3
        self.retry_delay: float = 1.0  # base delay in seconds

    # ── public API (sync) ──────────────────────────────────────────────

    def run(self, subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Search Tavily for every subtask and return a flat list of results.

        This is the single entry-point called by ``search_node`` in
        ``orchestrator/graph.py``.  Internally it delegates to
        ``run_async()`` for parallel execution.

        Args:
            subtasks: List of subtask dicts, each containing at least
                ``subtask_id`` and ``search_query``.

        Returns:
            A ``list[dict]`` of search results ready to merge into
            ``NewsForgeState["search_results"]``.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an async context (e.g. FastAPI, Jupyter)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.run_async(subtasks))
                return future.result()
        else:
            return asyncio.run(self.run_async(subtasks))

    # ── public API (async) ─────────────────────────────────────────────

    async def run_async(
        self, subtasks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run all subtask searches in parallel via asyncio.gather.

        Each synchronous Tavily call is wrapped with ``asyncio.to_thread``
        so it executes in a thread-pool worker without blocking the loop.

        Args:
            subtasks: List of subtask dicts.

        Returns:
            Flat ``list[dict]`` of search results (same contract as ``run``).
        """
        if not subtasks:
            print("[Search] No subtasks to process.")
            return []

        print(f"[Search] Running {len(subtasks)} searches in parallel...")
        start = time.time()

        # Build per-subtask result offsets so result_ids are globally unique.
        # Each subtask gets up to self.max_results ids starting at its offset.
        offsets = [i * self.max_results for i in range(len(subtasks))]

        tasks = [
            self._search_subtask_async(subtask, offset)
            for subtask, offset in zip(subtasks, offsets)
        ]

        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten, handling per-subtask exceptions gracefully
        all_results: list[SearchResultSchema] = []
        failed_subtask_ids: list[str] = []

        for i, result in enumerate(results_nested):
            subtask_id = subtasks[i].get("subtask_id", "unknown")
            if isinstance(result, Exception):
                print(f"[Search] Subtask {subtask_id} failed: {result}")
                failed_subtask_ids.append(subtask_id)
                continue
            if result:
                all_results.extend(result)
            else:
                failed_subtask_ids.append(subtask_id)

        elapsed = time.time() - start

        # Determine overall status
        if not failed_subtask_ids:
            status = "complete"
        elif all_results:
            status = "partial"
        else:
            status = "failed"

        print(
            f"[Search] Complete — {len(all_results)} results, "
            f"{len(failed_subtask_ids)} failed, {elapsed:.1f}s"
        )

        return [r.model_dump() for r in all_results]

    # ── async wrapper for per-subtask search ───────────────────────────

    async def _search_subtask_async(
        self,
        subtask: dict[str, Any],
        result_offset: int,
    ) -> list[SearchResultSchema]:
        """Run ``_search_subtask`` in a thread so it doesn't block the loop."""
        return await asyncio.to_thread(
            self._search_subtask, subtask, result_offset
        )

    # ── per-subtask search (sync, unchanged) ───────────────────────────

    def _search_subtask(
        self,
        subtask: dict[str, Any],
        result_offset: int,
    ) -> list[SearchResultSchema]:
        """Execute a single Tavily search for one subtask.

        Args:
            subtask: A subtask dict with ``subtask_id`` and ``search_query``.
            result_offset: Running counter so ``result_id`` values are
                globally unique across all subtasks.

        Returns:
            A list of ``SearchResultSchema`` objects, or an empty list on
            failure.
        """
        subtask_id: str = subtask.get("subtask_id", "unknown")
        query: str = subtask.get("search_query", "")

        if not query:
            print(f"[Search] Skipping subtask {subtask_id} — empty query")
            return []

        try:
            print(f"[Search] Querying: {query!r} (subtask {subtask_id})")
            raw_results = self._call_tavily_with_retry(query)
        except Exception as e:
            print(f"[Search] FAILED subtask {subtask_id}: {e}")
            return []

        results: list[SearchResultSchema] = []
        for i, item in enumerate(raw_results):
            url = item.get("url", "")
            parsed = urlparse(url)
            domain = parsed.netloc.removeprefix("www.")

            results.append(
                SearchResultSchema(
                    result_id=f"result_{result_offset + i + 1:03d}",
                    subtask_id=subtask_id,
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("content", ""),
                    relevance_score=min(max(item.get("score", 0.0), 0.0), 1.0),
                    source_domain=domain,
                )
            )

        print(
            f"[Search] Got {len(results)} results for subtask {subtask_id}"
        )
        return results

    # ── Tavily call with retry ─────────────────────────────────────────

    def _call_tavily_with_retry(self, query: str) -> list[dict[str, Any]]:
        """Call Tavily search with exponential backoff retry.

        Retry policy:
            - Rate-limit errors → wait 60 s, then retry.
            - Timeout errors → log and retry with normal backoff.
            - Other errors → log and retry with exponential backoff.
            - After ``max_retries`` failures, the last exception is raised.

        Args:
            query: The search query string.

        Returns:
            A list of raw result dicts from the Tavily response.

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.tavily.search(
                    query=query,
                    max_results=self.max_results,
                )
                return response.get("results", [])

            except Exception as e:
                last_exc = e
                err_msg = str(e).lower()

                if "rate limit" in err_msg:
                    wait = 60
                    print(
                        f"[Search] Rate limit hit (attempt {attempt}/"
                        f"{self.max_retries}). Waiting {wait}s..."
                    )
                    time.sleep(wait)
                elif "timeout" in err_msg:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    print(
                        f"[Search] Timeout (attempt {attempt}/"
                        f"{self.max_retries}). Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    print(
                        f"[Search] Error (attempt {attempt}/"
                        f"{self.max_retries}): {e}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

        raise last_exc  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    mock_subtasks = [
        {
            "subtask_id": "subtask_001",
            "title": "AI diagnostics in radiology",
            "search_query": "AI diagnostics radiology healthcare 2025",
            "priority": 1,
            "status": "pending",
            "reasoning": "Radiology is a leading area for AI adoption",
        },
        {
            "subtask_id": "subtask_002",
            "title": "AI drug discovery",
            "search_query": "AI drug discovery pharmaceutical 2025",
            "priority": 2,
            "status": "pending",
            "reasoning": "Drug discovery is being transformed by AI",
        },
        {
            "subtask_id": "subtask_003",
            "title": "AI ethics in clinical settings",
            "search_query": "AI ethics bias clinical healthcare 2025",
            "priority": 3,
            "status": "pending",
            "reasoning": "Ethical concerns are critical for AI adoption",
        },
    ]

    agent = SearchAgent()

    # Time the parallel run
    t0 = time.time()
    results = agent.run(mock_subtasks)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"Search output — {len(results)} total results in {elapsed:.1f}s:")
    print("=" * 60)
    pprint.pprint(results)
