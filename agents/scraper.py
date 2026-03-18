"""Scraper agent that extracts structured content from web pages.

Reads:  state["search_results"]  ->  list[SearchResult]
Writes: state["scraped_content"] ->  list[ScrapedContent]

Scraping strategy:
1. Try httpx (fast, low overhead).
2. If content too short (<min_content_words) or blocked, fall back to Playwright.
3. Never raises — every URL resolves to a ScrapedContentSchema.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScraperConfig:
    """Tuneable knobs for the ScraperAgent."""

    request_timeout: int = 15
    playwright_timeout: int = 30
    min_content_words: int = 150
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_urls: int = 10
    request_delay: float = 1.0
    user_agent: str = "NewsForge-Research-Bot/1.0"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 schema
# ═══════════════════════════════════════════════════════════════════════════


class ScrapedContentSchema(BaseModel):
    """Structured representation of a scraped web page.

    ``scrape_method`` records which fetcher produced the final content:
    ``"httpx"``, ``"playwright"``, or ``"failed"`` (neither succeeded).

    ``scrape_status`` records the outcome:
    ``"success"`` | ``"failed"`` | ``"blocked"`` | ``"paywall"`` | ``"too_short"``
    """

    result_id: str
    subtask_id: str
    url: str
    title: str
    raw_text: str
    chunks: list[str]
    word_count: int
    scrape_method: str   # "httpx" | "playwright" | "failed"
    scrape_status: str   # "success" | "failed" | "blocked" | "paywall" | "too_short"


# ═══════════════════════════════════════════════════════════════════════════
# ScraperAgent
# ═══════════════════════════════════════════════════════════════════════════

_PAYWALL_MARKERS: list[str] = [
    "subscribe to read",
    "premium content",
    "sign in to continue",
]

_NOISE_TAGS: list[str] = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "iframe", "noscript",
]

_NOISE_CLASSES: list[str] = ["ad", "cookie", "banner", "popup"]


class ScraperAgent:
    """Fetches, cleans, and chunks web pages sourced from SearchResult URLs.

    Scraping flow per URL:
    - ``httpx`` is tried first (fast, low-overhead).
    - If httpx is blocked (403/429), Playwright is attempted immediately.
    - If httpx content is shorter than ``min_content_words``, Playwright is
      attempted as an upgrade; the richer result wins.
    - Any unhandled exception produces a ``"failed"`` schema — the agent
      never propagates exceptions to the caller.
    """

    def __init__(self, config: ScraperConfig | None = None) -> None:
        """Initialise the ScraperAgent with an httpx.Client.

        Playwright is **not** instantiated here; it is initialised lazily
        inside each ``_scrape_with_playwright`` call via ``sync_playwright()``.

        Args:
            config: Optional ``ScraperConfig``; sensible defaults are used
                if omitted.
        """
        self.config = config or ScraperConfig()
        self._http = httpx.Client(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.request_timeout,
            follow_redirects=True,
        )

    # ── public API ────────────────────────────────────────────────────────

    def run(self, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scrape each URL from *search_results* and return ScrapedContent dicts.

        Results are sorted by ``relevance_score`` (descending) and capped at
        ``config.max_urls`` before scraping.  A ``request_delay`` pause is
        inserted between consecutive requests to avoid triggering rate limits.

        Args:
            search_results: List of SearchResult dicts.  Each must contain at
                minimum the keys ``url``, ``result_id``, ``subtask_id``,
                ``title``, and ``relevance_score``.

        Returns:
            A ``list[dict]`` of ScrapedContent records (one per URL), ready
            to merge into ``NewsForgeState["scraped_content"]``.
        """
        sorted_results = sorted(
            search_results,
            key=lambda r: r.get("relevance_score", 0.0),
            reverse=True,
        )
        batch = sorted_results[: self.config.max_urls]
        total = len(batch)

        scraped: list[dict[str, Any]] = []
        for i, result in enumerate(batch):
            url = result.get("url", "")
            print(f"[Scraper] {i + 1}/{total} — {url[:60]}")
            content = self._scrape_url(result)
            scraped.append(content.model_dump())
            if i < total - 1:
                time.sleep(self.config.request_delay)

        success_count = sum(1 for s in scraped if s["scrape_status"] == "success")
        print(
            f"[Scraper] Complete — {len(scraped)} pages scraped "
            f"({success_count} success, {len(scraped) - success_count} non-success)"
        )
        return scraped

    # ── per-URL orchestration ─────────────────────────────────────────────

    def _scrape_url(self, result: dict[str, Any]) -> ScrapedContentSchema:
        """Scrape a single URL with httpx → Playwright fallback.

        Decision tree:
        1. Call ``_scrape_with_httpx``.
        2. If status is ``"paywall"`` → return paywall schema (no fallback).
        3. If status is ``"not_found"`` / ``"timeout"`` / ``"failed"`` →
           return failed schema.
        4. If status is ``"blocked"`` → try Playwright; if Playwright also
           fails return blocked schema.
        5. Clean HTML and count words.
        6. If ``word_count < min_content_words`` and httpx was used →
           try Playwright as a content upgrade; keep the richer result.
        7. If word count is still below the minimum → return ``"too_short"``.
        8. On any unhandled exception → return failed schema.

        Args:
            result: A SearchResult dict with ``url``, ``result_id``,
                ``subtask_id``, and ``title`` keys.

        Returns:
            A fully populated ``ScrapedContentSchema`` — never raises.
        """
        url = result.get("url", "")
        result_id = result.get("result_id", "")
        subtask_id = result.get("subtask_id", "")
        title = result.get("title", "")

        def _make_schema(
            status: str,
            method: str = "failed",
            text: str = "",
            chunks: list[str] | None = None,
        ) -> ScrapedContentSchema:
            """Build a ScrapedContentSchema with computed word_count."""
            return ScrapedContentSchema(
                result_id=result_id,
                subtask_id=subtask_id,
                url=url,
                title=title,
                raw_text=text,
                chunks=chunks if chunks is not None else [],
                word_count=len(text.split()) if text else 0,
                scrape_method=method,
                scrape_status=status,
            )

        try:
            html, http_status = self._scrape_with_httpx(url)
            method = "httpx"

            # ── terminal httpx statuses ───────────────────────────────────
            if http_status == "paywall":
                return _make_schema("paywall", "httpx")

            if http_status in ("not_found", "timeout", "failed"):
                return _make_schema("failed", "httpx")

            if http_status == "blocked":
                # Escalate immediately — bot-blocked pages need a real browser
                html, _ = self._scrape_with_playwright(url)
                if not html:
                    return _make_schema("blocked", "playwright")
                method = "playwright"

            # ── content extraction ────────────────────────────────────────
            text = self._clean_html(html)
            word_count = len(text.split())

            # ── Playwright upgrade for thin httpx content ─────────────────
            if word_count < self.config.min_content_words and method == "httpx":
                pw_html, _ = self._scrape_with_playwright(url)
                if pw_html:
                    pw_text = self._clean_html(pw_html)
                    pw_word_count = len(pw_text.split())
                    if pw_word_count > word_count:
                        text = pw_text
                        word_count = pw_word_count
                        method = "playwright"

            if word_count < self.config.min_content_words:
                return ScrapedContentSchema(
                    result_id=result_id,
                    subtask_id=subtask_id,
                    url=url,
                    title=title,
                    raw_text=text,
                    chunks=self._chunk_text(text),
                    word_count=word_count,
                    scrape_method=method,
                    scrape_status="too_short",
                )

            chunks = self._chunk_text(text)
            return ScrapedContentSchema(
                result_id=result_id,
                subtask_id=subtask_id,
                url=url,
                title=title,
                raw_text=text,
                chunks=chunks,
                word_count=word_count,
                scrape_method=method,
                scrape_status="success",
            )

        except Exception as exc:
            print(f"[Scraper] Unhandled error for {url}: {exc}")
            return _make_schema("failed", "failed")

    # ── httpx fetch ───────────────────────────────────────────────────────

    def _scrape_with_httpx(self, url: str) -> tuple[str, str]:
        """Fetch *url* with httpx and return ``(html, status)``.

        Status values returned:
        - ``"success"``   — 2xx response with an HTML body.
        - ``"blocked"``   — HTTP 403 or 429 (bot/rate-limit wall).
        - ``"not_found"`` — HTTP 404.
        - ``"timeout"``   — ``httpx.TimeoutException`` raised.
        - ``"paywall"``   — Body contains known paywall marker text.
        - ``"failed"``    — Any other non-success condition.

        Args:
            url: The URL to fetch.

        Returns:
            Tuple of ``(raw_html, status_string)``.  ``raw_html`` is an empty
            string for all non-success statuses.
        """
        try:
            response = self._http.get(url)
        except httpx.TimeoutException:
            return ("", "timeout")
        except Exception:
            return ("", "failed")

        if response.status_code in (403, 429):
            return ("", "blocked")
        if response.status_code == 404:
            return ("", "not_found")
        if not response.is_success:
            return ("", "failed")

        html = response.text
        lower_html = html.lower()
        for marker in _PAYWALL_MARKERS:
            if marker in lower_html:
                return (html, "paywall")

        return (html, "success")

    # ── Playwright fetch ──────────────────────────────────────────────────

    def _scrape_with_playwright(self, url: str) -> tuple[str, str]:
        """Fetch *url* using headless Chromium via Playwright.

        Waits for ``"networkidle"`` so JS-rendered content is fully present
        before ``page.content()`` is called.  Playwright is instantiated
        fresh for each call (``sync_playwright`` context manager) to avoid
        stale browser-state issues across long pipeline runs.

        Args:
            url: The URL to render.

        Returns:
            ``(html, "success")`` on success, or ``("", "failed")`` on any
            error (import failure, browser crash, navigation timeout, etc.).
        """
        try:
            from playwright.sync_api import sync_playwright  # noqa: PLC0415

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    context = browser.new_context(user_agent=self.config.user_agent)
                    page = context.new_page()
                    page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=self.config.playwright_timeout * 1_000,
                    )
                    html = page.content()
                finally:
                    browser.close()

            return (html, "success")

        except Exception as exc:
            print(f"[Scraper] Playwright failed for {url}: {exc}")
            return ("", "failed")

    # ── HTML cleaning ─────────────────────────────────────────────────────

    def _clean_html(self, raw_html: str) -> str:
        """Strip boilerplate from *raw_html* and return clean body text.

        Cleaning pipeline:
        1. Parse with ``BeautifulSoup`` (``html.parser``).
        2. Decompose structural noise: ``<script>``, ``<style>``, ``<nav>``,
           ``<header>``, ``<footer>``, ``<aside>``, ``<form>``, ``<iframe>``,
           ``<noscript>``.
        3. Decompose class-based noise elements whose ``class`` attribute
           contains any of: ``ad``, ``cookie``, ``banner``, ``popup``.
        4. Select the highest-priority content container found:
           ``<article>`` → ``<main>`` → ``<div class="content">`` → ``<body>``
           → (whole soup as final fallback).
        5. Extract text, collapse whitespace to single spaces.

        Args:
            raw_html: Raw HTML string to process.

        Returns:
            Clean plain-text string with normalised whitespace.  Returns an
            empty string if *raw_html* is falsy.
        """
        if not raw_html:
            return ""

        soup = BeautifulSoup(raw_html, "html.parser")

        # 1. Remove structural noise tags
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # 2. Remove class-based noise elements
        for element in soup.find_all(class_=True):
            classes = " ".join(element.get("class", [])).lower()
            if any(noise in classes for noise in _NOISE_CLASSES):
                element.decompose()

        # 3. Content priority: article → main → div.content → body → soup
        content_node = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="content")
            or soup.find("body")
            or soup
        )

        text = content_node.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    # ── text chunking ─────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split *text* into overlapping word-level chunks.

        If the total word count is ≤ ``chunk_size``, returns ``[text]``
        unchanged — no splitting is necessary.

        Overlap is achieved by advancing the window start by
        ``chunk_size - chunk_overlap`` words each step, so consecutive chunks
        share ``chunk_overlap`` words at their boundary.

        Args:
            text: The plain text to split.

        Returns:
            A list of chunk strings.  Returns ``[]`` for empty *text* and
            ``[text]`` when *text* is shorter than ``chunk_size`` words.
        """
        if not text:
            return []

        words = text.split()
        if len(words) <= self.config.chunk_size:
            return [text]

        chunks: list[str] = []
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        start = 0

        while start < len(words):
            end = start + self.config.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += step

        return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Standalone smoke-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mock_results = [
        {
            "result_id": "result_001",
            "subtask_id": "subtask_001",
            "title": "AI in Healthcare 2025: Transforming Diagnostics",
            "url": "https://www.statnews.com/2024/12/10/artificial-intelligence-healthcare-radiology-diagnosis/",
            "snippet": "AI is transforming how clinicians diagnose disease...",
            "relevance_score": 1.0,
            "source_domain": "statnews.com",
        },
        {
            "result_id": "result_002",
            "subtask_id": "subtask_001",
            "title": "How AI Is Accelerating Drug Discovery",
            "url": "https://www.nature.com/articles/d41586-024-00027-4",
            "snippet": "Machine learning models are shortening the drug pipeline...",
            "relevance_score": 0.92,
            "source_domain": "nature.com",
        },
        {
            "result_id": "result_003",
            "subtask_id": "subtask_002",
            "title": "AI Ethics in Clinical Decision Support",
            "url": "https://www.healthaffairs.org/doi/10.1377/hlthaff.2023.01351",
            "snippet": "Bias in clinical AI tools remains a key challenge...",
            "relevance_score": 0.85,
            "source_domain": "healthaffairs.org",
        },
    ]

    agent = ScraperAgent()
    results = agent.run(mock_results)

    print("\n" + "=" * 70)
    print(f"Scraper output — {len(results)} pages:")
    print("=" * 70)
    for r in results:
        print(f"\n  URL     : {r['url']}")
        print(f"  Words   : {r['word_count']}")
        print(f"  Method  : {r['scrape_method']}")
        print(f"  Status  : {r['scrape_status']}")
        print(f"  Chunks  : {len(r['chunks'])}")
        preview = r["raw_text"][:200]
        print(f"  Preview : {preview!r}")
