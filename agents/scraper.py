"""Scraper agent that extracts structured content from web pages.

Uses httpx with Playwright fallback. Never raises -- every URL resolves to a ScrapedContentSchema dict.
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
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel


@dataclass
class ScraperConfig:
    request_timeout: int = 15
    playwright_timeout: int = 30
    min_content_words: int = 100
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_urls: int = 10
    request_delay: float = 1.0
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


class ScrapedContentSchema(BaseModel):
    """``scrape_method``: ``"httpx"`` | ``"playwright"`` | ``"failed"``
    ``scrape_status``: ``"success"`` | ``"failed"`` | ``"blocked"``
                     | ``"paywall"`` | ``"too_short"``
    """

    result_id: str
    subtask_id: str
    url: str
    title: str
    raw_text: str
    chunks: list[str]
    word_count: int
    scrape_method: str
    scrape_status: str


_NOISE_TAGS: list[str] = [
    "script", "style", "nav", "footer",
    "aside", "form", "iframe", "noscript",
]
# NOTE: "header" was removed from _NOISE_TAGS — many sites put article
# headers (title, byline, date) inside <header> tags. The site-wide nav
# is already handled by removing <nav>.

# Matched as whole CSS class tokens, not substrings.
_NOISE_CLASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^ad[-_]?(?:banner|box|slot|unit|wrapper|container|leaderboard)$", re.I),
    re.compile(r"^(?:ad|ads|advert|advertisement)$", re.I),
    re.compile(r"^cookie[-_]?(?:bar|banner|notice|consent|popup)$", re.I),
    re.compile(r"^(?:popup|modal)[-_]?(?:overlay|backdrop|container)?$", re.I),
    re.compile(r"^(?:banner)[-_]?(?:ad|promo|cookie)$", re.I),
]


def _is_noise_element(element: Tag) -> bool:
    """Check if an element's CSS classes indicate it is noise (ads, popups, etc.).

    Matches against individual class tokens using word-boundary-aware patterns,
    NOT substring search — so "has-global-padding" will NOT match "ad".
    """
    classes = element.get("class", [])
    if not classes:
        return False
    for cls_token in classes:
        token_lower = cls_token.lower()
        for pattern in _NOISE_CLASS_PATTERNS:
            if pattern.match(token_lower):
                return True
    return False


class ScraperAgent:
    """Fetches, cleans, and chunks web pages sourced from SearchResult URLs.

    httpx first, Playwright fallback for blocked/thin content.
    Playwright is optional -- if not installed, httpx-only mode is used.
    Never propagates exceptions to the caller.
    """

    def __init__(self, config: ScraperConfig | None = None) -> None:
        self.config = config or ScraperConfig()
        self._http = httpx.Client(
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.request_timeout,
            follow_redirects=True,
        )

    def run(self, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scrape each URL from *search_results* and return ScrapedContent dicts."""
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

    def _scrape_url(self, result: dict[str, Any]) -> ScrapedContentSchema:
        """Scrape a single URL with httpx -> Playwright fallback. Never raises."""
        try:
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

            html, http_status = self._scrape_with_httpx(url)
            method = "httpx"

            if http_status == "blocked":
                html, _ = self._scrape_with_playwright(url)
                if not html:
                    return _make_schema("blocked", "playwright")
                method = "playwright"

            elif http_status != "success":
                return _make_schema("failed", "httpx")

            text = self._clean_html(html)
            word_count = len(text.split())

            _PAYWALL_PHRASES = [
                "subscribe to read the full",
                "this content is for subscribers",
                "premium content. please subscribe",
                "sign in to read more",
                "create a free account to continue",
            ]
            cleaned_lower = text.lower()
            if word_count < 100 and any(p in cleaned_lower for p in _PAYWALL_PHRASES):
                return _make_schema("paywall", method)

            # Playwright upgrade for thin httpx content
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
            print(f"[Scraper] Unhandled error for {result.get('url', '')}: {exc}")
            return ScrapedContentSchema(
                result_id=result.get("result_id", "unknown"),
                subtask_id=result.get("subtask_id", "unknown"),
                url=result.get("url", ""),
                title=result.get("title", ""),
                raw_text="",
                chunks=[],
                word_count=0,
                scrape_method="failed",
                scrape_status="failed",
            )

    def _scrape_with_httpx(self, url: str) -> tuple[str, str]:
        """Fetch *url* with httpx and return ``(html, status)``."""
        try:
            response = self._http.get(url)
            response.raise_for_status()
            return (response.text, "success")

        except httpx.TimeoutException:
            return ("", "timeout")
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (403, 429):
                return ("", "blocked")
            if code == 404:
                return ("", "not_found")
            return ("", f"http_error_{code}")
        except Exception:
            return ("", "failed")

    def _scrape_with_playwright(self, url: str) -> tuple[str, str]:
        """Fetch *url* using headless Chromium via Playwright (optional dep)."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ("", "playwright_unavailable")

        try:
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

    def _clean_html(self, raw_html: str) -> str:
        """Strip boilerplate from *raw_html* and return clean plain text.

        Tries content containers in priority order (article > main > body > soup).
        Falls back to soup.get_text() if nothing has >50 words.
        """
        if not raw_html:
            return ""

        try:
            soup = BeautifulSoup(raw_html, "html.parser")

            for tag in soup.find_all(_NOISE_TAGS):
                tag.decompose()

            for element in soup.find_all(class_=True):
                if _is_noise_element(element):
                    element.decompose()

            candidates: list[tuple[str, Tag | None]] = [
                ("article", soup.find("article")),
                ("main", soup.find("main")),
                ("id=content", soup.find(attrs={"id": "content"})),
                ("id=main-content", soup.find(attrs={"id": "main-content"})),
                ("id=main", soup.find(attrs={"id": "main"})),
                ("div.content-class", soup.find(
                    "div",
                    attrs={
                        "class": lambda c: c and any(
                            k in " ".join(c).lower()
                            for k in [
                                "article", "content", "post", "entry",
                                "body", "story", "text",
                            ]
                        )
                    },
                )),
                ("body", soup.find("body")),
                ("soup", soup),
            ]

            content = ""
            matched_label = ""
            for label, candidate in candidates:
                if candidate is None:
                    continue
                t = candidate.get_text(separator=" ", strip=True)
                wc = len(t.split())
                if wc > 50:
                    content = t
                    matched_label = label
                    break

            # Guaranteed fallback -- never return empty when HTML has content
            if not content:
                fallback = soup.get_text(separator=" ", strip=True)
                content = fallback

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            result = "\n".join(lines) if lines else content

            return result

        except Exception as exc:
            print(f"[Scraper._clean_html] exception: {exc}")
            # Strip tags with regex as absolute last resort
            text = re.sub(r"<[^>]+>", " ", raw_html)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        """Split *text* into overlapping word-level chunks."""
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


if __name__ == "__main__":
    import textwrap

    print("=" * 70)
    print("STEP 1: Raw httpx diagnostic (Harvard URL)")
    print("=" * 70)

    diag_url = "https://news.harvard.edu/gazette/story/2025/03/how-ai-is-transforming-medicine-healthcare/"
    try:
        r = httpx.get(
            diag_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
            follow_redirects=True,
            timeout=15,
        )
        print(f"  Status       : {r.status_code}")
        print(f"  Content-Type : {r.headers.get('content-type')}")
        print(f"  HTML length  : {len(r.text)}")
        print(f"  First 500 chars:\n{textwrap.indent(r.text[:500], '    ')}")
    except Exception as exc:
        print(f"  Raw httpx failed: {exc}")

    print("\n" + "=" * 70)
    print("STEP 2: Full scraper pipeline test (3 URLs)")
    print("=" * 70)

    test_results = [
        {
            "result_id": "result_001",
            "subtask_id": "subtask_001",
            "title": "Harvard AI Medicine",
            "url": "https://news.harvard.edu/gazette/story/2025/03/how-ai-is-transforming-medicine-healthcare/",
            "snippet": "...",
            "relevance_score": 0.84,
            "source_domain": "harvard.edu",
        },
        {
            "result_id": "result_002",
            "subtask_id": "subtask_001",
            "title": "MIT AI Research",
            "url": "https://news.mit.edu/2025/new-ai-system-could-accelerate-clinical-research-0925",
            "snippet": "...",
            "relevance_score": 0.81,
            "source_domain": "mit.edu",
        },
        {
            "result_id": "result_003",
            "subtask_id": "subtask_002",
            "title": "HealthTech AI Trends",
            "url": "https://healthtechmagazine.net/article/2025/01/overview-2025-ai-trends-healthcare",
            "snippet": "...",
            "relevance_score": 1.0,
            "source_domain": "healthtechmagazine.net",
        },
    ]

    agent = ScraperAgent()
    results = agent.run(test_results)

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
