"""Tests for agents/scraper.py.

Coverage targets:
  - ScraperConfig default values
  - _chunk_text edge cases (empty, short, exact, long, overlap correctness)
  - _clean_html tag/class stripping and content-priority extraction
  - _scrape_with_httpx: success, 403, 429, 404, timeout, paywall, generic error
  - run(): output shape, max_urls cap, relevance sorting, sleep cadence
  - _scrape_url integration: full httpx→Playwright upgrade path
  - ScrapedContentSchema Pydantic V2 validation

All external HTTP calls are mocked — no real network I/O in any test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.scraper import ScraperAgent, ScraperConfig, ScrapedContentSchema


# ═══════════════════════════════════════════════════════════════════════════
# Shared sample data
# ═══════════════════════════════════════════════════════════════════════════

#: 250-word article HTML with nav / footer / script noise around it.
SAMPLE_HTML = """
<html>
<head><title>AI in Healthcare</title></head>
<body>
  <nav>Navigation menu home about contact subscribe</nav>
  <article>
    <h1>AI in Healthcare</h1>
    <p>Artificial intelligence is transforming healthcare in numerous ways.
    Machine learning algorithms can now detect diseases earlier than human
    doctors in many cases. The technology has shown particular promise in
    radiology and pathology where pattern recognition is critical. Recent
    studies have demonstrated accuracy rates exceeding ninety five percent
    for certain diagnostic tasks. Deep learning models trained on millions
    of medical images outperform experienced radiologists at identifying
    tumors in early stages. Natural language processing enables automated
    extraction of patient information from unstructured clinical notes.
    Predictive analytics helps hospitals anticipate patient deterioration
    before it becomes critical. Robotic surgery systems guided by artificial
    intelligence achieve greater precision than unaided human surgeons.
    Electronic health record systems now use machine learning to surface
    relevant patient history during clinical encounters. Drug discovery
    timelines have shrunk from over a decade to just a few years thanks to
    generative models that propose novel molecular candidates. Clinical
    trial matching algorithms connect eligible patients with experimental
    treatments they would otherwise never encounter. Remote patient
    monitoring using wearable devices combined with algorithmic analysis
    catches deterioration early. Hospital readmission prediction models
    allow targeted interventions that keep patients healthier at home.
    Mental health applications provide cognitive behavioural therapy to
    underserved populations who lack access to licensed therapists.
    Genomic medicine leverages machine learning to interpret complex variant
    data and guide personalised treatment strategies for individual patients.
    </p>
  </article>
  <footer>Copyright 2025 NewsForge. All rights reserved.</footer>
  <script>console.log("tracking pixel loaded");</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════


def make_mock_search_result(
    result_id: str = "result_001",
    subtask_id: str = "subtask_001",
    url: str = "https://example.com/article",
    relevance_score: float = 0.9,
) -> dict:
    """Return a minimal SearchResult dict suitable for scraper input."""
    return {
        "result_id": result_id,
        "subtask_id": subtask_id,
        "title": "Test Article",
        "url": url,
        "snippet": "Test snippet text for testing purposes.",
        "relevance_score": relevance_score,
        "source_domain": "example.com",
    }


def make_valid_schema(**overrides) -> ScrapedContentSchema:
    """Return a valid ScrapedContentSchema, optionally overriding any field."""
    defaults: dict = dict(
        result_id="result_001",
        subtask_id="subtask_001",
        url="https://example.com/article",
        title="Test Article",
        raw_text=" ".join(f"word{i}" for i in range(200)),
        chunks=["chunk one text", "chunk two text"],
        word_count=200,
        scrape_method="httpx",
        scrape_status="success",
    )
    defaults.update(overrides)
    return ScrapedContentSchema(**defaults)


def make_http_mock(status_code: int = 200, text: str = SAMPLE_HTML) -> MagicMock:
    """Return a mock httpx Response-like object.

    For non-2xx status codes, ``raise_for_status()`` is configured to raise
    ``httpx.HTTPStatusError`` (with the mock response attached), matching the
    behaviour of a real httpx Response and the new ``_scrape_with_httpx``
    implementation that calls ``response.raise_for_status()``.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.is_success = 200 <= status_code < 300

    if resp.is_success:
        resp.raise_for_status.return_value = None          # no-op for 2xx
    else:
        exc = httpx.HTTPStatusError(
            message=f"HTTP Error {status_code}",
            request=MagicMock(),
            response=resp,
        )
        resp.raise_for_status.side_effect = exc            # raises for 4xx/5xx

    return resp


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def agent() -> ScraperAgent:
    """ScraperAgent with request_delay=0 so tests do not sleep."""
    return ScraperAgent(config=ScraperConfig(request_delay=0.0))


# ═══════════════════════════════════════════════════════════════════════════
# 1. ScraperConfig defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestScraperConfigDefaults:
    """ScraperConfig must initialise every field to its documented default."""

    def test_scraper_config_defaults(self):
        """All fields match the values stated in the class docstring."""
        cfg = ScraperConfig()
        assert cfg.request_timeout == 15
        assert cfg.playwright_timeout == 30
        assert cfg.min_content_words == 100
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 100
        assert cfg.max_urls == 10
        assert cfg.request_delay == 1.0
        assert cfg.user_agent == "NewsForge-Research-Bot/1.0"

    def test_scraper_config_custom_values(self):
        """Custom constructor arguments should override every default."""
        cfg = ScraperConfig(chunk_size=500, max_urls=3, request_delay=0.25)
        assert cfg.chunk_size == 500
        assert cfg.max_urls == 3
        assert cfg.request_delay == 0.25


# ═══════════════════════════════════════════════════════════════════════════
# 2-3. _chunk_text
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkText:
    """_chunk_text must produce correct word-level, overlapping chunks."""

    def test_chunk_text_basic_produces_multiple_chunks(self, agent):
        """A 2000-word text should produce more than one chunk."""
        text = " ".join(f"word{i}" for i in range(2000))
        chunks = agent._chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_each_chunk_respects_chunk_size(self, agent):
        """Every chunk must contain no more than chunk_size words."""
        text = " ".join(f"word{i}" for i in range(2000))
        for chunk in agent._chunk_text(text):
            assert len(chunk.split()) <= agent.config.chunk_size

    def test_chunk_text_overlap_correctness(self, agent):
        """The tail of chunk N must equal the head of chunk N+1 for overlap words."""
        text = " ".join(f"word{i}" for i in range(2000))
        chunks = agent._chunk_text(text)
        overlap = agent.config.chunk_overlap
        for i in range(len(chunks) - 1):
            tail = chunks[i].split()[-overlap:]
            head = chunks[i + 1].split()[:overlap]
            assert tail == head, (
                f"Overlap mismatch between chunk {i} and {i + 1}: "
                f"tail={tail[:3]}... head={head[:3]}..."
            )

    def test_chunk_text_short_returns_single_element_list(self, agent):
        """Text shorter than chunk_size should be returned as a single-element list."""
        text = " ".join(f"word{i}" for i in range(50))
        result = agent._chunk_text(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_text_short_element_equals_original(self, agent):
        """The single element returned for short text must equal the original string."""
        text = "hello world this is a short piece of text"
        result = agent._chunk_text(text)
        assert result[0] == text

    def test_chunk_text_empty_returns_empty_list(self, agent):
        """Empty string input must return []."""
        assert agent._chunk_text("") == []

    def test_chunk_text_exact_chunk_size_not_split(self, agent):
        """Text of exactly chunk_size words must not be split."""
        text = " ".join(f"word{i}" for i in range(agent.config.chunk_size))
        result = agent._chunk_text(text)
        assert len(result) == 1

    def test_chunk_text_one_over_chunk_size_splits(self, agent):
        """Text of chunk_size + 1 words must produce at least two chunks."""
        text = " ".join(f"word{i}" for i in range(agent.config.chunk_size + 1))
        result = agent._chunk_text(text)
        assert len(result) >= 2

    def test_chunk_text_custom_chunk_size(self):
        """Chunk size override via ScraperConfig should be respected."""
        agent = ScraperAgent(config=ScraperConfig(chunk_size=50, chunk_overlap=5))
        text = " ".join(f"word{i}" for i in range(200))
        chunks = agent._chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 50


# ═══════════════════════════════════════════════════════════════════════════
# 4-5. _clean_html
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanHtml:
    """_clean_html must strip boilerplate and extract main body text."""

    def test_clean_html_removes_nav_text(self, agent):
        """Navigation text must not appear in cleaned output."""
        text = agent._clean_html(SAMPLE_HTML)
        assert "Navigation menu" not in text

    def test_clean_html_removes_footer_text(self, agent):
        """Footer text must not appear in cleaned output."""
        text = agent._clean_html(SAMPLE_HTML)
        assert "Copyright 2025" not in text

    def test_clean_html_removes_script_content(self, agent):
        """Script content must not appear in cleaned output."""
        text = agent._clean_html(SAMPLE_HTML)
        assert "console.log" not in text

    def test_clean_html_finds_article(self, agent):
        """Content inside <article> must be present in cleaned output."""
        html = """
        <html><body>
          <nav>Nav noise to be removed from the final output</nav>
          <article>
            <h1>Main Article Heading</h1>
            <p>This is the primary article body text that should be extracted.
            It contains enough words to exceed the fifty word minimum threshold
            required by the content detection fallback chain in the scraper agent.
            Additional filler sentences ensure the word count comfortably passes
            the check so that the article container is reliably selected.</p>
          </article>
          <aside>Sidebar content should be removed</aside>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Main Article Heading" in text
        assert "primary article body text" in text
        assert "Nav noise" not in text
        assert "Sidebar content" not in text

    def test_clean_html_prefers_article_over_main(self, agent):
        """<article> must take priority over <main> when both are present."""
        html = """
        <html><body>
          <main><p>Main section text that should not appear because article wins
          the content priority contest when both containers are present in the
          document and the article element has more than fifty words of content
          to satisfy the minimum threshold check in the fallback chain.</p></main>
          <article><p>Article section text that wins priority over main element
          because article appears first in the candidate fallback chain and has
          more than fifty words of body text ensuring reliable container selection
          during the content extraction phase of the html cleaning pipeline.</p></article>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Article section text" in text

    def test_clean_html_falls_back_to_main(self, agent):
        """When no <article> is present, <main> content must be extracted."""
        html = """
        <html><body>
          <nav>Nav noise that should be stripped from the final output</nav>
          <main><p>Main section content here with enough words to comfortably
          pass the fifty word minimum threshold used by the expanded content
          candidate selection chain so that the main element is reliably chosen
          during the fallback logic in the html cleaning pipeline of the scraper
          agent when no article container is found inside the document.</p></main>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Main section content" in text
        assert "Nav noise" not in text

    def test_clean_html_falls_back_to_body(self, agent):
        """When neither <article> nor <main> exist, <body> should be used."""
        html = """
        <html><body>
          <div><p>Body fallback content here with enough words to exceed the
          fifty word minimum threshold so that the body element is selected
          as the content container during the fallback chain evaluation. This
          paragraph adds enough filler text to reliably trigger body selection
          when no article or main element is present in the document markup.</p></div>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Body fallback content" in text

    def test_clean_html_removes_ad_class_elements(self, agent):
        """Elements with class containing 'ad' must be stripped."""
        html = """
        <html><body>
          <div class="ad-unit sponsored">Buy now! Advertisement content here.</div>
          <article><p>Real article content lives here with sufficient word count
          to satisfy the fifty word minimum threshold for container selection.
          This paragraph adds enough extra text so that the article element
          is chosen as the content candidate during the fallback chain check
          and the ad-unit div is properly removed by noise class filtering.</p></article>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Real article content" in text
        assert "Advertisement content" not in text

    def test_clean_html_removes_cookie_class_elements(self, agent):
        """Elements with class containing 'cookie' must be stripped."""
        html = """
        <html><body>
          <div class="cookie-banner">We use cookies to track you across the web.</div>
          <article><p>Article content here with enough words to comfortably
          pass the fifty word minimum threshold for the content candidate
          selection chain so that this article element is reliably chosen and
          the cookie banner div is correctly stripped by the noise class
          filtering step in the html cleaning pipeline of the scraper
          agent system.</p></article>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Article content here" in text
        assert "We use cookies" not in text

    def test_clean_html_removes_banner_and_popup_classes(self, agent):
        """Elements with class containing 'banner' or 'popup' must be stripped."""
        html = """
        <html><body>
          <div class="top-banner">Site-wide announcement text visible at the top.</div>
          <div class="popup-overlay">Newsletter signup modal for email collection.</div>
          <article><p>Core content text with enough words to exceed the fifty
          word minimum threshold required by the fallback chain so that the
          article element is selected as the content container and the banner
          and popup divs are correctly removed by the noise class filtering
          step during html cleaning in the scraper agent pipeline.</p></article>
        </body></html>
        """
        text = agent._clean_html(html)
        assert "Core content text" in text
        assert "Site-wide announcement" not in text
        assert "Newsletter signup" not in text

    def test_clean_html_empty_input_returns_empty_string(self, agent):
        """Empty string must return empty string without error."""
        assert agent._clean_html("") == ""

    def test_clean_html_normalises_whitespace(self, agent):
        """Cleaned output must not contain consecutive whitespace characters."""
        import re
        text = agent._clean_html(SAMPLE_HTML)
        assert text  # non-empty
        assert not re.search(r"\s{2,}", text), "Found consecutive whitespace in output"

    def test_clean_html_returns_string(self, agent):
        """_clean_html must always return a str, never None."""
        result = agent._clean_html(SAMPLE_HTML)
        assert isinstance(result, str)

    def test_clean_html_article_content_present(self, agent):
        """Known article words from SAMPLE_HTML must survive cleaning."""
        text = agent._clean_html(SAMPLE_HTML)
        assert "Artificial intelligence" in text or "artificial intelligence" in text


# ═══════════════════════════════════════════════════════════════════════════
# 6-9. _scrape_with_httpx
# ═══════════════════════════════════════════════════════════════════════════


class TestScrapeWithHttpx:
    """_scrape_with_httpx must map HTTP outcomes to the correct status strings."""

    def test_scrape_with_httpx_success(self, agent):
        """200 response with HTML body must return (html, 'success')."""
        agent._http.get = MagicMock(return_value=make_http_mock(200, SAMPLE_HTML))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert status == "success"
        assert len(html) > 0
        assert "AI in Healthcare" in html

    def test_scrape_with_httpx_blocked_403(self, agent):
        """HTTP 403 must return ('', 'blocked')."""
        agent._http.get = MagicMock(return_value=make_http_mock(403, ""))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert html == ""
        assert status == "blocked"

    def test_scrape_with_httpx_blocked_429(self, agent):
        """HTTP 429 (rate-limit) must return ('', 'blocked')."""
        agent._http.get = MagicMock(return_value=make_http_mock(429, ""))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert html == ""
        assert status == "blocked"

    def test_scrape_with_httpx_not_found(self, agent):
        """HTTP 404 must return ('', 'not_found')."""
        agent._http.get = MagicMock(return_value=make_http_mock(404, ""))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert html == ""
        assert status == "not_found"

    def test_scrape_with_httpx_timeout(self, agent):
        """httpx.TimeoutException must return ('', 'timeout')."""
        agent._http.get = MagicMock(side_effect=httpx.TimeoutException("timed out"))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert html == ""
        assert status == "timeout"

    def test_scrape_with_httpx_paywall_phrase_returns_success(self, agent):
        """Paywall detection moved to _scrape_url — _scrape_with_httpx returns 'success'
        even when the HTML contains paywall-like phrases. The caller (_scrape_url)
        evaluates paywall after cleaning using stricter phrase matching.
        """
        paywall_html = "<html><body><p>Please subscribe to read the full article.</p></body></html>"
        agent._http.get = MagicMock(return_value=make_http_mock(200, paywall_html))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert status == "success"
        assert html == paywall_html

    def test_scrape_with_httpx_nav_signin_does_not_trigger_paywall(self, agent):
        """Navigation 'sign in' links must NOT be treated as paywalls — paywall
        detection now happens post-cleaning in _scrape_url, not in _scrape_with_httpx.
        """
        nav_html = "<html><body><nav><a href='/login'>Sign in</a></nav><p>Article.</p></body></html>"
        agent._http.get = MagicMock(return_value=make_http_mock(200, nav_html))
        _, status = agent._scrape_with_httpx("https://example.com")
        assert status == "success"

    def test_scrape_with_httpx_returns_full_html_on_success(self, agent):
        """_scrape_with_httpx must return the raw HTML string unchanged on 2xx."""
        agent._http.get = MagicMock(return_value=make_http_mock(200, SAMPLE_HTML))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert status == "success"
        assert html == SAMPLE_HTML

    def test_scrape_with_httpx_2xx_always_success(self, agent):
        """Any 2xx response must return 'success' regardless of body content."""
        for body in [
            "subscribe to read the full article",
            "sign in to read more",
            "create a free account to continue",
        ]:
            html_body = f"<html><body><p>{body}</p></body></html>"
            agent._http.get = MagicMock(return_value=make_http_mock(200, html_body))
            _, status = agent._scrape_with_httpx("https://example.com")
            assert status == "success", f"Expected 'success' for body: {body!r}"

    def test_scrape_with_httpx_non_2xx_non_special(self, agent):
        """5xx and other unhandled HTTP codes must return ('', 'http_error_{code}').

        The new implementation returns a specific status string like
        'http_error_500' (via the HTTPStatusError catch branch) rather than
        the generic 'failed', so callers can distinguish HTTP-level errors
        from network/connection failures.
        """
        agent._http.get = MagicMock(return_value=make_http_mock(500, "Internal Server Error"))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert html == ""
        assert status == "http_error_500"

    def test_scrape_with_httpx_generic_exception_returns_failed(self, agent):
        """Any unexpected exception from httpx.get must return ('', 'failed')."""
        agent._http.get = MagicMock(side_effect=ConnectionError("network unreachable"))
        html, status = agent._scrape_with_httpx("https://example.com")
        assert html == ""
        assert status == "failed"

    def test_scrape_with_httpx_does_not_raise(self, agent):
        """_scrape_with_httpx must never propagate an exception to the caller."""
        agent._http.get = MagicMock(side_effect=MemoryError("oom"))
        try:
            agent._scrape_with_httpx("https://example.com")
        except Exception as exc:
            pytest.fail(f"_scrape_with_httpx raised unexpectedly: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# 10-12. run()
# ═══════════════════════════════════════════════════════════════════════════


class TestRunMethod:
    """run() must return correct dicts, respect max_urls, and sort by score."""

    # Required keys every output dict must carry
    REQUIRED_KEYS = frozenset(
        {"result_id", "subtask_id", "url", "title", "raw_text",
         "chunks", "word_count", "scrape_method", "scrape_status"}
    )

    def _valid_schema_for(self, result: dict) -> ScrapedContentSchema:
        return make_valid_schema(
            result_id=result["result_id"],
            url=result["url"],
        )

    def test_run_returns_list_of_dicts(self, agent):
        """run() must return a list of dicts, one per URL."""
        results = [
            make_mock_search_result(f"result_00{i}", url=f"https://example.com/{i}")
            for i in range(3)
        ]
        with patch.object(agent, "_scrape_url", side_effect=lambda r: self._valid_schema_for(r)):
            with patch("agents.scraper.time.sleep"):
                output = agent.run(results)

        assert isinstance(output, list)
        assert len(output) == 3
        for item in output:
            assert isinstance(item, dict)

    def test_run_dicts_contain_required_keys(self, agent):
        """Every output dict must contain all required keys."""
        results = [make_mock_search_result()]
        with patch.object(agent, "_scrape_url", return_value=make_valid_schema()):
            output = agent.run(results)

        missing = self.REQUIRED_KEYS - set(output[0].keys())
        assert not missing, f"Output dict missing keys: {missing}"

    def test_run_limits_to_max_urls(self, agent):
        """run() must process at most config.max_urls URLs."""
        agent.config.max_urls = 5
        results = [
            make_mock_search_result(f"result_{i:03d}", url=f"https://example.com/{i}")
            for i in range(15)
        ]
        call_log: list[str] = []

        def record(r: dict) -> ScrapedContentSchema:
            call_log.append(r["url"])
            return self._valid_schema_for(r)

        with patch.object(agent, "_scrape_url", side_effect=record):
            with patch("agents.scraper.time.sleep"):
                agent.run(results)

        assert len(call_log) == 5

    def test_run_sorts_by_relevance_descending(self, agent):
        """Highest-scored results must be processed first."""
        results = [
            make_mock_search_result("result_low",  url="https://low.com",  relevance_score=0.2),
            make_mock_search_result("result_high", url="https://high.com", relevance_score=0.95),
            make_mock_search_result("result_mid",  url="https://mid.com",  relevance_score=0.6),
        ]
        order: list[str] = []

        def record(r: dict) -> ScrapedContentSchema:
            order.append(r["result_id"])
            return self._valid_schema_for(r)

        with patch.object(agent, "_scrape_url", side_effect=record):
            with patch("agents.scraper.time.sleep"):
                agent.run(results)

        assert order == ["result_high", "result_mid", "result_low"]

    def test_run_sleeps_between_requests_not_after_last(self, agent):
        """time.sleep must be called N-1 times for N URLs, never after the last one."""
        agent.config.request_delay = 0.75
        results = [
            make_mock_search_result(f"result_{i}", url=f"https://example.com/{i}")
            for i in range(4)
        ]
        with patch.object(agent, "_scrape_url", side_effect=lambda r: self._valid_schema_for(r)):
            with patch("agents.scraper.time.sleep") as mock_sleep:
                agent.run(results)

        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(0.75)

    def test_run_empty_input_returns_empty_list(self, agent):
        """run() with [] must return [] without error."""
        assert agent.run([]) == []

    def test_run_returns_model_dump_dicts(self, agent):
        """Output dicts must be plain dicts (model_dump output), not Pydantic models."""
        results = [make_mock_search_result()]
        with patch.object(agent, "_scrape_url", return_value=make_valid_schema()):
            output = agent.run(results)
        assert type(output[0]) is dict  # not a Pydantic model


# ═══════════════════════════════════════════════════════════════════════════
# 13. Resilience — failed URLs must never crash the pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestResilience:
    """The agent must be bullet-proof: every URL resolves to a dict."""

    def test_failed_url_doesnt_crash_run(self, agent):
        """If both httpx and Playwright raise, run() must not propagate the error."""
        agent._http.get = MagicMock(side_effect=RuntimeError("catastrophic failure"))
        with patch.object(agent, "_scrape_with_playwright", return_value=("", "failed")):
            with patch("agents.scraper.time.sleep"):
                try:
                    results = agent.run([make_mock_search_result()])
                except Exception as exc:
                    pytest.fail(f"run() raised unexpectedly: {exc}")

        assert len(results) == 1

    def test_failed_url_produces_failed_status(self, agent):
        """A completely broken URL must yield scrape_status='failed' in the output."""
        agent._http.get = MagicMock(side_effect=RuntimeError("network error"))
        with patch.object(agent, "_scrape_with_playwright", return_value=("", "failed")):
            with patch("agents.scraper.time.sleep"):
                results = agent.run([make_mock_search_result()])

        assert results[0]["scrape_status"] == "failed"

    def test_multiple_failed_urls_all_return_failed(self, agent):
        """Multiple broken URLs must each produce a 'failed' dict, not crash."""
        agent._http.get = MagicMock(side_effect=Exception("boom"))
        with patch.object(agent, "_scrape_with_playwright", return_value=("", "failed")):
            with patch("agents.scraper.time.sleep"):
                results = agent.run([
                    make_mock_search_result(f"result_00{i}", url=f"https://crash.com/{i}")
                    for i in range(3)
                ])

        assert len(results) == 3
        for item in results:
            assert item["scrape_status"] == "failed"

    def test_scrape_url_returns_failed_schema_on_unhandled_exception(self, agent):
        """_scrape_url must never propagate an exception — any error must yield
        scrape_status='failed'.

        MemoryError is a subclass of Exception, so _scrape_with_httpx's inner
        ``except Exception`` absorbs it and returns ("", "failed").  _scrape_url
        then returns that as a normal httpx failure path, so scrape_method is
        "httpx" (the path that was attempted), not "failed".  The critical
        contract is that scrape_status == "failed" and no exception escapes.
        """
        agent._http.get = MagicMock(side_effect=MemoryError("out of memory"))
        schema = agent._scrape_url(make_mock_search_result())

        assert schema.scrape_status == "failed"
        assert schema.scrape_method == "httpx"   # absorbed by _scrape_with_httpx's except
        assert schema.raw_text == ""
        assert schema.chunks == []
        assert schema.word_count == 0

    def test_scrape_url_failed_schema_preserves_metadata(self, agent):
        """Even a failed schema must carry the original result_id, subtask_id, and url."""
        agent._http.get = MagicMock(side_effect=Exception("error"))
        result = make_mock_search_result(
            result_id="result_999",
            subtask_id="subtask_042",
            url="https://broken.com/page",
        )
        schema = agent._scrape_url(result)

        assert schema.result_id == "result_999"
        assert schema.subtask_id == "subtask_042"
        assert schema.url == "https://broken.com/page"


# ═══════════════════════════════════════════════════════════════════════════
# 14. ScrapedContentSchema Pydantic V2 validation
# ═══════════════════════════════════════════════════════════════════════════


class TestScrapedContentSchema:
    """ScrapedContentSchema must satisfy Pydantic V2 construction and serialisation."""

    REQUIRED_KEYS = {
        "result_id", "subtask_id", "url", "title",
        "raw_text", "chunks", "word_count", "scrape_method", "scrape_status",
    }

    def test_model_dump_returns_all_required_keys(self):
        """model_dump() must contain exactly the nine documented keys."""
        data = make_valid_schema().model_dump()
        assert self.REQUIRED_KEYS == set(data.keys())

    def test_model_dump_values_match_constructor_args(self):
        """model_dump() values must faithfully reflect what was passed in."""
        schema = ScrapedContentSchema(
            result_id="r001",
            subtask_id="s001",
            url="https://example.com",
            title="My Title",
            raw_text="hello world",
            chunks=["hello world"],
            word_count=2,
            scrape_method="httpx",
            scrape_status="success",
        )
        data = schema.model_dump()

        assert data["result_id"] == "r001"
        assert data["subtask_id"] == "s001"
        assert data["url"] == "https://example.com"
        assert data["title"] == "My Title"
        assert data["raw_text"] == "hello world"
        assert data["chunks"] == ["hello world"]
        assert data["word_count"] == 2
        assert data["scrape_method"] == "httpx"
        assert data["scrape_status"] == "success"

    def test_missing_required_field_raises_validation_error(self):
        """Omitting a required field must raise pydantic.ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ScrapedContentSchema(
                # result_id intentionally omitted
                subtask_id="s001",
                url="https://example.com",
                title="Title",
                raw_text="text",
                chunks=[],
                word_count=0,
                scrape_method="httpx",
                scrape_status="success",
            )

    def test_model_validate_from_dict(self):
        """ScrapedContentSchema.model_validate(dict) must construct a valid instance."""
        data = {
            "result_id": "r001",
            "subtask_id": "s001",
            "url": "https://example.com",
            "title": "Title",
            "raw_text": "some text",
            "chunks": ["some text"],
            "word_count": 2,
            "scrape_method": "playwright",
            "scrape_status": "success",
        }
        schema = ScrapedContentSchema.model_validate(data)
        assert schema.result_id == "r001"
        assert schema.scrape_method == "playwright"

    def test_chunks_is_list_of_strings(self):
        """chunks field must be a list where every element is a str."""
        schema = make_valid_schema(chunks=["a", "b", "c"])
        assert isinstance(schema.chunks, list)
        assert all(isinstance(c, str) for c in schema.chunks)

    def test_empty_chunks_accepted(self):
        """An empty chunks list must be valid (e.g. for failed scrapes)."""
        schema = make_valid_schema(chunks=[], word_count=0, raw_text="")
        assert schema.chunks == []

    def test_scrape_method_stored_correctly(self):
        """scrape_method must store whatever string is passed — no enum coercion."""
        for method in ("httpx", "playwright", "failed"):
            schema = make_valid_schema(scrape_method=method)
            assert schema.scrape_method == method

    def test_scrape_status_stored_correctly(self):
        """scrape_status must store whatever string is passed."""
        for status in ("success", "failed", "blocked", "paywall", "too_short"):
            schema = make_valid_schema(scrape_status=status)
            assert schema.scrape_status == status


# ═══════════════════════════════════════════════════════════════════════════
# Integration-style: full _scrape_url decision paths
# ═══════════════════════════════════════════════════════════════════════════


class TestScrapeUrlDecisionPaths:
    """Verify the httpx→Playwright orchestration logic inside _scrape_url."""

    def test_success_path_uses_httpx(self, agent):
        """When httpx returns rich content, method must be 'httpx'."""
        agent._http.get = MagicMock(return_value=make_http_mock(200, SAMPLE_HTML))
        schema = agent._scrape_url(make_mock_search_result())

        assert schema.scrape_status == "success"
        assert schema.scrape_method == "httpx"
        assert schema.word_count >= agent.config.min_content_words

    def test_success_path_produces_chunks(self, agent):
        """A successful scrape must produce at least one chunk."""
        agent._http.get = MagicMock(return_value=make_http_mock(200, SAMPLE_HTML))
        schema = agent._scrape_url(make_mock_search_result())
        assert len(schema.chunks) >= 1

    def test_success_path_copies_metadata(self, agent):
        """result_id, subtask_id, url, and title must be copied faithfully."""
        agent._http.get = MagicMock(return_value=make_http_mock(200, SAMPLE_HTML))
        result = make_mock_search_result(
            result_id="result_042",
            subtask_id="subtask_007",
            url="https://specific.com/page",
        )
        result["title"] = "Specific Title"
        schema = agent._scrape_url(result)

        assert schema.result_id == "result_042"
        assert schema.subtask_id == "subtask_007"
        assert schema.url == "https://specific.com/page"
        assert schema.title == "Specific Title"

    def test_paywall_returns_paywall_status(self, agent):
        """Paywall-detected pages must yield scrape_status='paywall'.

        Paywall is now detected post-cleaning in _scrape_url using stricter
        phrases. word_count must be < 100 AND a strict phrase must be present.
        """
        # Content must be > 50 words (so _clean_html's 50-word threshold is
        # satisfied) but < 100 words total (so the paywall word_count guard
        # fires). The phrase "subscribe to read the full" must be present.
        paywall_html = (
            "<html><body><p>Please subscribe to read the full article and all "
            "exclusive content on our website today. To access our premium "
            "journalism and in-depth investigative reporting you will need an "
            "active subscription plan. Our reporting spans technology health "
            "science and business. Join thousands of loyal readers who trust "
            "our daily news reporting every single day. Your subscription "
            "directly supports independent journalism.</p></body></html>"
        )
        agent._http.get = MagicMock(return_value=make_http_mock(200, paywall_html))
        schema = agent._scrape_url(make_mock_search_result())
        assert schema.scrape_status == "paywall"
        assert schema.scrape_method == "httpx"

    def test_blocked_escalates_to_playwright(self, agent):
        """HTTP 403 must trigger a Playwright attempt."""
        agent._http.get = MagicMock(return_value=make_http_mock(403, ""))
        with patch.object(
            agent, "_scrape_with_playwright", return_value=(SAMPLE_HTML, "success")
        ) as mock_pw:
            schema = agent._scrape_url(make_mock_search_result())

        mock_pw.assert_called_once()
        assert schema.scrape_method == "playwright"
        assert schema.scrape_status == "success"

    def test_blocked_playwright_also_fails_returns_blocked(self, agent):
        """If both httpx (403) and Playwright fail, status must be 'blocked'."""
        agent._http.get = MagicMock(return_value=make_http_mock(403, ""))
        with patch.object(agent, "_scrape_with_playwright", return_value=("", "failed")):
            schema = agent._scrape_url(make_mock_search_result())

        assert schema.scrape_status == "blocked"

    def test_thin_content_triggers_playwright_upgrade(self, agent):
        """When httpx content is below min_content_words, Playwright must be tried."""
        thin_html = "<html><body><article><p>Very short text.</p></article></body></html>"
        agent._http.get = MagicMock(return_value=make_http_mock(200, thin_html))
        with patch.object(
            agent, "_scrape_with_playwright", return_value=(SAMPLE_HTML, "success")
        ) as mock_pw:
            schema = agent._scrape_url(make_mock_search_result())

        mock_pw.assert_called_once()
        assert schema.scrape_method == "playwright"

    def test_thin_content_both_methods_returns_too_short(self, agent):
        """If both methods return thin content, status must be 'too_short'."""
        thin_html = "<html><body><article><p>Very short text only here.</p></article></body></html>"
        agent._http.get = MagicMock(return_value=make_http_mock(200, thin_html))
        with patch.object(agent, "_scrape_with_playwright", return_value=(thin_html, "success")):
            schema = agent._scrape_url(make_mock_search_result())

        assert schema.scrape_status == "too_short"

    def test_thin_content_playwright_richer_uses_playwright_text(self, agent):
        """When Playwright produces more words, the richer text must be used."""
        thin_html = "<html><body><article><p>Short.</p></article></body></html>"
        agent._http.get = MagicMock(return_value=make_http_mock(200, thin_html))
        with patch.object(
            agent, "_scrape_with_playwright", return_value=(SAMPLE_HTML, "success")
        ):
            schema = agent._scrape_url(make_mock_search_result())

        # SAMPLE_HTML has 200+ words — should now exceed min_content_words
        assert schema.word_count >= agent.config.min_content_words
        assert schema.scrape_method == "playwright"

    def test_not_found_returns_failed_no_playwright_attempt(self, agent):
        """HTTP 404 must return 'failed' immediately without trying Playwright."""
        agent._http.get = MagicMock(return_value=make_http_mock(404, ""))
        with patch.object(agent, "_scrape_with_playwright") as mock_pw:
            schema = agent._scrape_url(make_mock_search_result())

        mock_pw.assert_not_called()
        assert schema.scrape_status == "failed"

    def test_timeout_returns_failed_no_playwright_attempt(self, agent):
        """Timeout must return 'failed' immediately without trying Playwright."""
        agent._http.get = MagicMock(side_effect=httpx.TimeoutException("timed out"))
        with patch.object(agent, "_scrape_with_playwright") as mock_pw:
            schema = agent._scrape_url(make_mock_search_result())

        mock_pw.assert_not_called()
        assert schema.scrape_status == "failed"
