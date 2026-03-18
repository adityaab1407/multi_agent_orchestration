"""Tests for agents/writer.py.

Coverage targets:
  - _format_citations deduplication and numbering
  - _extract_title heading extraction and fallback
  - _extract_executive_summary section extraction
  - run() output shape, word count, markdown fence stripping
  - _build_user_prompt topic inclusion
  - Resilience with empty analysis

All LLM calls are mocked — no real API I/O in any test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.writer import WriterAgent, WriterOutputSchema


# ═══════════════════════════════════════════════════════════════════════════
# Shared mock data
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_MARKDOWN_REPORT = """\
# AI in Healthcare 2025: A Research Report

## Executive Summary
Artificial intelligence is transforming healthcare in 2025 across multiple dimensions.

This report examines the key trends, challenges, and opportunities in healthcare AI.

## Key Findings
- AI systems have demonstrated significant improvements in diagnostic accuracy.
- 47 states introduced healthcare AI bills in 2025.
- Implementation costs range from $50K to $500K per system.

## AI Diagnostic Accuracy
Healthcare AI achieved remarkable results in 2025 [1]. Studies show accuracy rates
exceeding 95% for certain diagnostic tasks.

AI-powered tools have been deployed across radiology, pathology, and primary care
settings with promising outcomes.

## Regulatory Landscape
The regulatory environment for healthcare AI evolved rapidly in 2025 [2]. Multiple
states introduced legislation governing AI use in clinical settings.

Federal agencies are developing new frameworks for evaluating AI-designed compounds.

## Economic Impact
AI implementation costs vary significantly depending on system complexity [3].
Organizations report ROI within 18-24 months of deployment.

The global healthcare AI market continues to expand, with investment projected to
reach $150B by 2026.

## Points of Debate
Some debate exists around cost-effectiveness [2]. While proponents cite 20-40% cost
reductions, critics note that implementation costs can offset savings in the short term.

## Methodology
This report was generated using the NewsForge multi-agent research pipeline, which
employs automated web search, content scraping, and AI-powered analysis.

## Limitations
Analysis limited to publicly available sources. Long-term patient outcome data remains
limited in the current body of research.

## References
1. 700 lives, $100M saved: Healthcare AI ROI — https://www.beckershospitalreview.com/...
2. Healthcare AI Regulation Compliance Guide — https://www.jimersonfirm.com/...
3. Cost of AI in Healthcare 2025 — https://shadhinlab.com/..."""

MOCK_SEARCH_RESULTS = [
    {
        "result_id": "r001",
        "subtask_id": "s001",
        "title": "700 lives, $100M saved: Healthcare AI ROI",
        "url": "https://www.beckershospitalreview.com/...",
        "snippet": "...",
        "relevance_score": 1.0,
        "source_domain": "beckershospitalreview.com",
    },
    {
        "result_id": "r002",
        "subtask_id": "s005",
        "title": "Healthcare AI Regulation Compliance Guide",
        "url": "https://www.jimersonfirm.com/...",
        "snippet": "...",
        "relevance_score": 1.0,
        "source_domain": "jimersonfirm.com",
    },
    {
        "result_id": "r003",
        "subtask_id": "s003",
        "title": "Cost of AI in Healthcare 2025",
        "url": "https://shadhinlab.com/...",
        "snippet": "...",
        "relevance_score": 1.0,
        "source_domain": "shadhinlab.com",
    },
]

MOCK_ANALYSIS = {
    "themes": [
        {
            "theme": "AI Diagnostic Accuracy",
            "supporting_sources": ["https://www.beckershospitalreview.com"],
            "confidence": 0.9,
            "key_facts": ["Healthcare AI saved 700 lives and $100M in 2025"],
        },
    ],
    "contradictions": [],
    "key_facts": ["700 lives saved by healthcare AI in 2025"],
    "confidence_score": 0.80,
    "coverage_gaps": ["Long-term patient outcome data limited"],
    "sources_analysed": 6,
    "iteration": 1,
}

MOCK_SUBTASKS = [
    {
        "subtask_id": "s001",
        "title": "AI Applications",
        "search_query": "AI medical diagnosis 2025",
        "priority": 1,
        "status": "done",
        "reasoning": "Core topic coverage",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_llm_response(content: str) -> MagicMock:
    """Return a mock LLM response object with a .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_agent_with_mock_llm(response_content: str = SAMPLE_MARKDOWN_REPORT) -> WriterAgent:
    """Return a WriterAgent with a mocked LLM that returns the given content."""
    agent = WriterAgent.__new__(WriterAgent)
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = _make_mock_llm_response(response_content)
    return agent


# ═══════════════════════════════════════════════════════════════════════════
# 1. _format_citations — deduplication
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatCitations:
    """_format_citations must produce deduplicated, numbered citation lists."""

    def test_format_citations_deduplicates(self):
        """Same URL appearing twice should only produce one citation."""
        agent = _make_agent_with_mock_llm()
        results = [
            {"url": "https://example.com/article", "title": "Article A"},
            {"url": "https://example.com/article", "title": "Article A Duplicate"},
            {"url": "https://other.com/page", "title": "Article B"},
        ]
        citations = agent._format_citations(results)
        assert len(citations) == 2
        urls_in_citations = [c.split(" — ")[-1] for c in citations]
        assert len(set(urls_in_citations)) == 2

    def test_format_citations_numbered(self):
        """Three unique results should produce '1. ...', '2. ...', '3. ...'."""
        agent = _make_agent_with_mock_llm()
        citations = agent._format_citations(MOCK_SEARCH_RESULTS)
        assert len(citations) == 3
        assert citations[0].startswith("1. ")
        assert citations[1].startswith("2. ")
        assert citations[2].startswith("3. ")

    def test_format_citations_contains_title_and_url(self):
        """Each citation should contain both the title and URL."""
        agent = _make_agent_with_mock_llm()
        citations = agent._format_citations(MOCK_SEARCH_RESULTS)
        for citation, result in zip(citations, MOCK_SEARCH_RESULTS):
            assert result["title"] in citation
            assert result["url"] in citation

    def test_format_citations_empty_input(self):
        """Empty search results should produce empty citation list."""
        agent = _make_agent_with_mock_llm()
        assert agent._format_citations([]) == []


# ═══════════════════════════════════════════════════════════════════════════
# 2. _extract_title
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractTitle:
    """_extract_title must find the first # heading or return fallback."""

    def test_extract_title_finds_heading(self):
        """Markdown with '# My Report' should return 'My Report'."""
        agent = _make_agent_with_mock_llm()
        title = agent._extract_title("# My Report\n\nSome content here.")
        assert title == "My Report"

    def test_extract_title_fallback(self):
        """Markdown with no # heading should return 'Research Report'."""
        agent = _make_agent_with_mock_llm()
        title = agent._extract_title("No heading here, just plain text.\n\nMore text.")
        assert title == "Research Report"

    def test_extract_title_ignores_h2(self):
        """## headings should not be matched as the title."""
        agent = _make_agent_with_mock_llm()
        title = agent._extract_title("## Not A Title\n\nContent.")
        assert title == "Research Report"

    def test_extract_title_from_sample(self):
        """Sample markdown report title should be extracted correctly."""
        agent = _make_agent_with_mock_llm()
        title = agent._extract_title(SAMPLE_MARKDOWN_REPORT)
        assert title == "AI in Healthcare 2025: A Research Report"


# ═══════════════════════════════════════════════════════════════════════════
# 3. _extract_executive_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractExecutiveSummary:
    """_extract_executive_summary must find the first paragraph after the heading."""

    def test_extract_executive_summary(self):
        """Should return the first paragraph after ## Executive Summary."""
        agent = _make_agent_with_mock_llm()
        summary = agent._extract_executive_summary(SAMPLE_MARKDOWN_REPORT)
        assert "Artificial intelligence is transforming healthcare" in summary

    def test_extract_executive_summary_no_section(self):
        """Markdown without Executive Summary should return empty string."""
        agent = _make_agent_with_mock_llm()
        summary = agent._extract_executive_summary("# Report\n\n## Key Findings\nStuff.")
        assert summary == ""

    def test_extract_executive_summary_empty_input(self):
        """Empty markdown should return empty string."""
        agent = _make_agent_with_mock_llm()
        assert agent._extract_executive_summary("") == ""


# ═══════════════════════════════════════════════════════════════════════════
# 4. run() — output schema validation
# ═══════════════════════════════════════════════════════════════════════════


class TestRunMethod:
    """run() must return a dict with all WriterOutputSchema keys."""

    def test_run_returns_valid_schema(self):
        """All keys from WriterOutputSchema must be present in the output dict."""
        agent = _make_agent_with_mock_llm()
        result = agent.run(
            topic="AI in healthcare",
            analysis=MOCK_ANALYSIS,
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        required_keys = {"title", "executive_summary", "full_report",
                         "word_count", "section_count", "citations"}
        assert required_keys == set(result.keys())

    def test_run_word_count_correct(self):
        """word_count must match len(full_report.split())."""
        agent = _make_agent_with_mock_llm()
        result = agent.run(
            topic="AI in healthcare",
            analysis=MOCK_ANALYSIS,
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        assert result["word_count"] == len(result["full_report"].split())

    def test_run_strips_markdown_fences(self):
        """LLM output wrapped in ```markdown ... ``` must be stripped."""
        fenced_report = f"```markdown\n{SAMPLE_MARKDOWN_REPORT}\n```"
        agent = _make_agent_with_mock_llm(fenced_report)
        result = agent.run(
            topic="AI in healthcare",
            analysis=MOCK_ANALYSIS,
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        assert not result["full_report"].startswith("```")
        assert not result["full_report"].endswith("```")

    def test_run_with_empty_analysis(self):
        """Empty analysis dict should not crash — produces a non-empty report."""
        agent = _make_agent_with_mock_llm()
        result = agent.run(
            topic="AI in healthcare",
            analysis={},
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        assert isinstance(result["full_report"], str)
        assert len(result["full_report"]) > 0

    def test_run_section_count(self):
        """section_count must match the number of ## headings in the report."""
        agent = _make_agent_with_mock_llm()
        result = agent.run(
            topic="AI in healthcare",
            analysis=MOCK_ANALYSIS,
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        import re
        expected = len(re.findall(r"^## ", result["full_report"], re.MULTILINE))
        assert result["section_count"] == expected

    def test_run_citations_match_search_results(self):
        """Citations list must have one entry per unique URL in search_results."""
        agent = _make_agent_with_mock_llm()
        result = agent.run(
            topic="AI in healthcare",
            analysis=MOCK_ANALYSIS,
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        unique_urls = {r["url"] for r in MOCK_SEARCH_RESULTS}
        assert len(result["citations"]) == len(unique_urls)

    def test_run_title_extracted(self):
        """Title should be extracted from the markdown heading."""
        agent = _make_agent_with_mock_llm()
        result = agent.run(
            topic="AI in healthcare",
            analysis=MOCK_ANALYSIS,
            search_results=MOCK_SEARCH_RESULTS,
            subtasks=MOCK_SUBTASKS,
        )
        assert result["title"] == "AI in Healthcare 2025: A Research Report"


# ═══════════════════════════════════════════════════════════════════════════
# 5. _build_user_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildUserPrompt:
    """_build_user_prompt must include all key data in the prompt string."""

    def test_build_user_prompt_includes_topic(self):
        """The topic string must appear in the user prompt."""
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            topic="impact of quantum computing on cryptography",
            analysis=MOCK_ANALYSIS,
            citations=["1. Test — https://example.com"],
            subtasks=MOCK_SUBTASKS,
        )
        assert "impact of quantum computing on cryptography" in prompt

    def test_build_user_prompt_includes_citations(self):
        """Citation strings must appear in the user prompt."""
        agent = _make_agent_with_mock_llm()
        citations = ["1. Article A — https://a.com", "2. Article B — https://b.com"]
        prompt = agent._build_user_prompt(
            topic="test topic",
            analysis=MOCK_ANALYSIS,
            citations=citations,
            subtasks=MOCK_SUBTASKS,
        )
        assert "1. Article A — https://a.com" in prompt
        assert "2. Article B — https://b.com" in prompt

    def test_build_user_prompt_includes_subtask_titles(self):
        """Subtask titles must appear in the user prompt."""
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            topic="test",
            analysis=MOCK_ANALYSIS,
            citations=[],
            subtasks=MOCK_SUBTASKS,
        )
        assert "AI Applications" in prompt

    def test_build_user_prompt_includes_confidence_score(self):
        """Confidence score from analysis must appear in the prompt."""
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            topic="test",
            analysis=MOCK_ANALYSIS,
            citations=[],
            subtasks=MOCK_SUBTASKS,
        )
        assert "0.8" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 6. WriterOutputSchema Pydantic V2 validation
# ═══════════════════════════════════════════════════════════════════════════


class TestWriterOutputSchema:
    """WriterOutputSchema must satisfy Pydantic V2 construction and serialisation."""

    def test_model_dump_returns_all_keys(self):
        """model_dump() must contain all six documented keys."""
        schema = WriterOutputSchema(
            title="Test",
            executive_summary="Summary text.",
            full_report="# Test\n\n## Executive Summary\nContent.",
            word_count=5,
            section_count=1,
            citations=["1. Test — https://example.com"],
        )
        data = schema.model_dump()
        expected_keys = {"title", "executive_summary", "full_report",
                         "word_count", "section_count", "citations"}
        assert expected_keys == set(data.keys())

    def test_model_validate_from_dict(self):
        """model_validate should construct a valid instance from a dict."""
        data = {
            "title": "Test",
            "executive_summary": "Summary.",
            "full_report": "# Test Report",
            "word_count": 3,
            "section_count": 0,
            "citations": [],
        }
        schema = WriterOutputSchema.model_validate(data)
        assert schema.title == "Test"
        assert schema.citations == []
