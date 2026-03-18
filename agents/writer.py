"""Writer agent that composes a polished research report from analysis output.

Reads:  state["topic"]          ->  str
        state["analysis"]       ->  AnalysisOutput dict
        state["search_results"] ->  list[SearchResult]
        state["subtasks"]       ->  list[Subtask]
Writes: state["draft_report"]   ->  str (full markdown report)

Architecture note:
Writer does NOT use a ReAct loop.  Planner and Analysis use ReAct because
they evaluate their own output quality and refine iteratively.  Writer's
job is formatting already-validated analysis into a report — a single
well-constructed prompt produces a good result without iteration.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_groq import ChatGroq
from pydantic import BaseModel

from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 schema
# ═══════════════════════════════════════════════════════════════════════════


class WriterOutputSchema(BaseModel):
    """Structured output of the WriterAgent.

    ``title``:              Report title extracted from the first ``#`` heading.
    ``executive_summary``:  First paragraph after the ``## Executive Summary`` heading.
    ``full_report``:        Complete markdown report text.
    ``word_count``:         Total word count of ``full_report``.
    ``section_count``:      Number of ``##`` headings in ``full_report``.
    ``citations``:          Numbered reference list (``["1. Title — URL", ...]``).
    """

    title: str
    executive_summary: str
    full_report: str
    word_count: int
    section_count: int
    citations: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# WriterAgent
# ═══════════════════════════════════════════════════════════════════════════


class WriterAgent:
    """Composes a structured research report from analysis output via a single LLM call.

    The agent does not use a ReAct loop — the analysis has already been validated
    by the Analysis Agent.  A single, well-constructed prompt with all themes,
    facts, contradictions, and citations produces a publication-quality report.
    """

    def __init__(self) -> None:
        """Initialise with a ChatGroq LLM instance.

        Reads ``GROQ_API_KEY`` and ``GROQ_MODEL_NAME`` from
        ``config.settings`` (loaded from ``.env``).
        """
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=0.5,
        )

    # ── public API ────────────────────────────────────────────────────────

    def run(
        self,
        topic: str,
        analysis: dict[str, Any],
        search_results: list[dict[str, Any]],
        subtasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a complete research report and return a WriterOutputSchema dict.

        Steps:
        1. Build a deduplicated numbered citation list from ``search_results``.
        2. Construct system + user prompts with all analysis data.
        3. Single LLM call to generate the full markdown report.
        4. Strip accidental backtick fences from LLM output.
        5. Extract title, executive summary, and section count from markdown.

        Args:
            topic: The research topic string.
            analysis: AnalysisOutput dict from the Analysis Agent.
            search_results: List of SearchResult dicts from the Search Agent.
            subtasks: List of Subtask dicts from the Planner Agent.

        Returns:
            A ``dict`` with keys matching ``WriterOutputSchema``.
        """
        print(f"[Writer] Generating report for: {topic!r}")

        # 1. Build citations
        citations = self._format_citations(search_results)

        # 2. Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(topic, analysis, citations, subtasks)

        # 3. Single LLM call
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        # 4. Extract and clean markdown
        markdown = response.content.strip()
        markdown = self._strip_markdown_fences(markdown)

        # 5. Build output schema
        title = self._extract_title(markdown)
        executive_summary = self._extract_executive_summary(markdown)
        word_count = len(markdown.split())
        section_count = len(re.findall(r"^## ", markdown, re.MULTILINE))

        schema = WriterOutputSchema(
            title=title,
            executive_summary=executive_summary,
            full_report=markdown,
            word_count=word_count,
            section_count=section_count,
            citations=citations,
        )

        print(f"[Writer] Done — {word_count} words, {section_count} sections")
        return schema.model_dump()

    # ── prompt builders ───────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Return the system prompt instructing the LLM to write a research report.

        Specifies the exact section order, citation format, tone, and constraints.

        Returns:
            A system prompt string.
        """
        return textwrap.dedent("""\
            You are a professional research report writer for the NewsForge
            multi-agent research pipeline.

            Write a comprehensive, well-structured research report in markdown format.

            REQUIRED SECTIONS IN THIS EXACT ORDER:
            # {Report Title}
            ## Executive Summary
            ## Key Findings
            ## {Theme 1 Name}
            ## {Theme 2 Name}
            ## ... (one section per major theme from the analysis)
            ## Points of Debate
            ## Methodology
            ## Limitations
            ## References

            WRITING RULES:
            - Use [N] inline citations that match the numbered References list.
            - Write in a professional but accessible tone.
            - Include specific data points and statistics from the analysis.
            - Each theme section: 2-3 paragraphs minimum.
            - Executive Summary: 2 paragraphs summarising the entire report.
            - Key Findings: bullet-point list of the most important facts.
            - Points of Debate: cover contradictions and disagreements found.
            - Methodology: explain how the research was conducted (multi-agent
              pipeline with automated web search, scraping, and AI analysis).
            - Limitations: mention coverage gaps and confidence caveats.
            - References: numbered list matching the citations provided.

            OUTPUT RULES:
            - Output ONLY the markdown report.
            - No preamble, no "here is your report", no commentary.
            - Start directly with # Title.
            - Do not wrap the output in backtick fences.
        """)

    def _build_user_prompt(
        self,
        topic: str,
        analysis: dict[str, Any],
        citations: list[str],
        subtasks: list[dict[str, Any]],
    ) -> str:
        """Build the user prompt containing all research data for the LLM.

        Includes the topic, subtask angles, themes with confidence and facts,
        contradictions, key facts, confidence score, coverage gaps, and the
        numbered citation list.

        Args:
            topic: The research topic string.
            analysis: AnalysisOutput dict from the Analysis Agent.
            citations: Numbered citation list from ``_format_citations``.
            subtasks: List of Subtask dicts from the Planner Agent.

        Returns:
            A user prompt string.
        """
        parts: list[str] = []

        # Topic
        parts.append(f"RESEARCH TOPIC: {topic}")
        parts.append("")

        # Research angles
        parts.append("RESEARCH ANGLES COVERED:")
        for st in subtasks:
            parts.append(
                f"  - {st.get('title', 'Untitled')}: {st.get('search_query', '')}"
            )
        parts.append("")

        # Themes
        themes = analysis.get("themes", [])
        parts.append("MAJOR THEMES IDENTIFIED:")
        if themes:
            for theme in themes:
                if isinstance(theme, dict):
                    parts.append(f"  Theme: {theme.get('theme', 'Unknown')}")
                    parts.append(f"  Confidence: {theme.get('confidence', 'N/A')}")
                    facts = theme.get("key_facts", [])
                    if facts:
                        parts.append("  Key Facts:")
                        for fact in facts:
                            parts.append(f"    - {fact}")
                    sources = theme.get("supporting_sources", [])
                    if sources:
                        parts.append(f"  Sources: {', '.join(sources)}")
                    parts.append("")
                else:
                    # Simple string theme (from AnalysisIterationOutput)
                    parts.append(f"  - {theme}")
        else:
            parts.append("  (No themes identified)")
        parts.append("")

        # Contradictions
        contradictions = analysis.get("contradictions", [])
        parts.append("CONTRADICTIONS FOUND:")
        if contradictions:
            for c in contradictions:
                if isinstance(c, dict):
                    parts.append(
                        f"  - {c.get('claim_a', '')} vs {c.get('claim_b', '')} "
                        f"(Source A: {c.get('source_a', '')}, "
                        f"Source B: {c.get('source_b', '')}, "
                        f"Severity: {c.get('severity', 'unknown')})"
                    )
                else:
                    parts.append(f"  - {c}")
        else:
            parts.append("  None identified")
        parts.append("")

        # Key facts
        key_facts = analysis.get("key_facts", [])
        parts.append("OVERALL KEY FACTS:")
        for fact in key_facts[:10]:
            parts.append(f"  - {fact}")
        parts.append("")

        # Metadata
        parts.append(
            f"CONFIDENCE SCORE: {analysis.get('confidence_score', 'N/A')}"
        )
        parts.append(
            f"SOURCES ANALYSED: {analysis.get('sources_analysed', 'N/A')}"
        )
        coverage_gaps = analysis.get("coverage_gaps", [])
        parts.append(
            f"COVERAGE GAPS: {', '.join(coverage_gaps) if coverage_gaps else 'None identified'}"
        )
        parts.append("")

        # Citations
        parts.append("NUMBERED CITATIONS:")
        for citation in citations:
            parts.append(f"  {citation}")
        parts.append("")

        parts.append("Write the complete research report now.")

        return "\n".join(parts)

    # ── citation formatting ───────────────────────────────────────────────

    def _format_citations(self, search_results: list[dict[str, Any]]) -> list[str]:
        """Build a deduplicated, numbered citation list from search results.

        Deduplicates by URL so the same source is not listed twice.

        Args:
            search_results: List of SearchResult dicts.

        Returns:
            A list of strings like ``["1. Title — URL", ...]``.
        """
        seen_urls: set[str] = set()
        citations: list[str] = []

        for result in search_results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = result.get("title", "Untitled")
            citations.append(f"{len(citations) + 1}. {title} — {url}")

        return citations

    # ── markdown extraction helpers ───────────────────────────────────────

    def _extract_title(self, markdown: str) -> str:
        """Extract the report title from the first ``#`` heading in the markdown.

        Looks for a line starting with ``# `` (single hash, not ``##``).

        Args:
            markdown: The full markdown report string.

        Returns:
            The title string, or ``"Research Report"`` as a fallback.
        """
        match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Research Report"

    def _extract_executive_summary(self, markdown: str) -> str:
        """Extract the first paragraph after the ``## Executive Summary`` heading.

        Finds the ``## Executive Summary`` heading and returns the text between
        it and the next ``##`` heading (or end of document), taking the first
        non-empty paragraph.

        Args:
            markdown: The full markdown report string.

        Returns:
            The executive summary paragraph, or ``""`` if not found.
        """
        pattern = r"## Executive Summary\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, markdown, re.DOTALL)
        if not match:
            return ""

        section_text = match.group(1).strip()
        # Return first non-empty paragraph
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        return paragraphs[0] if paragraphs else ""

    def _strip_markdown_fences(self, text: str) -> str:
        """Remove accidental markdown code fences from LLM output.

        Some LLMs wrap their output in backtick blocks despite instructions
        not to.  This strips leading/trailing fences.

        Args:
            text: Raw LLM output text.

        Returns:
            Text with backtick fences removed.
        """
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Standalone smoke-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mock_analysis = {
        "themes": [
            {
                "theme": "AI Diagnostic Accuracy",
                "supporting_sources": [
                    "https://www.beckershospitalreview.com"
                ],
                "confidence": 0.9,
                "key_facts": [
                    "Healthcare AI saved 700 lives and $100M in 2025",
                    "Administrative cost reductions of 20-40% reported",
                ],
            },
            {
                "theme": "Regulatory Landscape",
                "supporting_sources": [
                    "https://www.jimersonfirm.com",
                    "https://www.ama-assn.org",
                ],
                "confidence": 0.85,
                "key_facts": [
                    "47 states introduced healthcare AI bills in 2025",
                    "33 bills signed into law across states",
                ],
            },
            {
                "theme": "Economic Impact",
                "supporting_sources": [
                    "https://murphi.ai",
                    "https://shadhinlab.com",
                ],
                "confidence": 0.8,
                "key_facts": [
                    "AI implementation costs $50K-$500K per system",
                    "ROI typically achieved within 18-24 months",
                ],
            },
        ],
        "contradictions": [
            {
                "claim_a": "AI reduces costs by 20-40%",
                "claim_b": "Implementation costs offset savings",
                "source_a": "https://www.smartertech.com",
                "source_b": "https://murphi.ai",
                "severity": "minor",
            }
        ],
        "key_facts": [
            "700 lives saved by healthcare AI in 2025 (beckershospitalreview.com)",
            "$100M saved across healthcare systems (beckershospitalreview.com)",
            "47 states introduced AI healthcare bills (beckershospitalreview.com)",
            "20-40% administrative cost reduction reported (smartertech.com)",
            "AI investment projected to reach $150B by 2026 (various)",
        ],
        "confidence_score": 0.80,
        "coverage_gaps": ["Long-term patient outcome data limited"],
        "sources_analysed": 6,
        "iteration": 1,
    }

    mock_search_results = [
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

    mock_subtasks = [
        {
            "subtask_id": "s001",
            "title": "AI Applications",
            "search_query": "AI medical diagnosis 2025",
            "priority": 1,
            "status": "done",
            "reasoning": "Core topic coverage",
        },
        {
            "subtask_id": "s005",
            "title": "Regulatory Framework",
            "search_query": "AI healthcare regulations 2025",
            "priority": 5,
            "status": "done",
            "reasoning": "Compliance context",
        },
    ]

    agent = WriterAgent()
    result = agent.run(
        topic="impact of AI on healthcare in 2025",
        analysis=mock_analysis,
        search_results=mock_search_results,
        subtasks=mock_subtasks,
    )

    print("\n" + "=" * 60)
    print(f"Title      : {result['title']}")
    print(f"Words      : {result['word_count']}")
    print(f"Sections   : {result['section_count']}")
    print(f"Citations  : {len(result['citations'])}")
    print("\nEXECUTIVE SUMMARY:")
    print(result["executive_summary"][:300] + "...")
    print("\nFULL REPORT PREVIEW (first 800 chars):")
    print(result["full_report"][:800])
