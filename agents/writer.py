"""Writer agent that composes a polished research report from analysis output
via a single LLM call (no ReAct loop).
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

from config.settings import GROQ_API_KEY, GROQ_EXECUTION_MODEL
from utils.llm_utils import strip_llm_response


class WriterOutputSchema(BaseModel):
    title: str
    executive_summary: str
    full_report: str
    word_count: int
    section_count: int
    citations: list[str]


class WriterAgent:
    """Composes a structured research report from analysis output via a single LLM call.

    No ReAct loop — the analysis has already been validated by the Analysis Agent.
    """

    # Pool B — Execution model
    # Structured markdown generation does not
    # require reasoning capability.
    # Separate pool = independent rate limits.

    def __init__(self) -> None:
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_EXECUTION_MODEL,
            temperature=0.5,
        )

    def run(
        self,
        topic: str,
        analysis: dict[str, Any],
        search_results: list[dict[str, Any]],
        subtasks: list[dict[str, Any]],
        feedback_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete research report and return a WriterOutputSchema dict.

        When ``feedback_notes`` is provided (revision pass), they are appended
        to the prompt so the LLM knows what to fix.
        """
        if feedback_notes:
            print(f"[Writer] Revising report for: {topic!r} "
                  f"({len(feedback_notes)} feedback notes)")
        else:
            print(f"[Writer] Generating report for: {topic!r}")

        citations = self._format_citations(search_results)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            topic, analysis, citations, subtasks, feedback_notes,
        )

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        markdown = strip_llm_response(response.content)

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

    def _build_system_prompt(self) -> str:
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
        feedback_notes: list[str] | None = None,
    ) -> str:
        """Build the user prompt containing all research data for the LLM.

        When ``feedback_notes`` is provided, a REVISION INSTRUCTIONS section
        is prepended so the LLM knows what to fix.
        """
        parts: list[str] = []

        if feedback_notes:
            parts.append("⚠ REVISION INSTRUCTIONS (from quality review):")
            parts.append("The previous draft was reviewed and needs improvement.")
            parts.append("Address ALL of the following issues:")
            for i, note in enumerate(feedback_notes, 1):
                parts.append(f"  {i}. {note}")
            parts.append("")
            parts.append("Rewrite the COMPLETE report incorporating these fixes.")
            parts.append("Do not just patch — produce a polished final version.")
            parts.append("")

        parts.append(f"RESEARCH TOPIC: {topic}")
        parts.append("")

        parts.append("RESEARCH ANGLES COVERED:")
        for st in subtasks:
            parts.append(
                f"  - {st.get('title', 'Untitled')}: {st.get('search_query', '')}"
            )
        parts.append("")

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
                    parts.append(f"  - {theme}")
        else:
            parts.append("  (No themes identified)")
        parts.append("")

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

        key_facts = analysis.get("key_facts", [])
        parts.append("OVERALL KEY FACTS:")
        for fact in key_facts[:10]:
            parts.append(f"  - {fact}")
        parts.append("")

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

        parts.append("NUMBERED CITATIONS:")
        for citation in citations:
            parts.append(f"  {citation}")
        parts.append("")

        parts.append("Write the complete research report now.")

        return "\n".join(parts)

    def _format_citations(self, search_results: list[dict[str, Any]]) -> list[str]:
        """Build a deduplicated, numbered citation list from search results."""
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

    def _extract_title(self, markdown: str) -> str:
        """Extract the report title from the first ``#`` heading."""
        match = re.search(r"^# (.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Research Report"

    def _extract_executive_summary(self, markdown: str) -> str:
        """Extract the first paragraph after ``## Executive Summary``."""
        pattern = r"## Executive Summary\s*\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, markdown, re.DOTALL)
        if not match:
            return ""

        section_text = match.group(1).strip()
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        return paragraphs[0] if paragraphs else ""

    def _strip_markdown_fences(self, text: str) -> str:
        """Remove accidental markdown code fences that some LLMs add despite instructions."""
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()


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
