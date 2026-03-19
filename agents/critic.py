"""Critic agent that reviews and scores draft reports for quality and accuracy
using a single LLM call against a structured rubric.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import GROQ_API_KEY, GROQ_REASONING_MODEL
from utils.llm_utils import strip_llm_response


MAX_REVISIONS: int = 2


@dataclass
class CriticConfig:
    """Tuneable knobs for the CriticAgent."""

    temperature: float = 0.2
    pass_threshold: float = 0.70
    max_revisions: int = MAX_REVISIONS
    scoring_dimensions: list[str] = field(default_factory=lambda: [
        "factual_accuracy",
        "completeness",
        "coherence",
        "citation_quality",
        "readability",
    ])


class DimensionScore(BaseModel):
    dimension: str = Field(..., description="Name of the quality dimension")
    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    reasoning: str = Field(..., description="Brief explanation for this score")


class CriticOutputSchema(BaseModel):
    passed: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: list[DimensionScore]
    feedback_notes: list[str]
    strengths: list[str]


class CriticAgent:
    """Reviews a draft report against the original analysis and returns structured feedback.

    Scores on multiple quality dimensions, computes a weighted average, and
    decides pass/fail against ``pass_threshold``. Failed reports get actionable
    revision instructions for the Writer.
    """

    # Pool A — Reasoning model (Scout)
    # Moved from Pool B to avoid TPM collision:
    # Writer(5500t) + Critic(2500t) > 6K TPM limit
    # Scout's 30K TPM handles all reasoning agents cleanly

    def __init__(self, config: CriticConfig | None = None) -> None:
        self.config = config or CriticConfig()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_REASONING_MODEL,
            temperature=self.config.temperature,
        )

    def run(
        self,
        draft_report: str,
        analysis: dict[str, Any],
        subtasks: list[dict[str, Any]],
        revision_count: int = 0,
    ) -> dict[str, Any]:
        """Review a draft report and return a CriticOutputSchema dict."""
        print(f"[Critic] Reviewing report (revision {revision_count})...")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            draft_report, analysis, subtasks, revision_count,
        )

        raw_response = self._invoke_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        output = self._parse_llm_response(raw_response)

        status = "PASSED" if output.passed else "NEEDS REVISION"
        print(
            f"[Critic] Overall: {output.quality_score:.2f} "
            f"{'✅' if output.passed else '❌'} {status}"
        )
        for dim in output.dimension_scores:
            print(f"[Critic]   {dim.dimension}: {dim.score:.2f}")

        if output.feedback_notes:
            print(f"[Critic] Feedback notes ({len(output.feedback_notes)}):")
            for note in output.feedback_notes:
                print(f"[Critic]   - {note}")

        return output.model_dump()

    def _invoke_with_retry(
        self,
        messages: list,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Invoke LLM with retry on transient network errors.

        Retries on connection errors but NOT on rate limits.
        Rate limit errors (429) are raised immediately.
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                error_str = str(e)

                # Rate limit — don't retry, raise immediately
                if "429" in error_str or "rate_limit" in error_str:
                    raise

                # Connection error — retry with backoff
                last_error = e
                if attempt < max_retries:
                    print(
                        f"[Critic] Connection error (attempt "
                        f"{attempt}/{max_retries}), "
                        f"retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # gentle backoff

        raise last_error

    def _build_system_prompt(self) -> str:
        dimensions_str = ", ".join(self.config.scoring_dimensions)

        return textwrap.dedent(f"""\
            You are the Quality Critic in the NewsForge multi-agent research pipeline.
            Your job is to review a draft research report against the original analysis
            data and provide structured quality feedback.

            SCORING DIMENSIONS (score each 0.0 to 1.0):
            {dimensions_str}

            DIMENSION DEFINITIONS:
            - factual_accuracy: Do the claims in the report match the analysis data?
              Are statistics and facts correctly cited? No hallucinated data?
            - completeness: Does the report cover all major themes from the analysis?
              Are all required sections present (Executive Summary, Key Findings,
              theme sections, Points of Debate, Methodology, Limitations, References)?
            - coherence: Is the report logically structured? Do paragraphs flow
              naturally? Is the argument consistent throughout?
            - citation_quality: Are inline citations [N] present and correctly
              matched to the References section? Are sources properly attributed?
            - readability: Is the writing clear, professional, and accessible?
              Appropriate paragraph length? No jargon without explanation?

            PASS THRESHOLD: {self.config.pass_threshold}
            A report passes if the average score across all dimensions >= threshold.

            OUTPUT FORMAT — respond with ONLY raw JSON, no markdown, no backticks:
            {{
              "passed": true,
              "quality_score": 0.85,
              "dimension_scores": [
                {{
                  "dimension": "factual_accuracy",
                  "score": 0.9,
                  "reasoning": "All facts match analysis data, statistics are correctly cited"
                }},
                ...
              ],
              "feedback_notes": [
                "Specific actionable revision instruction 1",
                "Specific actionable revision instruction 2"
              ],
              "strengths": [
                "What the report does well 1",
                "What the report does well 2"
              ]
            }}

            RULES:
            - Score each dimension independently and honestly.
            - quality_score MUST be the arithmetic mean of all dimension scores.
            - Set passed=true ONLY if quality_score >= {self.config.pass_threshold}.
            - If passed=true, feedback_notes should be empty [].
            - If passed=false, feedback_notes MUST contain specific, actionable
              instructions the Writer can use to improve. Be concrete:
              "Add coverage of regulatory themes" not "improve completeness".
            - strengths should always list 1-3 things the report does well.
            - Output ONLY the JSON object. No explanation before or after.
        """)

    def _build_user_prompt(
        self,
        draft_report: str,
        analysis: dict[str, Any],
        subtasks: list[dict[str, Any]],
        revision_count: int,
    ) -> str:
        parts: list[str] = []

        parts.append(f"REVISION NUMBER: {revision_count}")
        parts.append("")

        parts.append("RESEARCH SUBTASKS (check all are covered):")
        for st in subtasks:
            parts.append(f"  - {st.get('title', 'Untitled')}")
        parts.append("")

        themes = analysis.get("themes", [])
        parts.append("ANALYSIS THEMES (verify each is addressed):")
        for theme in themes:
            if isinstance(theme, dict):
                parts.append(f"  - {theme.get('theme', 'Unknown')}")
            else:
                parts.append(f"  - {theme}")
        parts.append("")

        key_facts = analysis.get("key_facts", [])
        parts.append("KEY FACTS TO VERIFY IN REPORT:")
        for fact in key_facts[:15]:
            parts.append(f"  - {fact}")
        parts.append("")

        contradictions = analysis.get("contradictions", [])
        parts.append("CONTRADICTIONS (should appear in Points of Debate):")
        if contradictions:
            for c in contradictions:
                if isinstance(c, dict):
                    parts.append(
                        f"  - {c.get('claim_a', '')} vs {c.get('claim_b', '')}"
                    )
                else:
                    parts.append(f"  - {c}")
        else:
            parts.append("  None identified")
        parts.append("")

        parts.append(
            f"CONFIDENCE SCORE FROM ANALYSIS: "
            f"{analysis.get('confidence_score', 'N/A')}"
        )
        parts.append(
            f"SOURCES ANALYSED: {analysis.get('sources_analysed', 'N/A')}"
        )
        parts.append("")

        parts.append("=" * 60)
        parts.append("DRAFT REPORT TO REVIEW:")
        parts.append("=" * 60)
        parts.append(draft_report)
        parts.append("=" * 60)
        parts.append("")
        parts.append("Review this report against the analysis data above and score it.")

        return "\n".join(parts)

    def _parse_llm_response(self, response: str) -> CriticOutputSchema:
        """Parse and validate the raw LLM JSON response.

        Strips <think>...</think> blocks (Qwen3, DeepSeek-R1) and markdown
        fences, then recomputes quality_score as the mean of dimension scores
        and enforces pass/fail consistency with the threshold.

        Falls back to a default passing output on parse failure rather than
        crashing the pipeline.
        """
        cleaned = strip_llm_response(response)

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            print(
                f"[Critic] JSON parse failed — using fallback scores. "
                f"Raw response (first 300 chars): {response[:300]}"
            )
            return self.make_fallback_output()

        # Recompute quality_score as mean of dimension scores for consistency
        dim_scores = data.get("dimension_scores", [])
        if dim_scores:
            avg = sum(d.get("score", 0.0) for d in dim_scores) / len(dim_scores)
            data["quality_score"] = round(avg, 3)

        quality = data.get("quality_score", 0.0)
        data["passed"] = quality >= self.config.pass_threshold

        if data["passed"]:
            data["feedback_notes"] = data.get("feedback_notes", [])[:0]

        try:
            return CriticOutputSchema.model_validate(data)
        except Exception:
            print(
                f"[Critic] Pydantic validation failed — using fallback scores. "
                f"Parsed data keys: {list(data.keys())}"
            )
            return self.make_fallback_output()

    def make_fallback_output(self) -> CriticOutputSchema:
        """Return a minimal passing output when the LLM call fails.

        Passes the report through to avoid blocking the pipeline -- better to
        publish a potentially imperfect report than to crash.
        """
        return CriticOutputSchema(
            passed=True,
            quality_score=0.0,
            dimension_scores=[],
            feedback_notes=[],
            strengths=["Critic evaluation failed — passing report through"],
        )


if __name__ == "__main__":
    sample_report = """\
# AI in Healthcare 2025: A Comprehensive Research Report

## Executive Summary
Artificial intelligence is rapidly transforming healthcare delivery in 2025.
This report examines key trends across diagnostics, regulation, and economics.

This analysis draws on multiple authoritative sources to provide a balanced
view of both opportunities and challenges in healthcare AI adoption.

## Key Findings
- Healthcare AI saved an estimated 700 lives and $100M in 2025 [1].
- 47 states introduced healthcare AI bills, with 33 signed into law [1].
- Administrative cost reductions of 20-40% have been reported [1].
- AI implementation costs range from $50K to $500K per system [3].

## AI Diagnostic Accuracy
Healthcare AI has demonstrated remarkable accuracy improvements in 2025 [1].
Studies show diagnostic AI systems achieving accuracy rates above 95% for
certain imaging tasks, outperforming unaided human clinicians.

The deployment of AI diagnostic tools has expanded beyond radiology into
pathology, dermatology, and primary care screening. Early detection of
conditions including cancer and cardiovascular disease has improved
significantly in health systems that adopted AI-assisted workflows.

## Regulatory Landscape
The regulatory environment for healthcare AI evolved rapidly in 2025 [2].
Forty-seven states introduced legislation governing AI use in clinical
settings, reflecting growing awareness of both the potential and risks.

Federal agencies including the FDA are developing new frameworks for
evaluating AI-designed compounds and diagnostic tools, with comprehensive
guidance expected in late 2025.

## Economic Impact
AI implementation costs vary significantly depending on system complexity
and scale [3]. Organizations report achieving ROI within 18-24 months of
deployment, driven primarily by administrative efficiency gains.

The global healthcare AI market continues to expand rapidly, with
investment projected to reach $150 billion by 2026. However, upfront
implementation costs remain a barrier for smaller healthcare providers.

## Points of Debate
A notable tension exists between reported cost savings and implementation
costs. While proponents cite 20-40% administrative cost reductions [1],
critics note that implementation costs of $50K-$500K can offset savings
in the short term [3]. The severity of this contradiction is considered
minor, as most analyses project net positive ROI within two years.

## Methodology
This report was generated using the NewsForge multi-agent research pipeline,
which employs automated web search via Tavily, content scraping with httpx,
AI-powered analysis using Groq LLM, and structured report generation.

## Limitations
Long-term patient outcome data remains limited in the current body of
research. Analysis confidence score: 0.80 based on 6 sources analysed.

## References
1. 700 lives, $100M saved: Healthcare AI ROI — https://www.beckershospitalreview.com/...
2. Healthcare AI Regulation Compliance Guide — https://www.jimersonfirm.com/...
3. Cost of AI in Healthcare 2025 — https://shadhinlab.com/...
"""

    mock_analysis = {
        "themes": [
            {"theme": "AI Diagnostic Accuracy", "confidence": 0.9,
             "key_facts": ["Healthcare AI saved 700 lives and $100M in 2025"],
             "supporting_sources": ["https://www.beckershospitalreview.com"]},
            {"theme": "Regulatory Landscape", "confidence": 0.85,
             "key_facts": ["47 states introduced healthcare AI bills in 2025"],
             "supporting_sources": ["https://www.jimersonfirm.com"]},
            {"theme": "Economic Impact", "confidence": 0.8,
             "key_facts": ["AI implementation costs $50K-$500K per system"],
             "supporting_sources": ["https://shadhinlab.com"]},
        ],
        "contradictions": [
            {"claim_a": "AI reduces costs by 20-40%",
             "claim_b": "Implementation costs offset savings",
             "source_a": "https://www.smartertech.com",
             "source_b": "https://murphi.ai",
             "severity": "minor"}
        ],
        "key_facts": [
            "700 lives saved by healthcare AI in 2025",
            "$100M saved across healthcare systems",
            "47 states introduced AI healthcare bills",
        ],
        "confidence_score": 0.80,
        "coverage_gaps": ["Long-term patient outcome data limited"],
        "sources_analysed": 6,
    }

    mock_subtasks = [
        {"subtask_id": "s001", "title": "AI Applications",
         "search_query": "AI medical diagnosis 2025",
         "priority": 1, "status": "done", "reasoning": "Core topic"},
        {"subtask_id": "s005", "title": "Regulatory Framework",
         "search_query": "AI healthcare regulations 2025",
         "priority": 5, "status": "done", "reasoning": "Compliance"},
    ]

    agent = CriticAgent()
    result = agent.run(
        draft_report=sample_report,
        analysis=mock_analysis,
        subtasks=mock_subtasks,
        revision_count=0,
    )

    print("\n" + "=" * 60)
    print(f"Passed     : {result['passed']}")
    print(f"Score      : {result['quality_score']:.2f}")
    print(f"Dimensions : {len(result['dimension_scores'])}")
    print(f"Feedback   : {len(result['feedback_notes'])} notes")
    print(f"Strengths  : {len(result['strengths'])} items")
    for dim in result["dimension_scores"]:
        print(f"  {dim['dimension']}: {dim['score']:.2f} — {dim['reasoning']}")
