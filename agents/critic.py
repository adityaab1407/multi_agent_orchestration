"""Critic agent that reviews and scores draft reports for quality and accuracy.

Reads:  state["draft_report"]     ->  str
        state["analysis"]         ->  AnalysisOutput dict
        state["subtasks"]         ->  list[Subtask]
        state["revision_count"]   ->  int
Writes: state["critic_feedback"]  ->  CriticFeedback dict
        state["revision_count"]   ->  int (incremented)

Architecture note:
The Critic uses a single LLM call — not a ReAct loop.  It evaluates the
report once against a structured rubric and returns pass/fail with scores.
Unlike Analysis (which refines its own output), the Critic's job is to
produce a verdict and hand revision responsibility to the Writer.  Adding
a loop here would just re-score the same unchanged report.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

# Maximum number of Writer→Critic revision loops allowed.
MAX_REVISIONS: int = 2


@dataclass
class CriticConfig:
    """Tuneable knobs for the CriticAgent."""

    temperature: float = 0.2
    pass_threshold: float = 0.75
    max_revisions: int = MAX_REVISIONS
    scoring_dimensions: list[str] = field(default_factory=lambda: [
        "factual_accuracy",
        "completeness",
        "coherence",
        "citation_quality",
        "readability",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 schemas
# ═══════════════════════════════════════════════════════════════════════════


class DimensionScore(BaseModel):
    """A single quality dimension score from the Critic LLM."""

    dimension: str = Field(..., description="Name of the quality dimension")
    score: float = Field(..., ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    reasoning: str = Field(..., description="Brief explanation for this score")


class CriticOutputSchema(BaseModel):
    """Structured output of the CriticAgent.

    ``passed``:           Whether the report meets the quality threshold.
    ``quality_score``:    Weighted average across all dimensions (0.0-1.0).
    ``dimension_scores``: Per-dimension breakdown with reasoning.
    ``feedback_notes``:   Specific, actionable revision instructions (empty if passed).
    ``strengths``:        What the report does well (always populated).
    """

    passed: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: list[DimensionScore]
    feedback_notes: list[str]
    strengths: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# CriticAgent
# ═══════════════════════════════════════════════════════════════════════════


class CriticAgent:
    """Reviews a draft report against the original analysis and returns structured feedback.

    The Critic scores the report on multiple quality dimensions (factual accuracy,
    completeness, coherence, citation quality, readability), computes a weighted
    average, and decides pass/fail against ``pass_threshold``.

    If the report fails, ``feedback_notes`` contains specific, actionable
    revision instructions that the Writer can incorporate on the next pass.
    """

    def __init__(self, config: CriticConfig | None = None) -> None:
        """Initialise with a ChatGroq LLM instance and CriticConfig.

        Uses a low temperature (0.2) for consistent, deterministic scoring.

        Args:
            config: Optional CriticConfig; sensible defaults are used if omitted.
        """
        self.config = config or CriticConfig()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=self.config.temperature,
        )

    # ── public API ────────────────────────────────────────────────────────

    def run(
        self,
        draft_report: str,
        analysis: dict[str, Any],
        subtasks: list[dict[str, Any]],
        revision_count: int = 0,
    ) -> dict[str, Any]:
        """Review a draft report and return a CriticOutputSchema dict.

        Args:
            draft_report: The full markdown report string from the Writer.
            analysis: AnalysisOutput dict for fact-checking against.
            subtasks: Original Planner subtasks for coverage checking.
            revision_count: How many prior revisions have occurred.

        Returns:
            A ``dict`` with keys matching ``CriticOutputSchema``, plus the
            ``passed`` and ``quality_score`` keys that map directly to
            ``CriticFeedback`` in the state schema.
        """
        print(f"[Critic] Reviewing report (revision {revision_count})...")

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            draft_report, analysis, subtasks, revision_count,
        )

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        output = self._parse_llm_response(response.content)

        # Log verdict
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

    # ── prompt builders ───────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Return the system prompt instructing the LLM to review a research report.

        Defines the scoring rubric, output JSON schema, and grading rules.

        Returns:
            A system prompt string.
        """
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
        """Build the user prompt with the report and analysis data for comparison.

        Args:
            draft_report: The markdown report to review.
            analysis: AnalysisOutput dict to check facts against.
            subtasks: Planner subtasks to check coverage against.
            revision_count: Current revision number for context.

        Returns:
            A user prompt string.
        """
        parts: list[str] = []

        parts.append(f"REVISION NUMBER: {revision_count}")
        parts.append("")

        # Subtask coverage checklist
        parts.append("RESEARCH SUBTASKS (check all are covered):")
        for st in subtasks:
            parts.append(f"  - {st.get('title', 'Untitled')}")
        parts.append("")

        # Analysis themes to verify
        themes = analysis.get("themes", [])
        parts.append("ANALYSIS THEMES (verify each is addressed):")
        for theme in themes:
            if isinstance(theme, dict):
                parts.append(f"  - {theme.get('theme', 'Unknown')}")
            else:
                parts.append(f"  - {theme}")
        parts.append("")

        # Key facts for fact-checking
        key_facts = analysis.get("key_facts", [])
        parts.append("KEY FACTS TO VERIFY IN REPORT:")
        for fact in key_facts[:15]:
            parts.append(f"  - {fact}")
        parts.append("")

        # Contradictions that should be discussed
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

        # The actual report to review
        parts.append("=" * 60)
        parts.append("DRAFT REPORT TO REVIEW:")
        parts.append("=" * 60)
        parts.append(draft_report)
        parts.append("=" * 60)
        parts.append("")
        parts.append("Review this report against the analysis data above and score it.")

        return "\n".join(parts)

    # ── response parsing ──────────────────────────────────────────────────

    def _parse_llm_response(self, response: str) -> CriticOutputSchema:
        """Parse and validate the raw LLM JSON response into a CriticOutputSchema.

        Handles markdown fence stripping and enforces consistency between
        ``quality_score``, ``passed``, and the dimension scores.

        Args:
            response: Raw text content returned by the Groq LLM.

        Returns:
            A validated ``CriticOutputSchema`` instance.

        Raises:
            ValueError: If the response is not valid JSON or fails validation.
        """
        cleaned = response.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Critic LLM returned invalid JSON. "
                f"JSONDecodeError: {exc}. "
                f"Raw response (first 500 chars): {response[:500]}"
            ) from exc

        # Recompute quality_score as mean of dimension scores for consistency
        dim_scores = data.get("dimension_scores", [])
        if dim_scores:
            avg = sum(d.get("score", 0.0) for d in dim_scores) / len(dim_scores)
            data["quality_score"] = round(avg, 3)

        # Enforce passed consistency with threshold (guard against missing key)
        quality = data.get("quality_score", 0.0)
        data["passed"] = quality >= self.config.pass_threshold

        # Ensure feedback_notes is empty when passed
        if data["passed"]:
            data["feedback_notes"] = data.get("feedback_notes", [])[:0]

        try:
            return CriticOutputSchema.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Critic LLM JSON failed Pydantic validation: {exc}. "
                f"Parsed data keys: {list(data.keys())}"
            ) from exc

    # ── fallback ──────────────────────────────────────────────────────────

    def make_fallback_output(self) -> CriticOutputSchema:
        """Return a minimal passing output when the LLM call fails.

        The fallback passes the report through to avoid blocking the pipeline.
        This is a deliberate design choice: if we can't evaluate quality,
        it's better to publish a potentially imperfect report than to crash.

        Returns:
            A ``CriticOutputSchema`` that passes with a note about the failure.
        """
        return CriticOutputSchema(
            passed=True,
            quality_score=0.0,
            dimension_scores=[],
            feedback_notes=[],
            strengths=["Critic evaluation failed — passing report through"],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Standalone smoke-test
# ═══════════════════════════════════════════════════════════════════════════

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
