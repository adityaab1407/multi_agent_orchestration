"""Analysis agent that extracts themes, facts, and contradictions from scraped content.

Reads:  state["scraped_content"]  ->  list[ScrapedContent]
        state["subtasks"]         ->  list[Subtask]
Writes: state["analysis"]         ->  AnalysisOutput dict

ReAct pattern (mirrors PlannerAgent exactly):
  ACT     — call Groq LLM with a condensed source corpus
  THINK   — parse + validate the JSON response through Pydantic
  OBSERVE — check confidence_score against threshold; if too low, pass
            coverage_notes into the next iteration as targeted focus areas.

The highest-confidence output across all iterations is returned.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    MAX_REACT_ITERATIONS,
    PLANNING_QUALITY_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AnalysisConfig:
    """Tuneable knobs for the AnalysisAgent."""

    temperature: float = 0.2
    max_iterations: int = MAX_REACT_ITERATIONS
    quality_threshold: float = PLANNING_QUALITY_THRESHOLD
    # llama-3.3-70b-versatile has a 128K context window;
    # 20 000 chars ≈ 5 000 tokens — leaves ample room for the JSON reply.
    max_content_chars: int = 20_000
    max_chars_per_article: int = 2_000
    max_themes: int = 8
    max_key_facts: int = 15


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 schema
# ═══════════════════════════════════════════════════════════════════════════


class AnalysisIterationOutput(BaseModel):
    """Structured output of a single AnalysisAgent ReAct iteration.

    ``confidence_score`` is the LLM's honest self-assessment of how well the
    source corpus covers the research topic.  If it falls below
    ``quality_threshold``, the loop runs another pass using the identified
    ``coverage_notes`` as targeted focus areas for self-correction.
    """

    themes: list[str] = Field(
        ...,
        description="3-8 major themes identified across multiple sources",
    )
    key_facts: list[str] = Field(
        ...,
        description='Up to 15 specific facts, each ending with [source: domain]',
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Conflicting claims found across sources; empty list if none",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-assessed analysis quality, 0.0 to 1.0",
    )
    coverage_notes: list[str] = Field(
        default_factory=list,
        description="Gaps / under-explored angles to address in the next iteration",
    )
    iteration: int = Field(
        ...,
        ge=1,
        description="Which ReAct iteration produced this output",
    )


# ═══════════════════════════════════════════════════════════════════════════
# AnalysisAgent
# ═══════════════════════════════════════════════════════════════════════════


class AnalysisAgent:
    """ReAct-style agent that extracts structured insights from scraped articles.

    The agent loops up to ``max_iterations`` times.  On every iteration it:

    1. **ACT**     — call the Groq LLM with a condensed source corpus.
    2. **THINK**   — parse + Pydantic-validate the JSON response.
    3. **OBSERVE** — compare ``confidence_score`` against ``quality_threshold``.
                     If the score is too low, ``coverage_notes`` from this pass
                     feed into the next prompt so the LLM can self-correct.

    Per-iteration errors are caught and logged; the loop continues to the next
    attempt.  The highest-confidence output across all iterations is returned.
    If every iteration fails, a minimal fallback dict is returned so the
    pipeline never crashes.
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialise with a ChatGroq LLM instance and AnalysisConfig.

        Reads ``GROQ_API_KEY`` and ``GROQ_MODEL_NAME`` from
        ``config.settings`` (loaded from ``.env``).

        Args:
            config: Optional AnalysisConfig; sensible defaults are used if omitted.
        """
        self.config = config or AnalysisConfig()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=self.config.temperature,
        )

    # ── public API ─────────────────────────────────────────────────────────

    def run(
        self,
        scraped_content: list[dict[str, Any]],
        subtasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyse *scraped_content* and return a structured AnalysisOutput dict.

        This is the single entry-point called by ``analysis_node`` in
        ``orchestrator/graph.py``.

        Args:
            scraped_content: List of ScrapedContent dicts from the Scraper Agent.
                Items with ``scrape_status == "success"`` are prioritised;
                failed/blocked/paywall items are excluded from the corpus.
            subtasks: Original Subtask dicts from the Planner, used to add
                research-topic context to the corpus header.

        Returns:
            A ``dict`` with keys: ``themes``, ``key_facts``, ``contradictions``,
            ``confidence_score``, ``coverage_notes``, ``iteration`` — ready to
            be stored as ``NewsForgeState["analysis"]``.
        """
        corpus = self._build_corpus(scraped_content, subtasks)
        usable = sum(
            1 for item in scraped_content
            if item.get("scrape_status") in ("success", "too_short")
        )
        print(
            f"[Analysis] Corpus built — {usable}/{len(scraped_content)} usable "
            f"articles, {len(corpus):,} chars"
        )

        previous_notes: list[str] = []
        best_output: AnalysisIterationOutput | None = None
        system_prompt = self._build_system_prompt()

        for iteration in range(1, self.config.max_iterations + 1):

            # ── ACT: call the LLM ─────────────────────────────────────────
            user_prompt = self._build_user_prompt(corpus, iteration, previous_notes)

            try:
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ])

                # ── THINK: parse + validate ───────────────────────────────
                output = self._parse_llm_response(response.content, iteration)

            except Exception as exc:
                print(f"[Analysis] Iteration {iteration} failed: {exc}")
                continue

            # Track the highest-confidence output across all iterations
            if best_output is None or output.confidence_score > best_output.confidence_score:
                best_output = output

            print(
                f"[Analysis THINK] Iteration {iteration} "
                f"— confidence: {output.confidence_score:.2f} "
                f"| themes: {len(output.themes)} "
                f"| facts: {len(output.key_facts)} "
                f"| contradictions: {len(output.contradictions)}"
            )

            # ── OBSERVE: quality gate ─────────────────────────────────────
            if output.confidence_score >= self.config.quality_threshold:
                print(
                    f"[Analysis OBSERVE] Confidence {output.confidence_score:.2f} "
                    f">= threshold {self.config.quality_threshold:.2f}. Done."
                )
                break

            previous_notes = output.coverage_notes
            print(
                f"[Analysis OBSERVE] Coverage notes: {previous_notes}. "
                f"Refining in next iteration..."
            )

        if best_output is None:
            print("[Analysis] All iterations failed — returning fallback output.")
            best_output = self._make_fallback_output()

        print(
            f"[Analysis] Final — {len(best_output.themes)} themes, "
            f"{len(best_output.key_facts)} facts, "
            f"{len(best_output.contradictions)} contradictions, "
            f"confidence: {best_output.confidence_score:.2f}"
        )
        return best_output.model_dump()

    # ── corpus builder ─────────────────────────────────────────────────────

    def _build_corpus(
        self,
        scraped_content: list[dict[str, Any]],
        subtasks: list[dict[str, Any]],
    ) -> str:
        """Condense scraped articles into a single text corpus for the LLM.

        Ordering strategy:
        1. Filter to ``scrape_status in ("success", "too_short")``.
        2. Sort: ``"success"`` before ``"too_short"`` so richest content leads.
        3. Per article: include title, domain, status/word-count header, then
           up to ``max_chars_per_article`` characters of ``raw_text``.
        4. Stop appending once total corpus exceeds ``max_content_chars`` and
           note how many sources were truncated.

        Args:
            scraped_content: List of ScrapedContent dicts.
            subtasks: List of Subtask dicts (added as a topic-context header).

        Returns:
            A single corpus string ready to embed in a user prompt.
        """
        parts: list[str] = []

        # ── Subtask context header ────────────────────────────────────────
        if subtasks:
            parts.append("RESEARCH SUBTASKS (for context):")
            for st in subtasks:
                parts.append(f"  • {st.get('title', '(untitled)')}")
            parts.append("")

        # ── Filter + sort usable content ──────────────────────────────────
        _PRIORITY: dict[str, int] = {"success": 0, "too_short": 1}
        usable = [
            item for item in scraped_content
            if item.get("scrape_status") in _PRIORITY
        ]
        usable.sort(key=lambda x: _PRIORITY.get(x.get("scrape_status", ""), 99))

        parts.append(
            f"SOURCE DOCUMENTS "
            f"({len(usable)} usable of {len(scraped_content)} scraped):"
        )
        parts.append("=" * 60)

        chars_used = sum(len(p) for p in parts)

        for i, item in enumerate(usable, 1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            try:
                domain = urlparse(url).netloc.removeprefix("www.") or url
            except Exception:
                domain = url

            text = item.get("raw_text", "")[: self.config.max_chars_per_article]
            word_count = item.get("word_count", len(text.split()))

            block = (
                f"\n[Source {i}] {title}\n"
                f"Domain: {domain} | "
                f"Status: {item.get('scrape_status')} | "
                f"Words: {word_count}\n\n"
                f"{text}\n"
                f"{'─' * 40}"
            )

            if chars_used + len(block) > self.config.max_content_chars:
                remaining = len(usable) - i + 1
                parts.append(
                    f"\n[{remaining} more source(s) omitted — "
                    f"corpus limit of {self.config.max_content_chars:,} chars reached]"
                )
                break

            parts.append(block)
            chars_used += len(block)

        if not usable:
            parts.append(
                "\n[WARNING: No successfully scraped content available. "
                "Analysis will be limited to what can be inferred from context.]\n"
            )

        return "\n".join(parts)

    # ── prompt builders ────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Return the system prompt that forces structured JSON output.

        Instructs the LLM to:
        - Extract themes, key facts with source attribution, and contradictions.
        - Self-assess confidence honestly on a 0.0-1.0 scale.
        - List coverage notes (gaps) to guide refinement if score is low.
        - Output **only** valid JSON matching the ``AnalysisIterationOutput`` schema.
        """
        return textwrap.dedent(f"""\
            You are the Analysis Agent in a multi-agent research system called NewsForge.

            YOUR TASK:
            Given a set of scraped research articles (the source corpus), perform a
            deep analysis to extract structured insights for a downstream report writer.

            WHAT TO EXTRACT:
            1. THEMES — Up to {self.config.max_themes} major themes that recur across
               multiple sources. Write each theme as a full, informative sentence
               (not a single word or vague label).
            2. KEY FACTS — Up to {self.config.max_key_facts} specific, verifiable facts.
               Be precise and data-rich (numbers, percentages, named studies).
               Every fact MUST end with [source: domain] e.g. "[source: nature.com]".
            3. CONTRADICTIONS — Any conflicting claims between two or more sources.
               Format: "Source A (domain1) claims X, while Source B (domain2) claims Y."
               Output an empty list if no genuine contradictions exist.
            4. CONFIDENCE SCORE — Your honest 0.0-1.0 assessment of how well the
               sources cover the research topic breadth and depth.
            5. COVERAGE NOTES — Specific angles, dimensions, or data types that are
               absent from the sources. These guide a refinement iteration.

            OUTPUT FORMAT — respond with ONLY raw JSON, no markdown, no backticks, no preamble:
            {{
              "themes": [
                "Theme description as a full informative sentence"
              ],
              "key_facts": [
                "Specific, data-rich fact [source: domain.com]"
              ],
              "contradictions": [
                "Source A (domain1.com) claims X, while Source B (domain2.com) claims Y."
              ],
              "confidence_score": 0.85,
              "coverage_notes": ["Missing cost/economic data", "No regulatory perspective"],
              "iteration": 1
            }}

            RULES:
            - Extract only from the provided sources — never hallucinate facts.
            - Every key_fact MUST have a [source: domain] attribution.
            - confidence_score must honestly reflect source quality and topic coverage.
            - Output ONLY the JSON object. No explanation before or after.
            - If contradictions is empty, output: "contradictions": []
        """)

    def _build_user_prompt(
        self,
        corpus: str,
        iteration: int,
        previous_notes: list[str],
    ) -> str:
        """Build the user message for a given ReAct iteration.

        On the first iteration the prompt contains only the source corpus.
        On subsequent iterations, ``previous_notes`` (coverage gaps from the
        prior pass) are prepended so the LLM can self-correct.

        Args:
            corpus: The condensed source text corpus from ``_build_corpus``.
            iteration: Current iteration number (1-based).
            previous_notes: Coverage gaps identified in the prior iteration
                (empty list on the first pass).

        Returns:
            A prompt string ready to send to the LLM.
        """
        if iteration == 1 or not previous_notes:
            return (
                f"Analyse the following source corpus and extract structured insights.\n\n"
                f"{corpus}\n\n"
                f"Generate analysis for iteration {iteration}."
            )

        notes_text = "\n".join(f"  - {note}" for note in previous_notes)
        return (
            f"The previous iteration identified these coverage gaps that need addressing:\n"
            f"{notes_text}\n\n"
            f"Please focus on these gaps in your revised analysis.\n\n"
            f"{corpus}\n\n"
            f"Generate improved analysis for iteration {iteration}."
        )

    # ── response parsing ───────────────────────────────────────────────────

    def _parse_llm_response(
        self,
        response: str,
        iteration: int,
    ) -> AnalysisIterationOutput:
        """Parse and validate the raw LLM string into an ``AnalysisIterationOutput``.

        Strips markdown fences defensively (the LLM sometimes wraps JSON in
        ``` blocks despite instructions).  The ``iteration`` field is always
        overridden with the caller's value so it is authoritative.

        Args:
            response: The raw text content returned by the Groq LLM.
            iteration: The current iteration number (1-based).

        Returns:
            A validated ``AnalysisIterationOutput`` instance.

        Raises:
            ValueError: If the response is not valid JSON or fails Pydantic
                validation, with a message showing the raw response excerpt.
        """
        cleaned = response.strip()

        # Strip markdown fences if the LLM wraps its output despite instructions
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Analysis LLM returned invalid JSON on iteration {iteration}. "
                f"JSONDecodeError: {exc}. "
                f"Raw response (first 500 chars): {response[:500]}"
            ) from exc

        # Always override iteration to the authoritative value
        data["iteration"] = iteration

        try:
            return AnalysisIterationOutput.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Analysis LLM JSON failed Pydantic validation on "
                f"iteration {iteration}: {exc}. "
                f"Parsed data keys: {list(data.keys())}"
            ) from exc

    # ── fallback ───────────────────────────────────────────────────────────

    def _make_fallback_output(self) -> AnalysisIterationOutput:
        """Return a minimal valid output when every LLM iteration fails.

        Ensures the pipeline never crashes due to Analysis Agent failures.
        Downstream agents (Writer, Critic) will see ``confidence_score=0.0``
        and can handle this gracefully.

        Returns:
            An ``AnalysisIterationOutput`` with empty fields and zero confidence.
        """
        return AnalysisIterationOutput(
            themes=["Analysis unavailable — all LLM iterations failed"],
            key_facts=[],
            contradictions=[],
            confidence_score=0.0,
            coverage_notes=["All LLM iterations failed; rerun the analysis stage"],
            iteration=1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Standalone smoke-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    mock_subtasks = [
        {
            "subtask_id": "subtask_001",
            "title": "AI diagnostics in radiology",
            "search_query": "AI radiology diagnostics 2025",
            "priority": 1,
            "status": "done",
            "reasoning": "Radiology is a leading area for AI adoption",
        },
        {
            "subtask_id": "subtask_002",
            "title": "AI drug discovery",
            "search_query": "AI drug discovery pharmaceutical 2025",
            "priority": 2,
            "status": "done",
            "reasoning": "Drug discovery timelines are being compressed by AI",
        },
        {
            "subtask_id": "subtask_003",
            "title": "AI ethics in clinical settings",
            "search_query": "AI ethics bias clinical healthcare 2025",
            "priority": 3,
            "status": "done",
            "reasoning": "Ethical concerns are critical for safe AI adoption",
        },
    ]

    mock_scraped_content = [
        {
            "result_id": "result_001",
            "subtask_id": "subtask_001",
            "url": "https://www.statnews.com/ai-radiology",
            "title": "AI Matches Radiologists in Detecting Early-Stage Tumors",
            "raw_text": (
                "A landmark study published in Nature Medicine found that a deep learning "
                "model achieved 94.5% accuracy in detecting lung nodules on CT scans, "
                "compared to 88.2% for board-certified radiologists working alone. The model "
                "was trained on 42,000 annotated chest CT scans from four major hospital "
                "systems. Researchers noted the AI reduced false-positive rates by 11%, "
                "which could prevent unnecessary follow-up biopsies. However, the study also "
                "found the model performed significantly worse on patients from underrepresented "
                "demographic groups, with accuracy dropping to 79% for certain populations. "
                "The FDA has now cleared over 500 AI-assisted radiology tools for clinical use "
                "as of early 2025, with adoption accelerating rapidly across major health systems. "
                "Workflow integration remains the primary barrier, with radiologists reporting "
                "alert fatigue when AI flags are too frequent. Several leading academic medical "
                "centers have begun deploying AI triage systems that prioritise urgent findings "
                "in the radiology worklist, reducing time-to-read for critical cases by 40%. "
                "The global AI radiology market is projected to reach $2.8 billion by 2027. "
                "Despite these advances, most radiologists report they view AI as a second "
                "reader rather than a replacement, and the field is converging on a human-plus-AI "
                "collaborative model as the new standard of care."
            ),
            "chunks": [],
            "word_count": 187,
            "scrape_method": "httpx",
            "scrape_status": "success",
        },
        {
            "result_id": "result_002",
            "subtask_id": "subtask_002",
            "url": "https://www.nature.com/drug-discovery-ai",
            "title": "How Generative AI Is Compressing Drug Discovery Timelines",
            "raw_text": (
                "AlphaFold3, released by DeepMind in 2024, has been downloaded and used by "
                "over two million researchers worldwide to predict protein structures with "
                "near-experimental accuracy. Pharmaceutical companies including Pfizer, Roche, "
                "and Novo Nordisk have all launched dedicated AI drug discovery programs using "
                "foundation models. Insilico Medicine became the first company to advance a "
                "fully AI-designed drug candidate into Phase 2 clinical trials for idiopathic "
                "pulmonary fibrosis, compressing the preclinical phase from an average of 5-6 "
                "years to just 18 months. The AI-first approach reduced the cost of identifying "
                "a viable drug candidate by an estimated 70% compared to traditional methods. "
                "However, Nature editorial noted that while AI excels at molecular generation "
                "and property prediction, it cannot yet replace human intuition in understanding "
                "complex biological mechanisms. Critics argue that the true test of AI drug "
                "discovery will come in late-stage clinical trials, where historical failure rates "
                "remain above 90% regardless of how the molecule was designed. A 2025 meta-analysis "
                "of 34 AI-assisted drug programs found that 23 had advanced beyond Phase 1, "
                "suggesting genuine progress but cautioning against overhyped projections. "
                "Regulatory agencies including the FDA are developing new frameworks for "
                "evaluating AI-designed compounds, with guidance expected in late 2025."
            ),
            "chunks": [],
            "word_count": 195,
            "scrape_method": "httpx",
            "scrape_status": "success",
        },
        {
            "result_id": "result_003",
            "subtask_id": "subtask_003",
            "url": "https://www.healthaffairs.org/ai-ethics",
            "title": "Bias and Accountability in Clinical AI: A 2025 Assessment",
            "raw_text": (
                "A comprehensive review published in Health Affairs examined 127 FDA-cleared AI "
                "medical devices and found that fewer than 30% had been validated on datasets "
                "reflecting the racial and ethnic diversity of the U.S. patient population. "
                "The analysis revealed that commercial AI diagnostic tools showed a 12–18 percentage "
                "point performance gap between majority and minority patient groups across multiple "
                "specialties. The Biden administration's executive order on AI safety mandated "
                "bias testing for AI tools used in federal health programs, with enforcement "
                "beginning in 2025. Several high-profile incidents have drawn attention to the "
                "problem, including a pulse oximetry AI that performed poorly on patients with "
                "darker skin tones, potentially delaying treatment. Legal scholars are debating "
                "liability frameworks: when an AI system makes an error, is the hospital, the "
                "vendor, or the clinician responsible? The EU AI Act, which came into force in "
                "2024, classifies medical AI as high-risk and requires mandatory human oversight "
                "for all clinical decision-making tools. Researchers at Stanford propose mandatory "
                "algorithmic audits every two years for all deployed clinical AI systems. In "
                "contrast, a competing perspective from the American Medical Informatics Association "
                "argues that over-regulation will slow beneficial innovation and widen the "
                "healthcare access gap, as AI has enormous potential to extend specialist "
                "expertise to underserved rural communities. Both perspectives acknowledge the "
                "need for better data diversity in AI training sets as a prerequisite for equitable deployment."
            ),
            "chunks": [],
            "word_count": 210,
            "scrape_method": "httpx",
            "scrape_status": "success",
        },
        {
            "result_id": "result_004",
            "subtask_id": "subtask_001",
            "url": "https://www.wsj.com/ai-hospital-blocked",
            "title": "Hospital AI Systems Face Pushback",
            "raw_text": "",
            "chunks": [],
            "word_count": 0,
            "scrape_method": "playwright",
            "scrape_status": "blocked",
        },
    ]

    print("=" * 70)
    print("AnalysisAgent — standalone smoke-test")
    print("Topic: impact of AI on healthcare in 2025")
    print("=" * 70)

    agent = AnalysisAgent()
    result = agent.run(
        scraped_content=mock_scraped_content,
        subtasks=mock_subtasks,
    )

    print("\n" + "=" * 70)
    print("Analysis output:")
    print("=" * 70)
    pprint.pprint(result)
