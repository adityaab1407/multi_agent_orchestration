"""Critic agent that reviews and scores draft reports for quality and accuracy.

Phase 2 — Will use Groq LLM to:
1. Receive the draft report from the Writer Agent.
2. Score it on factual accuracy, completeness, coherence, and style.
3. Provide actionable revision suggestions.
4. Trigger a revision loop (back to Writer) if score < threshold.
5. Return CriticFeedback dict.

State contract:
    Reads:  state["draft_report"]     ->  str
            state["analysis"]          ->  AnalysisOutput
            state["revision_count"]    ->  int
    Writes: state["critic_feedback"]   ->  CriticFeedback
            state["revision_count"]    ->  int  (incremented)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CriticConfig:
    """Tuneable knobs for the Critic Agent."""

    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2
    pass_threshold: float = 0.75  # min score to skip revision
    max_revisions: int = 2
    scoring_rubric: list[str] | None = None  # custom rubric items

    def __post_init__(self) -> None:
        if self.scoring_rubric is None:
            self.scoring_rubric = [
                "factual_accuracy",
                "completeness",
                "coherence",
                "citation_quality",
                "readability",
            ]


class CriticAgent:
    """Reviews draft reports and provides quality feedback."""

    def __init__(self, config: CriticConfig | None = None) -> None:
        self.config = config or CriticConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        draft_report: str,
        analysis: dict[str, Any],
        revision_count: int,
    ) -> dict[str, Any]:
        """Review a draft report and return CriticFeedback.

        Args:
            draft_report: Markdown report string.
            analysis: AnalysisOutput for fact-checking.
            revision_count: number of prior revisions.

        Returns:
            CriticFeedback dict with keys:
                overall_score, dimension_scores, suggestions,
                verdict ("pass" | "revise")
        """
        raise NotImplementedError(
            "CriticAgent.run() is a Phase 2 feature. "
            "Implement with Groq quality-check loop."
        )

    # ------------------------------------------------------------------
    # Internal helpers (Phase 2)
    # ------------------------------------------------------------------

    def _build_review_prompt(self, report: str, facts: list[str]) -> str:
        """Build LLM prompt for report review."""
        raise NotImplementedError

    def _parse_scores(self, llm_output: str) -> dict[str, float]:
        """Extract dimensional scores from LLM response."""
        raise NotImplementedError

    def _should_revise(self, overall_score: float, revision_count: int) -> bool:
        """Decide whether to trigger another revision."""
        return (
            overall_score < self.config.pass_threshold
            and revision_count < self.config.max_revisions
        )
