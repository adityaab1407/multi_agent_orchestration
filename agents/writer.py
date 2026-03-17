"""Writer agent that composes polished research reports from analyzed data.

Phase 2 — Will use Groq LLM with structured generation to:
1. Receive analysis output + visuals from upstream agents.
2. Compose a well-structured Markdown research report.
3. Embed visual references inline.
4. Apply citation formatting for all sourced facts.
5. Return the draft report as a string.

State contract:
    Reads:  state["analysis"]    ->  AnalysisOutput
            state["visuals"]     ->  list[VisualOutput]
            state["subtasks"]    ->  list[Subtask]
            state["topic"]       ->  str
    Writes: state["draft_report"] ->  str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WriterConfig:
    """Tuneable knobs for the Writer Agent."""

    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.5
    max_report_words: int = 3000
    include_table_of_contents: bool = True
    citation_style: str = "inline"  # "inline" | "footnote"
    output_format: str = "markdown"


class WriterAgent:
    """Composes structured research reports from analysis and visuals."""

    def __init__(self, config: WriterConfig | None = None) -> None:
        self.config = config or WriterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        topic: str,
        analysis: dict[str, Any],
        visuals: list[dict[str, Any]],
        subtasks: list[dict[str, Any]],
    ) -> str:
        """Compose a draft research report.

        Args:
            topic: original research topic.
            analysis: AnalysisOutput dict.
            visuals: list of VisualOutput dicts.
            subtasks: original subtask list for structure.

        Returns:
            Markdown-formatted draft report string.
        """
        raise NotImplementedError(
            "WriterAgent.run() is a Phase 2 feature. "
            "Implement with Groq structured generation."
        )

    # ------------------------------------------------------------------
    # Internal helpers (Phase 2)
    # ------------------------------------------------------------------

    def _build_outline(self, subtasks: list[dict], themes: list[str]) -> str:
        """Generate a report outline from subtasks and themes."""
        raise NotImplementedError

    def _format_citations(self, text: str, facts: list[str]) -> str:
        """Insert inline or footnote citations."""
        raise NotImplementedError

    def _insert_visuals(self, report: str, visuals: list[dict]) -> str:
        """Embed visual references into the report."""
        raise NotImplementedError
