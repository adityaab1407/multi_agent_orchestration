"""Publisher agent that formats and delivers final reports to output channels.

Phase 2 — Will:
1. Take the critic-approved final report.
2. Format it for multiple output channels (Markdown file, PDF, email).
3. Store the final artifact in the designated output directory.
4. Return publishing metadata (file path, timestamp, format).

State contract:
    Reads:  state["draft_report"]      ->  str
            state["critic_feedback"]    ->  CriticFeedback
            state["topic"]             ->  str
            state["research_id"]        ->  str
    Writes: state["published_url"]      ->  str
            state["published_record_id"] ->  str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PublisherConfig:
    """Tuneable knobs for the Publisher Agent."""

    output_dir: str = "outputs/reports"
    formats: list[str] = field(default_factory=lambda: ["markdown", "pdf"])
    filename_template: str = "{research_id}_{topic_slug}"
    max_title_length: int = 50


class PublisherAgent:
    """Formats and delivers final research reports."""

    def __init__(self, config: PublisherConfig | None = None) -> None:
        self.config = config or PublisherConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        research_id: str,
        topic: str,
        draft_report: str,
        critic_feedback: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish the final report.

        Args:
            research_id: unique research run id.
            topic: original research topic.
            draft_report: final Markdown report string.
            critic_feedback: CriticFeedback dict.

        Returns:
            dict with keys: published_url, published_record_id, formats
        """
        raise NotImplementedError(
            "PublisherAgent.run() is a Phase 2 feature. "
            "Implement file export + optional email delivery."
        )

    # ------------------------------------------------------------------
    # Internal helpers (Phase 2)
    # ------------------------------------------------------------------

    def _slugify(self, text: str) -> str:
        """Convert topic to a filesystem-safe slug."""
        raise NotImplementedError

    def _save_markdown(self, content: str, path: str) -> str:
        """Write Markdown file and return path."""
        raise NotImplementedError

    def _save_pdf(self, content: str, path: str) -> str:
        """Convert Markdown to PDF and return path."""
        raise NotImplementedError
