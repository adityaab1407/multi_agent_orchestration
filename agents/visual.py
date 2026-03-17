"""Visual agent that generates charts, graphs, and visual summaries.

Phase 2 — Will use Groq Vision / matplotlib / plotly to:
1. Receive analysis output (themes, facts, stats).
2. Decide which visual types best represent the data.
3. Generate charts (bar, pie, timeline) or diagrams.
4. Return VisualOutput dicts (base64-encoded images or chart JSON).

State contract:
    Reads:  state["analysis"]  ->  AnalysisOutput
    Writes: state["visuals"]   ->  list[VisualOutput]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VisualConfig:
    """Tuneable knobs for the Visual Agent."""

    max_visuals: int = 5
    default_chart_lib: str = "matplotlib"  # or "plotly"
    image_format: str = "png"
    dpi: int = 150
    style_theme: str = "dark_background"


class VisualAgent:
    """Generates visual summaries of research analysis."""

    def __init__(self, config: VisualConfig | None = None) -> None:
        self.config = config or VisualConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Create visual outputs from analysis data.

        Args:
            analysis: AnalysisOutput dict with themes, key_facts, etc.

        Returns:
            list of VisualOutput dicts with keys:
                visual_id, description, visual_type, data
        """
        raise NotImplementedError(
            "VisualAgent.run() is a Phase 2 feature. "
            "Implement with matplotlib / plotly."
        )

    # ------------------------------------------------------------------
    # Internal helpers (Phase 2)
    # ------------------------------------------------------------------

    def _generate_theme_chart(self, themes: list[str]) -> bytes:
        """Generate a bar chart of theme frequency."""
        raise NotImplementedError

    def _generate_timeline(self, facts: list[str]) -> bytes:
        """Generate a timeline of key events."""
        raise NotImplementedError

    def _encode_image(self, img_bytes: bytes) -> str:
        """Base64-encode an image for JSON transport."""
        raise NotImplementedError
