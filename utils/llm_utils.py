"""Shared LLM response cleaning utilities for NewsForge agents.

Handles reasoning model artifacts (Qwen3 <think> blocks, DeepSeek-R1)
and markdown code fences that models emit despite instructions.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def strip_llm_response(response: str) -> str:
    """Strip reasoning model artifacts before JSON parsing.

    Handles:
    1. <think>...</think> blocks (Qwen3, DeepSeek-R1)
    2. Markdown code fences (```json ... ```)
    3. Leading/trailing whitespace

    Works for any model — if no think block exists,
    returns cleaned string unchanged.
    """
    # Remove <think>...</think> block if present
    # Use DOTALL so . matches newlines
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL,
    ).strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()
