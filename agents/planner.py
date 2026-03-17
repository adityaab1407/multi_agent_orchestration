"""Planner agent that decomposes research topics into actionable sub-tasks.

This module contains:

* **SubtaskSchema / PlannerOutput** — Pydantic V2 models that define the
  structured JSON contract between the LLM and the rest of the pipeline.
* **PlannerAgent** — A ReAct-style agent that calls Groq (llama-3.3-70b)
  up to ``MAX_REACT_ITERATIONS`` times, self-assessing coverage quality
  and refining subtasks until the score exceeds
  ``PLANNING_QUALITY_THRESHOLD``.

The agent is invoked by the ``planner_node`` in ``orchestrator/graph.py``
but has **zero LangGraph imports** — it is pure agent logic.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    MAX_REACT_ITERATIONS,
    PLANNING_QUALITY_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 schemas
# ═══════════════════════════════════════════════════════════════════════════


class SubtaskSchema(BaseModel):
    """A single research subtask produced by the Planner.

    Each subtask carries a Tavily-ready ``search_query`` so the downstream
    Search Agent can execute it without further transformation.
    """

    subtask_id: str = Field(
        ...,
        description='Sequential id, e.g. "subtask_001"',
    )
    title: str = Field(
        ...,
        description='Human-readable label, e.g. "AI diagnostics in radiology"',
    )
    search_query: str = Field(
        ...,
        description="Exact query string to pass to Tavily search",
    )
    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="1 (highest) to 5 (lowest)",
    )
    status: str = Field(
        default="pending",
        description='Always "pending" when created by the Planner',
    )
    reasoning: str = Field(
        ...,
        description="Why this subtask matters for the research topic",
    )


class PlannerOutput(BaseModel):
    """Complete output of a single Planner ReAct iteration.

    ``coverage_score`` is the LLM's self-assessed quality metric.  If it
    falls below ``PLANNING_QUALITY_THRESHOLD`` the ReAct loop will run an
    additional iteration using the identified ``coverage_gaps``.
    """

    subtasks: list[SubtaskSchema]
    coverage_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-assessed topic coverage, 0.0 to 1.0",
    )
    coverage_gaps: list[str] = Field(
        default_factory=list,
        description="Angles / dimensions still missing from the plan",
    )
    iteration: int = Field(
        ...,
        ge=1,
        description="Which ReAct iteration produced this output",
    )


# ═══════════════════════════════════════════════════════════════════════════
# PlannerAgent
# ═══════════════════════════════════════════════════════════════════════════


class PlannerAgent:
    """ReAct-style planner that decomposes a research topic into subtasks.

    The agent loops up to ``max_iterations`` times.  On every iteration it:

    1. **ACT**   — calls the Groq LLM with a structured JSON prompt.
    2. **THINK** — parses the response through Pydantic validation.
    3. **OBSERVE** — checks ``coverage_score`` against the threshold.
       If the score is too low the identified *gaps* are fed back into
       the next iteration's prompt so the LLM can self-correct.
    """

    def __init__(self) -> None:
        """Initialise the Planner with a ChatGroq LLM and tuning params.

        Reads ``GROQ_API_KEY`` and ``GROQ_MODEL_NAME`` from
        ``config.settings`` (which in turn loads them from ``.env``).
        """
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=0.4,
        )
        self.max_iterations: int = MAX_REACT_ITERATIONS
        self.quality_threshold: float = PLANNING_QUALITY_THRESHOLD

    # ── public API ────────────────────────────────────────────────────

    def run(self, topic: str, research_id: str) -> list[dict[str, Any]]:
        """Decompose *topic* into a list of subtask dicts via a ReAct loop.

        This is the single entry-point called by ``planner_node`` in
        ``orchestrator/graph.py``.

        Args:
            topic: The user's research topic string.
            research_id: Unique pipeline run id (for logging / tracing).

        Returns:
            A ``list[dict]`` of subtasks ready to merge into
            ``NewsForgeState["subtasks"]`` (converted from Pydantic models
            so they are plain dicts compatible with LangGraph state).
        """
        previous_gaps: list[str] = []
        best_output: PlannerOutput | None = None

        for iteration in range(1, self.max_iterations + 1):
            # ── ACT: call the LLM ─────────────────────────────────────
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(topic, iteration, previous_gaps)

            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            # ── THINK: parse + validate ───────────────────────────────
            output = self._parse_llm_response(
                response.content, iteration
            )
            best_output = output

            print(
                f"[Planner THINK] Iteration {iteration} "
                f"— coverage score: {output.coverage_score:.2f}"
            )

            # ── OBSERVE: good enough? ─────────────────────────────────
            if output.coverage_score >= self.quality_threshold:
                break

            previous_gaps = output.coverage_gaps
            print(
                f"[Planner OBSERVE] Gaps found: {previous_gaps}. "
                f"Refining..."
            )

        subtask_dicts = [
            subtask.model_dump() for subtask in best_output.subtasks
        ]
        print(
            f"[Planner] Final output: {len(subtask_dicts)} subtasks generated"
        )
        return subtask_dicts

    # ── prompt builders ───────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Return the system prompt that forces structured JSON output.

        The prompt instructs the LLM to:
        - Think about what dimensions of the topic need coverage.
        - Generate 3-5 subtasks with specific, Tavily-ready search queries.
        - Self-assess coverage on a 0.0-1.0 scale.
        - List any gaps it identifies.
        - Output **only** valid JSON matching the ``PlannerOutput`` schema.
        """
        return textwrap.dedent("""\
            You are the Planner agent in a multi-agent research system called NewsForge.

            YOUR TASK:
            Given a research topic, decompose it into 3-5 actionable subtasks.
            Each subtask must include an exact search query suitable for the Tavily web-search API.

            THINKING PROCESS:
            1. Identify the key dimensions / angles of the topic (economic, social, technical, ethical, geographic, temporal, etc.).
            2. For each important dimension, create a subtask with:
               - A clear, specific title
               - A Tavily-optimised search query (concise, keyword-rich, no boolean operators)
               - A priority from 1 (most important) to 5 (least)
               - A one-sentence reasoning explaining why this subtask matters
            3. Self-assess how well your subtasks cover the full topic on a 0.0-1.0 scale.
            4. List any remaining gaps (angles you did NOT cover).

            OUTPUT FORMAT — respond with ONLY raw JSON, no markdown, no backticks, no preamble:
            {
              "subtasks": [
                {
                  "subtask_id": "subtask_001",
                  "title": "...",
                  "search_query": "...",
                  "priority": 1,
                  "status": "pending",
                  "reasoning": "..."
                }
              ],
              "coverage_score": 0.85,
              "coverage_gaps": ["..."],
              "iteration": 1
            }

            RULES:
            - Generate between 3 and 5 subtasks.
            - subtask_id must be sequential: subtask_001, subtask_002, etc.
            - status must always be "pending".
            - coverage_score must honestly reflect how much of the topic your subtasks cover.
            - Output ONLY the JSON object. No explanation before or after.
        """)

    def _build_user_prompt(
        self,
        topic: str,
        iteration: int,
        previous_gaps: list[str],
    ) -> str:
        """Build the user message for a given ReAct iteration.

        Args:
            topic: The research topic.
            iteration: Current iteration number (1-based).
            previous_gaps: Gaps identified in the prior iteration (empty on
                the first pass).

        Returns:
            A prompt string.  On iteration 1 it is simply the topic.  On
            subsequent iterations it includes the gaps so the LLM can
            self-correct.
        """
        if iteration == 1 or not previous_gaps:
            return (
                f"Research topic: {topic}\n\n"
                f"Generate subtasks for iteration {iteration}."
            )

        gaps_text = "; ".join(previous_gaps)
        return (
            f"Research topic: {topic}\n\n"
            f"Previous gaps identified: {gaps_text}. "
            f"Please address these in your revised subtasks.\n\n"
            f"Generate subtasks for iteration {iteration}."
        )

    # ── response parsing ──────────────────────────────────────────────

    def _parse_llm_response(
        self,
        response: str,
        iteration: int,
    ) -> PlannerOutput:
        """Parse and validate the raw LLM string into a ``PlannerOutput``.

        Args:
            response: The raw text content returned by the LLM.
            iteration: The current iteration (used to override the
                ``iteration`` field if the LLM gets it wrong).

        Returns:
            A validated ``PlannerOutput`` instance.

        Raises:
            ValueError: If the response is not valid JSON or fails Pydantic
                validation, with a message showing what went wrong.
        """
        # Strip markdown fences if the LLM wraps its output despite instructions
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Planner LLM returned invalid JSON on iteration {iteration}. "
                f"JSONDecodeError: {exc}. Raw response (first 500 chars): "
                f"{response[:500]}"
            ) from exc

        # Force the correct iteration number
        data["iteration"] = iteration

        try:
            return PlannerOutput.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Planner LLM JSON failed Pydantic validation on iteration "
                f"{iteration}: {exc}. Parsed data keys: {list(data.keys())}"
            ) from exc


# ═══════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pprint

    agent = PlannerAgent()
    subtasks = agent.run(
        topic="impact of AI on healthcare in 2025",
        research_id="test_planner_001",
    )

    print("\n" + "=" * 60)
    print("Planner output — subtasks:")
    print("=" * 60)
    pprint.pprint(subtasks)
