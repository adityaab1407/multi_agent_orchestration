"""Planner agent that decomposes research topics into subtasks via a ReAct loop
with self-assessed coverage quality.

Why ReAct here (and not single-call):
  The Planner must evaluate whether its subtasks cover the topic adequately.
  A single LLM call produces subtasks but cannot self-assess gaps.  The ReAct
  loop adds a coverage_score + coverage_gaps evaluation after each pass.
  If coverage < 0.70, the gaps are fed back for refinement.  In practice,
  most topics pass on iteration 1, but 2-3 topics per benchmark needed a
  second pass to catch blind spots (e.g. "prevention" missing from a
  healthcare topic that only covered "treatment").

Temperature: 0.4
  Balances creativity (diverse subtask angles) with consistency (valid JSON).
  Lower (0.1-0.2) produces near-identical subtasks across runs.
  Higher (0.7+) breaks JSON formatting too often for reliable parsing.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.settings import (
    GROQ_API_KEY,
    GROQ_REASONING_MODEL,
    MAX_REACT_ITERATIONS,
    PLANNING_QUALITY_THRESHOLD,
)
from utils.llm_utils import strip_llm_response


class SubtaskSchema(BaseModel):
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
    """coverage_score drives the ReAct loop: if below threshold, coverage_gaps
    are fed back to the LLM for refinement."""

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


class PlannerAgent:
    """ReAct-style planner: loops up to max_iterations, calling the LLM,
    parsing the response, and checking coverage_score against the threshold.
    Gaps from low-scoring iterations are fed back for self-correction."""

    # Pool A — Reasoning model
    # ReAct loop requires strong instruction following
    # and consistent JSON output under iteration pressure

    def __init__(self) -> None:
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_REASONING_MODEL,
            temperature=0.4,
        )
        self.max_iterations: int = MAX_REACT_ITERATIONS
        self.quality_threshold: float = PLANNING_QUALITY_THRESHOLD

    def run(self, topic: str, research_id: str) -> list[dict[str, Any]]:
        """Decompose *topic* into subtask dicts via the ReAct loop.

        Returns plain dicts (not Pydantic models) for LangGraph state compat.
        """
        previous_gaps: list[str] = []
        best_output: PlannerOutput | None = None

        for iteration in range(1, self.max_iterations + 1):
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(topic, iteration, previous_gaps)

            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            output = self._parse_llm_response(
                response.content, iteration
            )
            best_output = output

            print(
                f"[Planner THINK] Iteration {iteration} "
                f"— coverage score: {output.coverage_score:.2f}"
            )

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

    def _build_system_prompt(self) -> str:
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

    def _parse_llm_response(
        self,
        response: str,
        iteration: int,
    ) -> PlannerOutput:
        cleaned = strip_llm_response(response)

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
