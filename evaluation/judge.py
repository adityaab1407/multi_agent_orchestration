"""LLM-as-judge evaluator for NewsForge benchmark.

LLM-as-Judge Pattern:
  An independent LLM evaluates pipeline output against explicit rubric
  criteria.  This is standard practice for evaluating generative AI systems
  where ground-truth answers don't exist (unlike classification tasks).

  The judge scores on 5 dimensions (0-10 each): research_depth,
  source_diversity, topic_coverage, factual_coherence, report_quality.
  Overall score = average of dimensions.  Pass threshold: 6.0/10.

Why separate from Critic:
  The Critic is an internal quality gate — it runs inside the pipeline and
  can trigger revisions.  The Judge is external — it runs after the pipeline
  completes, during benchmarks only.  They measure different things:
  Critic catches structural/format issues (coherence, citations).
  Judge catches research quality issues (depth, coverage, diversity).
  Combining them into one model would conflate internal and external quality.

Uses llama-3.1-8b-instant (Pool B) by default for efficiency.
Override with GROQ_JUDGE_MODEL env var.

Note on rate limits: the judge adds ~10 LLM calls on top of the
~100 pipeline calls per benchmark.  Recommended: run in batches of 3 topics.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GROQ_API_KEY, GROQ_JUDGE_MODEL
from langchain_groq import ChatGroq
from utils.llm_utils import strip_llm_response


class JudgeOutput(BaseModel):
    """Structured evaluation output from the LLM judge."""

    topic_id: str = Field(description="Benchmark topic ID")
    topic: str = Field(description="The research topic that was evaluated")
    research_depth: float = Field(ge=0, le=10, description="Score 0-10: depth of research")
    source_diversity: float = Field(ge=0, le=10, description="Score 0-10: diversity of sources")
    topic_coverage: float = Field(ge=0, le=10, description="Score 0-10: how fully the topic is covered")
    factual_coherence: float = Field(ge=0, le=10, description="Score 0-10: internal consistency")
    report_quality: float = Field(ge=0, le=10, description="Score 0-10: writing and structure quality")
    overall_score: float = Field(ge=0, le=10, description="Weighted average of all dimensions")
    passed: bool = Field(description="True if overall_score >= 6.0")
    strengths: list[str] = Field(description="What the report did well")
    weaknesses: list[str] = Field(description="What was lacking in the report")
    judge_reasoning: str = Field(description="One paragraph summary of the evaluation")


PASS_THRESHOLD = 6.0


class LLMJudge:
    """Evaluates pipeline output quality using an LLM with structured prompting."""

    # Pool B — Execution model (same as Critic)
    # Benchmark evaluation is structured scoring
    # against explicit rubric criteria.
    # 8B at temperature 0.1 = consistent results.

    def __init__(self) -> None:
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_JUDGE_MODEL,
            temperature=0.1,
        )

    def judge(self, topic_data: dict, pipeline_result: dict) -> JudgeOutput:
        """Evaluate a single pipeline result against topic quality criteria.

        Args:
            topic_data: Benchmark topic dict with quality_criteria and metadata.
            pipeline_result: Full pipeline state dict with draft_report, analysis, etc.

        Returns:
            JudgeOutput with scores, strengths, weaknesses, and reasoning.
        """
        prompt = self._build_judge_prompt(topic_data, pipeline_result)
        response = self.llm.invoke(prompt)
        return self._parse_response(response.content, topic_data)

    def _build_judge_prompt(self, topic_data: dict, pipeline_result: dict) -> str:
        """Build the evaluation prompt with rubric and pipeline output."""
        topic = topic_data["topic"]
        topic_id = topic_data["topic_id"]
        quality_criteria = topic_data.get("quality_criteria", [])
        known_challenges = topic_data.get("known_challenges", [])

        draft_report = pipeline_result.get("draft_report") or "(No report was generated)"
        analysis = pipeline_result.get("analysis") or {}
        search_results = pipeline_result.get("search_results", [])
        scraped_content = pipeline_result.get("scraped_content", [])

        # Summarize sources for the judge
        source_domains = list({r.get("source_domain", "unknown") for r in search_results})
        scrape_success = sum(
            1 for s in scraped_content if s.get("scrape_status") == "success"
        )
        themes = analysis.get("themes", [])
        confidence = analysis.get("confidence_score", 0.0)

        criteria_list = "\n".join(f"  - {c}" for c in quality_criteria)
        challenges_list = "\n".join(f"  - {c}" for c in known_challenges)
        domains_list = ", ".join(source_domains[:15]) if source_domains else "none"
        themes_list = ", ".join(themes[:10]) if themes else "none identified"

        return f"""You are an expert research evaluator. Your job is to objectively score
a research report produced by an automated pipeline.

TOPIC: {topic}
TOPIC ID: {topic_id}

QUALITY CRITERIA (a good report on this topic should):
{criteria_list}

KNOWN CHALLENGES for this topic:
{challenges_list}

PIPELINE METADATA:
- Source domains used: {domains_list}
- Successfully scraped pages: {scrape_success} / {len(scraped_content)}
- Analysis themes identified: {themes_list}
- Analysis confidence score: {confidence:.2f}

REPORT TO EVALUATE:
---
{draft_report}
---

SCORING RUBRIC (score each 0-10):

1. RESEARCH_DEPTH (0-10):
   - 0-2: Surface-level, no specifics, reads like a generic summary
   - 3-4: Some specifics but lacks depth, few statistics or dates
   - 5-6: Decent depth with some statistics and specific examples
   - 7-8: Strong depth with multiple data points, specific studies cited
   - 9-10: Exceptional depth rivaling professional research reports

2. SOURCE_DIVERSITY (0-10):
   - 0-2: All information from 1-2 sources or single domain
   - 3-4: Limited sources, mostly same type (all news or all blogs)
   - 5-6: Mix of sources but could be more diverse
   - 7-8: Good mix of academic, industry, and news sources
   - 9-10: Excellent diversity across multiple domains and perspectives

3. TOPIC_COVERAGE (0-10):
   - 0-2: Covers only one aspect, most quality criteria unmet
   - 3-4: Partial coverage, several criteria missed
   - 5-6: Covers main points but misses some criteria
   - 7-8: Comprehensive coverage, most quality criteria met
   - 9-10: Exhaustive coverage, all quality criteria fully addressed

4. FACTUAL_COHERENCE (0-10):
   - 0-2: Major contradictions, implausible claims
   - 3-4: Some inconsistencies or unsupported claims
   - 5-6: Mostly coherent with minor issues
   - 7-8: Internally consistent, claims well-supported
   - 9-10: Flawless coherence, all claims substantiated

5. REPORT_QUALITY (0-10):
   - 0-2: Poorly structured, hard to follow, no sections
   - 3-4: Basic structure but poorly written
   - 5-6: Readable with clear sections but room for improvement
   - 7-8: Well-structured, professional tone, proper citations
   - 9-10: Publication-ready quality with excellent flow

OUTPUT FORMAT: Respond with ONLY a valid JSON object (no markdown, no backticks):
{{
  "topic_id": "{topic_id}",
  "topic": "{topic}",
  "research_depth": <float 0-10>,
  "source_diversity": <float 0-10>,
  "topic_coverage": <float 0-10>,
  "factual_coherence": <float 0-10>,
  "report_quality": <float 0-10>,
  "overall_score": <float 0-10, average of the 5 scores above>,
  "passed": <true if overall_score >= 6.0, false otherwise>,
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", "weakness 2", ...],
  "judge_reasoning": "One paragraph summarizing your evaluation."
}}"""

    def _parse_response(self, content: str, topic_data: dict) -> JudgeOutput:
        """Parse the LLM response into a JudgeOutput, handling edge cases."""
        text = strip_llm_response(content)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
            else:
                return self._fallback_output(topic_data, f"Failed to parse judge response: {text[:200]}")

        # Calculate overall_score if missing or wrong
        scores = [
            data.get("research_depth", 0),
            data.get("source_diversity", 0),
            data.get("topic_coverage", 0),
            data.get("factual_coherence", 0),
            data.get("report_quality", 0),
        ]
        calculated_overall = sum(scores) / len(scores) if scores else 0.0
        data["overall_score"] = round(calculated_overall, 2)
        data["passed"] = calculated_overall >= PASS_THRESHOLD

        # Ensure required fields
        data.setdefault("topic_id", topic_data["topic_id"])
        data.setdefault("topic", topic_data["topic"])
        data.setdefault("strengths", [])
        data.setdefault("weaknesses", [])
        data.setdefault("judge_reasoning", "")

        return JudgeOutput(**data)

    @staticmethod
    def _fallback_output(topic_data: dict, reason: str) -> JudgeOutput:
        """Return a zero-score output when judge fails to parse."""
        return JudgeOutput(
            topic_id=topic_data["topic_id"],
            topic=topic_data["topic"],
            research_depth=0.0,
            source_diversity=0.0,
            topic_coverage=0.0,
            factual_coherence=0.0,
            report_quality=0.0,
            overall_score=0.0,
            passed=False,
            strengths=[],
            weaknesses=[f"Judge evaluation failed: {reason}"],
            judge_reasoning=f"Unable to evaluate: {reason}",
        )
