"""Tests for agents/critic.py.

Coverage targets:
  - CriticConfig default values
  - CriticOutputSchema Pydantic V2 validation
  - _parse_llm_response JSON parsing, score recomputation, pass/fail enforcement
  - _build_user_prompt content inclusion
  - _build_system_prompt rubric inclusion
  - run() output shape, score reporting, feedback generation
  - make_fallback_output passthrough behaviour
  - Resilience with empty/minimal inputs

All LLM calls are mocked — no real API I/O in any test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.critic import (
    CriticAgent,
    CriticConfig,
    CriticOutputSchema,
    DimensionScore,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared mock data
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_REPORT = """\
# AI in Healthcare 2025

## Executive Summary
AI is transforming healthcare delivery across diagnostics and operations.

## Key Findings
- 700 lives saved by AI in 2025 [1].
- 47 states introduced AI healthcare bills [2].

## AI Diagnostic Accuracy
Healthcare AI achieved remarkable accuracy in 2025 [1].

## Points of Debate
Cost savings vs implementation costs remain debated [2].

## Methodology
Multi-agent research pipeline with automated web search.

## Limitations
Long-term patient outcome data is limited.

## References
1. Healthcare AI ROI — https://example.com/1
2. AI Regulation — https://example.com/2
"""

MOCK_ANALYSIS = {
    "themes": [
        {"theme": "AI Diagnostic Accuracy", "confidence": 0.9,
         "key_facts": ["700 lives saved"], "supporting_sources": ["https://example.com"]},
    ],
    "contradictions": [
        {"claim_a": "AI saves costs", "claim_b": "Costs offset savings",
         "source_a": "a.com", "source_b": "b.com", "severity": "minor"},
    ],
    "key_facts": ["700 lives saved by healthcare AI in 2025"],
    "confidence_score": 0.80,
    "coverage_gaps": ["Long-term data limited"],
    "sources_analysed": 6,
}

MOCK_SUBTASKS = [
    {"subtask_id": "s001", "title": "AI Applications",
     "search_query": "AI diagnosis 2025", "priority": 1,
     "status": "done", "reasoning": "Core topic"},
]

#: A passing LLM JSON response with all dimensions scoring high.
PASSING_LLM_RESPONSE = json.dumps({
    "passed": True,
    "quality_score": 0.88,
    "dimension_scores": [
        {"dimension": "factual_accuracy", "score": 0.9,
         "reasoning": "Facts match analysis data"},
        {"dimension": "completeness", "score": 0.85,
         "reasoning": "All sections present"},
        {"dimension": "coherence", "score": 0.9,
         "reasoning": "Well structured"},
        {"dimension": "citation_quality", "score": 0.85,
         "reasoning": "Citations properly formatted"},
        {"dimension": "readability", "score": 0.9,
         "reasoning": "Clear and professional"},
    ],
    "feedback_notes": [],
    "strengths": ["Comprehensive coverage", "Good use of citations"],
})

#: A failing LLM JSON response with low scores and feedback.
FAILING_LLM_RESPONSE = json.dumps({
    "passed": False,
    "quality_score": 0.55,
    "dimension_scores": [
        {"dimension": "factual_accuracy", "score": 0.7,
         "reasoning": "Some facts not cited"},
        {"dimension": "completeness", "score": 0.4,
         "reasoning": "Missing regulatory section"},
        {"dimension": "coherence", "score": 0.6,
         "reasoning": "Abrupt transitions"},
        {"dimension": "citation_quality", "score": 0.5,
         "reasoning": "References incomplete"},
        {"dimension": "readability", "score": 0.6,
         "reasoning": "Some jargon unexplained"},
    ],
    "feedback_notes": [
        "Add a dedicated section covering the regulatory landscape",
        "Ensure all factual claims have inline [N] citations",
        "Smooth transitions between sections",
    ],
    "strengths": ["Good executive summary"],
})


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_llm_response(content: str) -> MagicMock:
    """Return a mock LLM response object with a .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_agent_with_mock_llm(
    response_content: str = PASSING_LLM_RESPONSE,
    config: CriticConfig | None = None,
) -> CriticAgent:
    """Return a CriticAgent with a mocked LLM that returns the given content."""
    agent = CriticAgent.__new__(CriticAgent)
    agent.config = config or CriticConfig()
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = _make_mock_llm_response(response_content)
    return agent


# ═══════════════════════════════════════════════════════════════════════════
# 1. CriticConfig defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestCriticConfig:
    """CriticConfig must initialise every field to its documented default."""

    def test_default_values(self):
        cfg = CriticConfig()
        assert cfg.temperature == 0.2
        assert cfg.pass_threshold == 0.75
        assert cfg.max_revisions == 2
        assert len(cfg.scoring_dimensions) == 5
        assert "factual_accuracy" in cfg.scoring_dimensions

    def test_custom_values(self):
        cfg = CriticConfig(pass_threshold=0.9, max_revisions=5)
        assert cfg.pass_threshold == 0.9
        assert cfg.max_revisions == 5


# ═══════════════════════════════════════════════════════════════════════════
# 2. CriticOutputSchema Pydantic V2 validation
# ═══════════════════════════════════════════════════════════════════════════


class TestCriticOutputSchema:
    """CriticOutputSchema must satisfy Pydantic V2 construction and serialisation."""

    def test_model_dump_returns_all_keys(self):
        schema = CriticOutputSchema(
            passed=True,
            quality_score=0.85,
            dimension_scores=[
                DimensionScore(dimension="test", score=0.85, reasoning="good"),
            ],
            feedback_notes=[],
            strengths=["Well written"],
        )
        data = schema.model_dump()
        expected_keys = {"passed", "quality_score", "dimension_scores",
                         "feedback_notes", "strengths"}
        assert expected_keys == set(data.keys())

    def test_model_validate_from_dict(self):
        data = {
            "passed": False,
            "quality_score": 0.5,
            "dimension_scores": [
                {"dimension": "readability", "score": 0.5, "reasoning": "ok"},
            ],
            "feedback_notes": ["Fix formatting"],
            "strengths": ["Good structure"],
        }
        schema = CriticOutputSchema.model_validate(data)
        assert schema.passed is False
        assert schema.quality_score == 0.5

    def test_score_bounds_enforced(self):
        """Scores outside 0.0-1.0 must raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DimensionScore(dimension="test", score=1.5, reasoning="too high")
        with pytest.raises(ValidationError):
            DimensionScore(dimension="test", score=-0.1, reasoning="negative")


# ═══════════════════════════════════════════════════════════════════════════
# 3. _parse_llm_response
# ═══════════════════════════════════════════════════════════════════════════


class TestParseLlmResponse:
    """_parse_llm_response must parse JSON, recompute scores, and enforce pass/fail."""

    def test_parse_valid_passing_response(self):
        agent = _make_agent_with_mock_llm()
        output = agent._parse_llm_response(PASSING_LLM_RESPONSE)
        assert output.passed is True
        assert output.quality_score > 0.75
        assert len(output.dimension_scores) == 5
        assert output.feedback_notes == []

    def test_parse_valid_failing_response(self):
        agent = _make_agent_with_mock_llm()
        output = agent._parse_llm_response(FAILING_LLM_RESPONSE)
        assert output.passed is False
        assert output.quality_score < 0.75
        assert len(output.feedback_notes) > 0

    def test_parse_recomputes_quality_score(self):
        """quality_score must be recomputed as mean of dimension scores."""
        agent = _make_agent_with_mock_llm()
        output = agent._parse_llm_response(PASSING_LLM_RESPONSE)
        expected_avg = sum(
            d.score for d in output.dimension_scores
        ) / len(output.dimension_scores)
        assert abs(output.quality_score - expected_avg) < 0.001

    def test_parse_enforces_passed_consistency(self):
        """passed must be recomputed from quality_score vs threshold."""
        # LLM says passed=True but scores average below threshold
        bad_data = json.dumps({
            "passed": True,  # LLM lies
            "quality_score": 0.9,  # LLM lies
            "dimension_scores": [
                {"dimension": "factual_accuracy", "score": 0.5, "reasoning": "x"},
                {"dimension": "completeness", "score": 0.5, "reasoning": "x"},
                {"dimension": "coherence", "score": 0.5, "reasoning": "x"},
                {"dimension": "citation_quality", "score": 0.5, "reasoning": "x"},
                {"dimension": "readability", "score": 0.5, "reasoning": "x"},
            ],
            "feedback_notes": [],
            "strengths": ["ok"],
        })
        agent = _make_agent_with_mock_llm()
        output = agent._parse_llm_response(bad_data)
        # Average is 0.5, which is below 0.75 threshold
        assert output.passed is False
        assert output.quality_score == 0.5

    def test_parse_clears_feedback_when_passed(self):
        """feedback_notes must be empty when the report passes."""
        # LLM returns feedback even though scores are high
        data = json.dumps({
            "passed": True,
            "quality_score": 0.9,
            "dimension_scores": [
                {"dimension": "factual_accuracy", "score": 0.9, "reasoning": "x"},
                {"dimension": "completeness", "score": 0.9, "reasoning": "x"},
                {"dimension": "coherence", "score": 0.9, "reasoning": "x"},
                {"dimension": "citation_quality", "score": 0.9, "reasoning": "x"},
                {"dimension": "readability", "score": 0.9, "reasoning": "x"},
            ],
            "feedback_notes": ["Unnecessary feedback that should be cleared"],
            "strengths": ["Good"],
        })
        agent = _make_agent_with_mock_llm()
        output = agent._parse_llm_response(data)
        assert output.passed is True
        assert output.feedback_notes == []

    def test_parse_strips_markdown_fences(self):
        """JSON wrapped in backtick fences must still parse correctly."""
        fenced = f"```json\n{PASSING_LLM_RESPONSE}\n```"
        agent = _make_agent_with_mock_llm()
        output = agent._parse_llm_response(fenced)
        assert output.passed is True

    def test_parse_invalid_json_raises_valueerror(self):
        """Non-JSON response must raise ValueError."""
        agent = _make_agent_with_mock_llm()
        with pytest.raises(ValueError, match="invalid JSON"):
            agent._parse_llm_response("This is not JSON at all.")

    def test_parse_missing_fields_raises_valueerror(self):
        """JSON missing required fields must raise ValueError."""
        agent = _make_agent_with_mock_llm()
        with pytest.raises(ValueError, match="Pydantic validation"):
            agent._parse_llm_response('{"passed": true}')


# ═══════════════════════════════════════════════════════════════════════════
# 4. run() — output shape and behaviour
# ═══════════════════════════════════════════════════════════════════════════


class TestRunMethod:
    """run() must return a dict with all CriticOutputSchema keys."""

    def test_run_returns_all_keys(self):
        agent = _make_agent_with_mock_llm(PASSING_LLM_RESPONSE)
        result = agent.run(
            draft_report=SAMPLE_REPORT,
            analysis=MOCK_ANALYSIS,
            subtasks=MOCK_SUBTASKS,
            revision_count=0,
        )
        required_keys = {"passed", "quality_score", "dimension_scores",
                         "feedback_notes", "strengths"}
        assert required_keys == set(result.keys())

    def test_run_passing_report(self):
        agent = _make_agent_with_mock_llm(PASSING_LLM_RESPONSE)
        result = agent.run(
            draft_report=SAMPLE_REPORT,
            analysis=MOCK_ANALYSIS,
            subtasks=MOCK_SUBTASKS,
        )
        assert result["passed"] is True
        assert result["quality_score"] >= 0.75
        assert result["feedback_notes"] == []

    def test_run_failing_report(self):
        agent = _make_agent_with_mock_llm(FAILING_LLM_RESPONSE)
        result = agent.run(
            draft_report=SAMPLE_REPORT,
            analysis=MOCK_ANALYSIS,
            subtasks=MOCK_SUBTASKS,
        )
        assert result["passed"] is False
        assert len(result["feedback_notes"]) > 0

    def test_run_calls_llm_with_system_and_user(self):
        """run() must invoke the LLM with exactly two messages."""
        agent = _make_agent_with_mock_llm(PASSING_LLM_RESPONSE)
        agent.run(
            draft_report=SAMPLE_REPORT,
            analysis=MOCK_ANALYSIS,
            subtasks=MOCK_SUBTASKS,
        )
        agent.llm.invoke.assert_called_once()
        messages = agent.llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_run_with_empty_analysis(self):
        """Empty analysis should not crash."""
        agent = _make_agent_with_mock_llm(PASSING_LLM_RESPONSE)
        result = agent.run(
            draft_report=SAMPLE_REPORT,
            analysis={},
            subtasks=[],
        )
        assert isinstance(result, dict)
        assert "passed" in result

    def test_run_with_revision_count(self):
        """revision_count should be passed through without error."""
        agent = _make_agent_with_mock_llm(PASSING_LLM_RESPONSE)
        result = agent.run(
            draft_report=SAMPLE_REPORT,
            analysis=MOCK_ANALYSIS,
            subtasks=MOCK_SUBTASKS,
            revision_count=2,
        )
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 5. _build_user_prompt content checks
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildUserPrompt:
    """_build_user_prompt must include all relevant data for the LLM."""

    def test_includes_report(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            SAMPLE_REPORT, MOCK_ANALYSIS, MOCK_SUBTASKS, 0,
        )
        assert "AI in Healthcare 2025" in prompt

    def test_includes_subtask_titles(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            SAMPLE_REPORT, MOCK_ANALYSIS, MOCK_SUBTASKS, 0,
        )
        assert "AI Applications" in prompt

    def test_includes_theme_names(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            SAMPLE_REPORT, MOCK_ANALYSIS, MOCK_SUBTASKS, 0,
        )
        assert "AI Diagnostic Accuracy" in prompt

    def test_includes_key_facts(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            SAMPLE_REPORT, MOCK_ANALYSIS, MOCK_SUBTASKS, 0,
        )
        assert "700 lives saved" in prompt

    def test_includes_contradictions(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            SAMPLE_REPORT, MOCK_ANALYSIS, MOCK_SUBTASKS, 0,
        )
        assert "AI saves costs" in prompt

    def test_includes_revision_number(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_user_prompt(
            SAMPLE_REPORT, MOCK_ANALYSIS, MOCK_SUBTASKS, 2,
        )
        assert "REVISION NUMBER: 2" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 6. _build_system_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildSystemPrompt:
    """_build_system_prompt must include rubric dimensions and threshold."""

    def test_includes_all_dimensions(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_system_prompt()
        for dim in agent.config.scoring_dimensions:
            assert dim in prompt

    def test_includes_pass_threshold(self):
        agent = _make_agent_with_mock_llm()
        prompt = agent._build_system_prompt()
        assert str(agent.config.pass_threshold) in prompt

    def test_custom_threshold_reflected(self):
        cfg = CriticConfig(pass_threshold=0.9)
        agent = _make_agent_with_mock_llm(config=cfg)
        prompt = agent._build_system_prompt()
        assert "0.9" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 7. make_fallback_output
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbackOutput:
    """make_fallback_output must return a passing schema to avoid blocking."""

    def test_fallback_passes(self):
        agent = _make_agent_with_mock_llm()
        output = agent.make_fallback_output()
        assert output.passed is True

    def test_fallback_has_empty_feedback(self):
        agent = _make_agent_with_mock_llm()
        output = agent.make_fallback_output()
        assert output.feedback_notes == []

    def test_fallback_has_strength_note(self):
        agent = _make_agent_with_mock_llm()
        output = agent.make_fallback_output()
        assert len(output.strengths) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. Integration: graph routing logic
# ═══════════════════════════════════════════════════════════════════════════


class TestCriticRouter:
    """Verify the _critic_router function from graph.py."""

    def test_router_passes_when_passed(self):
        from orchestrator.graph import _critic_router
        state = {
            "critic_feedback": {"passed": True, "quality_score": 0.9,
                                "feedback_notes": []},
            "revision_count": 1,
        }
        assert _critic_router(state) == "done"

    def test_router_revises_when_failed(self):
        from orchestrator.graph import _critic_router
        state = {
            "critic_feedback": {"passed": False, "quality_score": 0.5,
                                "feedback_notes": ["Fix X"]},
            "revision_count": 1,
        }
        assert _critic_router(state) == "revise"

    def test_router_stops_at_max_revisions(self):
        from orchestrator.graph import _critic_router, MAX_REVISIONS
        state = {
            "critic_feedback": {"passed": False, "quality_score": 0.5,
                                "feedback_notes": ["Fix X"]},
            "revision_count": MAX_REVISIONS + 1,
        }
        assert _critic_router(state) == "done"

    def test_router_handles_no_feedback(self):
        from orchestrator.graph import _critic_router
        state = {"critic_feedback": None, "revision_count": 0}
        assert _critic_router(state) == "done"

    def test_router_revision_count_boundary(self):
        """At exactly MAX_REVISIONS, should still allow one more revision."""
        from orchestrator.graph import _critic_router, MAX_REVISIONS
        state = {
            "critic_feedback": {"passed": False, "quality_score": 0.5,
                                "feedback_notes": ["Fix X"]},
            "revision_count": MAX_REVISIONS,
        }
        assert _critic_router(state) == "revise"
