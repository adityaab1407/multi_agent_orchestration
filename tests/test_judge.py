"""Tests for the LLM-as-judge evaluation module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.judge import PASS_THRESHOLD, JudgeOutput, LLMJudge


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_topic() -> dict:
    """A minimal benchmark topic for testing."""
    return {
        "topic_id": "topic_001",
        "topic": "Impact of AI on healthcare in 2025",
        "category": "technology",
        "expected_difficulty": "medium",
        "expected_subtask_count": 5,
        "quality_criteria": [
            "Covers AI diagnostics",
            "Mentions specific companies",
            "Includes statistics",
        ],
        "known_challenges": ["Paywalled sources"],
    }


@pytest.fixture
def sample_pipeline_result() -> dict:
    """A minimal pipeline result dict for testing."""
    return {
        "research_id": "test-123",
        "topic": "Impact of AI on healthcare in 2025",
        "subtasks": [
            {"subtask_id": "s1", "title": "AI diagnostics", "search_query": "AI diagnostics 2025",
             "priority": 1, "status": "done", "reasoning": "Core topic"},
        ],
        "search_results": [
            {"result_id": "r1", "subtask_id": "s1", "title": "AI in Medicine",
             "url": "https://example.com/ai", "snippet": "AI is transforming...",
             "relevance_score": 0.9, "source_domain": "example.com"},
            {"result_id": "r2", "subtask_id": "s1", "title": "Healthcare AI",
             "url": "https://health.org/ai", "snippet": "Healthcare applications...",
             "relevance_score": 0.8, "source_domain": "health.org"},
        ],
        "scraped_content": [
            {"result_id": "r1", "url": "https://example.com/ai",
             "raw_text": "AI is transforming healthcare...", "chunks": ["chunk1"],
             "scrape_status": "success"},
        ],
        "analysis": {
            "themes": ["AI diagnostics", "Drug discovery", "Patient care"],
            "key_facts": ["AI reduces diagnosis time by 40%"],
            "contradictions": [],
            "confidence_score": 0.85,
        },
        "draft_report": "# AI in Healthcare 2025\n\nA comprehensive report on AI...\n" * 50,
        "critic_feedback": {"passed": True, "quality_score": 7.5, "feedback_notes": ["Good depth"]},
        "revision_count": 1,
    }


def _make_judge_response(scores: dict) -> str:
    """Build a valid JSON judge response from score overrides."""
    base = {
        "topic_id": "topic_001",
        "topic": "Impact of AI on healthcare in 2025",
        "research_depth": 7.5,
        "source_diversity": 6.0,
        "topic_coverage": 7.0,
        "factual_coherence": 8.0,
        "report_quality": 7.0,
        "overall_score": 7.1,
        "passed": True,
        "strengths": ["Good depth", "Well structured"],
        "weaknesses": ["Limited source diversity"],
        "judge_reasoning": "The report demonstrates solid research depth.",
    }
    base.update(scores)
    return json.dumps(base)


# ── Tests ─────────────────────────────────────────────────────────────

class TestJudgeOutput:
    """Tests for JudgeOutput Pydantic model validation."""

    def test_judge_returns_valid_schema(self, sample_topic, sample_pipeline_result):
        """Judge should return a valid JudgeOutput with all required fields."""
        mock_response = MagicMock()
        mock_response.content = _make_judge_response({})

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            judge.llm = MagicMock()
            judge.llm.invoke.return_value = mock_response

            output = judge.judge(sample_topic, sample_pipeline_result)

        assert isinstance(output, JudgeOutput)
        assert output.topic_id == "topic_001"
        assert output.topic == "Impact of AI on healthcare in 2025"
        assert 0 <= output.research_depth <= 10
        assert 0 <= output.source_diversity <= 10
        assert 0 <= output.topic_coverage <= 10
        assert 0 <= output.factual_coherence <= 10
        assert 0 <= output.report_quality <= 10
        assert 0 <= output.overall_score <= 10
        assert isinstance(output.passed, bool)
        assert isinstance(output.strengths, list)
        assert isinstance(output.weaknesses, list)
        assert isinstance(output.judge_reasoning, str)

    def test_judge_score_calculation(self, sample_topic, sample_pipeline_result):
        """Overall score should be the average of all five dimension scores."""
        scores = {
            "research_depth": 8.0,
            "source_diversity": 6.0,
            "topic_coverage": 7.0,
            "factual_coherence": 9.0,
            "report_quality": 5.0,
        }
        expected_avg = sum(scores.values()) / 5  # 7.0

        mock_response = MagicMock()
        mock_response.content = _make_judge_response(scores)

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            judge.llm = MagicMock()
            judge.llm.invoke.return_value = mock_response

            output = judge.judge(sample_topic, sample_pipeline_result)

        assert output.overall_score == expected_avg

    def test_judge_handles_empty_report(self, sample_topic):
        """Judge should handle a pipeline result with no report gracefully."""
        empty_result = {
            "draft_report": None,
            "analysis": None,
            "search_results": [],
            "scraped_content": [],
        }

        mock_response = MagicMock()
        mock_response.content = _make_judge_response({
            "research_depth": 1.0,
            "source_diversity": 0.0,
            "topic_coverage": 0.0,
            "factual_coherence": 0.0,
            "report_quality": 0.0,
        })

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            judge.llm = MagicMock()
            judge.llm.invoke.return_value = mock_response

            output = judge.judge(sample_topic, empty_result)

        assert isinstance(output, JudgeOutput)
        assert output.overall_score < PASS_THRESHOLD

    def test_judge_passes_above_threshold(self, sample_topic, sample_pipeline_result):
        """Reports with overall_score >= 6.0 should pass."""
        scores = {
            "research_depth": 7.0,
            "source_diversity": 7.0,
            "topic_coverage": 7.0,
            "factual_coherence": 7.0,
            "report_quality": 7.0,
        }

        mock_response = MagicMock()
        mock_response.content = _make_judge_response(scores)

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            judge.llm = MagicMock()
            judge.llm.invoke.return_value = mock_response

            output = judge.judge(sample_topic, sample_pipeline_result)

        assert output.overall_score == 7.0
        assert output.passed is True

    def test_judge_fails_below_threshold(self, sample_topic, sample_pipeline_result):
        """Reports with overall_score < 6.0 should fail."""
        scores = {
            "research_depth": 4.0,
            "source_diversity": 3.0,
            "topic_coverage": 5.0,
            "factual_coherence": 4.0,
            "report_quality": 3.0,
        }

        mock_response = MagicMock()
        mock_response.content = _make_judge_response(scores)

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            judge.llm = MagicMock()
            judge.llm.invoke.return_value = mock_response

            output = judge.judge(sample_topic, sample_pipeline_result)

        assert output.overall_score == pytest.approx(3.8)
        assert output.passed is False


class TestJudgeResponseParsing:
    """Tests for edge cases in LLM response parsing."""

    def test_handles_markdown_fences(self, sample_topic):
        """Judge should strip markdown code fences from response."""
        raw = "```json\n" + _make_judge_response({}) + "\n```"

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            output = judge._parse_response(raw, sample_topic)

        assert isinstance(output, JudgeOutput)

    def test_handles_think_tags(self, sample_topic):
        """Judge should strip reasoning model <think> tags."""
        raw = "<think>Let me analyze this...</think>\n" + _make_judge_response({})

        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            output = judge._parse_response(raw, sample_topic)

        assert isinstance(output, JudgeOutput)

    def test_fallback_on_invalid_json(self, sample_topic):
        """Judge should return zero-score fallback on unparseable response."""
        with patch.object(LLMJudge, "__init__", lambda self: None):
            judge = LLMJudge()
            output = judge._parse_response("not json at all", sample_topic)

        assert isinstance(output, JudgeOutput)
        assert output.overall_score == 0.0
        assert output.passed is False
        assert len(output.weaknesses) > 0
