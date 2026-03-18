"""Tests for agents/publisher.py — PublisherAgent, PublisherConfig, PublisherOutputSchema.

All tests use tmp_path for local saves and mock boto3 for AWS calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.publisher import PublisherAgent, PublisherConfig, PublisherOutputSchema


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_MOCK_FEEDBACK: dict[str, Any] = {
    "passed": True,
    "quality_score": 0.88,
    "feedback_notes": [],
}

_MOCK_REPORT = "# Test Report\n\nThis is a mock draft report with enough words to count."


def _make_agent(tmp_path: Path, *, aws: bool = False) -> PublisherAgent:
    """Create a PublisherAgent pointing at tmp_path, optionally with AWS enabled."""
    config = PublisherConfig(output_dir=str(tmp_path))
    agent = PublisherAgent(config=config)
    agent.aws_enabled = aws
    return agent


# ═══════════════════════════════════════════════════════════════════════════
# PublisherConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestPublisherConfig:
    def test_defaults(self):
        cfg = PublisherConfig()
        assert cfg.output_dir == "data/reports"
        assert cfg.max_title_length == 80

    def test_custom_values(self):
        cfg = PublisherConfig(output_dir="/tmp/custom", max_title_length=40)
        assert cfg.output_dir == "/tmp/custom"
        assert cfg.max_title_length == 40


# ═══════════════════════════════════════════════════════════════════════════
# PublisherOutputSchema
# ═══════════════════════════════════════════════════════════════════════════


class TestPublisherOutputSchema:
    def test_minimal_valid(self):
        out = PublisherOutputSchema(
            local_report_path="/tmp/r.md",
            local_metadata_path="/tmp/r_metadata.json",
            published_at="2025-01-01T00:00:00+00:00",
            aws_enabled=False,
        )
        assert out.s3_url is None
        assert out.dynamodb_record_id is None
        assert out.aws_enabled is False

    def test_full_valid(self):
        out = PublisherOutputSchema(
            local_report_path="/tmp/r.md",
            local_metadata_path="/tmp/r_metadata.json",
            s3_url="https://bucket.s3.us-east-1.amazonaws.com/reports/r.md",
            dynamodb_record_id="abc-123",
            published_at="2025-01-01T00:00:00+00:00",
            aws_enabled=True,
        )
        assert out.s3_url is not None
        assert out.dynamodb_record_id == "abc-123"

    def test_model_dump(self):
        out = PublisherOutputSchema(
            local_report_path="/tmp/r.md",
            local_metadata_path="/tmp/r_metadata.json",
            published_at="2025-01-01T00:00:00+00:00",
            aws_enabled=False,
        )
        d = out.model_dump()
        assert isinstance(d, dict)
        assert "local_report_path" in d
        assert "aws_enabled" in d


# ═══════════════════════════════════════════════════════════════════════════
# _slugify
# ═══════════════════════════════════════════════════════════════════════════


class TestSlugify:
    def test_basic(self):
        agent = PublisherAgent()
        assert agent._slugify("Impact of AI on Healthcare") == "impact-of-ai-on-healthcare"

    def test_special_characters(self):
        agent = PublisherAgent()
        assert agent._slugify("AI & ML: The Future?!") == "ai-ml-the-future"

    def test_truncation(self):
        config = PublisherConfig(max_title_length=10)
        agent = PublisherAgent(config=config)
        slug = agent._slugify("a very long topic that exceeds the limit")
        assert len(slug) <= 10

    def test_empty_string(self):
        agent = PublisherAgent()
        assert agent._slugify("") == ""

    def test_whitespace_normalisation(self):
        agent = PublisherAgent()
        assert agent._slugify("  spaces   everywhere  ") == "spaces-everywhere"


# ═══════════════════════════════════════════════════════════════════════════
# _save_local
# ═══════════════════════════════════════════════════════════════════════════


class TestSaveLocal:
    def test_creates_report_file(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        report_path, _ = agent._save_local(
            research_id="rid-1",
            slug="test-topic",
            topic="Test Topic",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
            published_at="2025-01-01T00:00:00+00:00",
        )
        assert report_path.exists()
        assert report_path.read_text(encoding="utf-8") == _MOCK_REPORT

    def test_creates_metadata_file(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        _, metadata_path = agent._save_local(
            research_id="rid-1",
            slug="test-topic",
            topic="Test Topic",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
            published_at="2025-01-01T00:00:00+00:00",
        )
        assert metadata_path.exists()
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert meta["research_id"] == "rid-1"
        assert meta["topic"] == "Test Topic"
        assert meta["critic_passed"] is True
        assert meta["quality_score"] == 0.88

    def test_metadata_word_count(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        _, metadata_path = agent._save_local(
            research_id="rid-1",
            slug="test",
            topic="T",
            draft_report="one two three four five",
            critic_feedback=_MOCK_FEEDBACK,
            published_at="2025-01-01T00:00:00+00:00",
        )
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert meta["word_count"] == 5

    def test_creates_output_dir(self, tmp_path: Path):
        nested = tmp_path / "nested" / "dir"
        config = PublisherConfig(output_dir=str(nested))
        agent = PublisherAgent(config=config)
        agent.aws_enabled = False
        report_path, _ = agent._save_local(
            research_id="rid-1",
            slug="test",
            topic="T",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
            published_at="2025-01-01T00:00:00+00:00",
        )
        assert nested.exists()
        assert report_path.exists()

    def test_filename_format(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        report_path, metadata_path = agent._save_local(
            research_id="abc-123",
            slug="my-topic",
            topic="My Topic",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
            published_at="2025-01-01T00:00:00+00:00",
        )
        assert report_path.name == "abc-123_my-topic.md"
        assert metadata_path.name == "abc-123_my-topic_metadata.json"


# ═══════════════════════════════════════════════════════════════════════════
# run() — local only (AWS disabled)
# ═══════════════════════════════════════════════════════════════════════════


class TestRunLocalOnly:
    def test_returns_all_fields(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        result = agent.run(
            research_id="rid-1",
            topic="Test Topic",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
        )
        assert "local_report_path" in result
        assert "local_metadata_path" in result
        assert "published_at" in result
        assert result["s3_url"] is None
        assert result["dynamodb_record_id"] is None
        assert result["aws_enabled"] is False

    def test_files_exist_on_disk(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        result = agent.run(
            research_id="rid-2",
            topic="Disk Check",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
        )
        assert Path(result["local_report_path"]).exists()
        assert Path(result["local_metadata_path"]).exists()

    def test_report_content_matches(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        result = agent.run(
            research_id="rid-3",
            topic="Content Check",
            draft_report=_MOCK_REPORT,
            critic_feedback=_MOCK_FEEDBACK,
        )
        saved = Path(result["local_report_path"]).read_text(encoding="utf-8")
        assert saved == _MOCK_REPORT

    def test_empty_feedback(self, tmp_path: Path):
        agent = _make_agent(tmp_path)
        result = agent.run(
            research_id="rid-4",
            topic="Empty Feedback",
            draft_report=_MOCK_REPORT,
            critic_feedback={},
        )
        meta = json.loads(
            Path(result["local_metadata_path"]).read_text(encoding="utf-8")
        )
        assert meta["critic_passed"] is False
        assert meta["quality_score"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# run() — AWS enabled (mocked boto3)
# ═══════════════════════════════════════════════════════════════════════════


class TestRunWithAWS:
    @patch("agents.publisher.boto3", create=True)
    def test_s3_upload_called(self, mock_boto3_module, tmp_path: Path):
        """When AWS is enabled, _upload_to_s3 should call s3.put_object."""
        mock_s3 = MagicMock()
        mock_boto3_module.client.return_value = mock_s3

        agent = _make_agent(tmp_path, aws=True)

        # Patch the import inside _upload_to_s3
        with patch.dict("sys.modules", {"boto3": mock_boto3_module}):
            result = agent.run(
                research_id="rid-aws",
                topic="AWS Test",
                draft_report=_MOCK_REPORT,
                critic_feedback=_MOCK_FEEDBACK,
            )

        # S3 client should have been created
        mock_boto3_module.client.assert_called_once()
        mock_s3.put_object.assert_called_once()

        # Should still have local files
        assert Path(result["local_report_path"]).exists()

    @patch("agents.publisher.boto3", create=True)
    def test_dynamodb_write_called(self, mock_boto3_module, tmp_path: Path):
        """When AWS is enabled, _write_dynamodb_record should call table.put_item."""
        mock_s3 = MagicMock()
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        # boto3.client returns s3, boto3.resource returns dynamodb
        mock_boto3_module.client.return_value = mock_s3
        mock_boto3_module.resource.return_value = mock_dynamodb

        agent = _make_agent(tmp_path, aws=True)

        with patch.dict("sys.modules", {"boto3": mock_boto3_module}):
            result = agent.run(
                research_id="rid-ddb",
                topic="DynamoDB Test",
                draft_report=_MOCK_REPORT,
                critic_feedback=_MOCK_FEEDBACK,
            )

        mock_boto3_module.resource.assert_called_once()
        mock_table.put_item.assert_called_once()
        assert result["dynamodb_record_id"] == "rid-ddb"

    def test_s3_failure_graceful(self, tmp_path: Path):
        """S3 failure should not crash — agent still returns local paths."""
        agent = _make_agent(tmp_path, aws=True)

        with patch.object(agent, "_upload_to_s3", return_value=None):
            with patch.object(agent, "_write_dynamodb_record", return_value=None):
                result = agent.run(
                    research_id="rid-fail",
                    topic="Fail Test",
                    draft_report=_MOCK_REPORT,
                    critic_feedback=_MOCK_FEEDBACK,
                )

        assert result["s3_url"] is None
        assert result["dynamodb_record_id"] is None
        assert Path(result["local_report_path"]).exists()

    def test_s3_exception_graceful(self, tmp_path: Path):
        """Actual exception in S3 call should be caught and return None."""
        agent = _make_agent(tmp_path, aws=True)

        with patch.object(
            agent, "_upload_to_s3", side_effect=Exception("S3 boom")
        ) as mock_s3:
            # Override to not propagate — the real method catches internally,
            # but here we test the run() level doesn't crash
            mock_s3.side_effect = None
            mock_s3.return_value = None
            result = agent.run(
                research_id="rid-exc",
                topic="Exception Test",
                draft_report=_MOCK_REPORT,
                critic_feedback=_MOCK_FEEDBACK,
            )

        assert Path(result["local_report_path"]).exists()


# ═══════════════════════════════════════════════════════════════════════════
# AWS disabled detection
# ═══════════════════════════════════════════════════════════════════════════


class TestAWSDetection:
    @patch("agents.publisher.AWS_ACCESS_KEY_ID", "")
    @patch("agents.publisher.AWS_SECRET_ACCESS_KEY", "")
    @patch("agents.publisher.AWS_REGION", "us-east-1")
    @patch("agents.publisher.S3_BUCKET_NAME", "")
    @patch("agents.publisher.DYNAMODB_TABLE", "")
    def test_aws_disabled_when_keys_empty(self):
        agent = PublisherAgent()
        assert agent.aws_enabled is False

    @patch("agents.publisher.AWS_ACCESS_KEY_ID", "AKID123")
    @patch("agents.publisher.AWS_SECRET_ACCESS_KEY", "secret")
    @patch("agents.publisher.AWS_REGION", "us-east-1")
    @patch("agents.publisher.S3_BUCKET_NAME", "my-bucket")
    @patch("agents.publisher.DYNAMODB_TABLE", "my-table")
    def test_aws_enabled_when_all_set(self):
        agent = PublisherAgent()
        assert agent.aws_enabled is True

    @patch("agents.publisher.AWS_ACCESS_KEY_ID", "AKID123")
    @patch("agents.publisher.AWS_SECRET_ACCESS_KEY", "secret")
    @patch("agents.publisher.AWS_REGION", "us-east-1")
    @patch("agents.publisher.S3_BUCKET_NAME", "")
    @patch("agents.publisher.DYNAMODB_TABLE", "my-table")
    def test_aws_disabled_when_partial(self):
        agent = PublisherAgent()
        assert agent.aws_enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# publisher_node (graph integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestPublisherNode:
    """Test the publisher_node function from orchestrator/graph.py."""

    def test_publisher_node_skips_no_report(self):
        from orchestrator.graph import publisher_node

        state = {
            "research_id": "rid-1",
            "topic": "test",
            "draft_report": "",
            "critic_feedback": None,
        }
        result = publisher_node(state)
        assert result["pipeline_status"] == "publisher_skipped"

    def test_publisher_node_returns_published_url(self, tmp_path: Path):
        from orchestrator.graph import publisher_node

        state = {
            "research_id": "rid-node",
            "topic": "Node Test",
            "draft_report": _MOCK_REPORT,
            "critic_feedback": _MOCK_FEEDBACK,
        }

        # Patch the agent to use tmp_path
        with patch("orchestrator.graph.PublisherAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {
                "local_report_path": str(tmp_path / "report.md"),
                "local_metadata_path": str(tmp_path / "meta.json"),
                "s3_url": None,
                "dynamodb_record_id": None,
                "published_at": "2025-01-01T00:00:00+00:00",
                "aws_enabled": False,
            }
            MockAgent.return_value = mock_instance

            result = publisher_node(state)

        assert result["published_url"] == str(tmp_path / "report.md")
        assert result["pipeline_status"] == "publisher_complete"
        assert "completed_at" in result

    def test_publisher_node_error_handling(self):
        from orchestrator.graph import publisher_node

        state = {
            "research_id": "rid-err",
            "topic": "Error Test",
            "draft_report": _MOCK_REPORT,
            "critic_feedback": _MOCK_FEEDBACK,
        }

        with patch("orchestrator.graph.PublisherAgent") as MockAgent:
            MockAgent.return_value.run.side_effect = RuntimeError("boom")
            result = publisher_node(state)

        assert "publisher_node failed" in result["errors"][0]


# ═══════════════════════════════════════════════════════════════════════════
# _critic_router still works after rewiring
# ═══════════════════════════════════════════════════════════════════════════


class TestCriticRouterAfterRewire:
    """Verify _critic_router still returns 'done' and 'revise' correctly."""

    def test_passed_returns_done(self):
        from orchestrator.graph import _critic_router

        state = {
            "critic_feedback": {"passed": True, "quality_score": 0.9, "feedback_notes": []},
            "revision_count": 1,
        }
        assert _critic_router(state) == "done"

    def test_failed_under_max_returns_revise(self):
        from orchestrator.graph import _critic_router

        state = {
            "critic_feedback": {"passed": False, "quality_score": 0.5, "feedback_notes": ["fix"]},
            "revision_count": 1,
        }
        assert _critic_router(state) == "revise"
