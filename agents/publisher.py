"""Publisher agent that saves final reports locally and optionally to AWS
(S3 + DynamoDB).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from config.settings import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    DYNAMODB_TABLE,
    S3_BUCKET_NAME,
)

logger = logging.getLogger(__name__)


class PublisherConfig(BaseModel):
    """Tuneable knobs for the Publisher Agent."""

    output_dir: str = "data/reports"
    max_title_length: int = 80


class PublisherOutputSchema(BaseModel):
    local_report_path: str = Field(description="Local file path to the saved .md report")
    local_metadata_path: str = Field(description="Local file path to the metadata JSON")
    s3_url: str | None = Field(default=None, description="S3 object URL if uploaded")
    dynamodb_record_id: str | None = Field(default=None, description="DynamoDB record id if stored")
    published_at: str = Field(description="ISO 8601 timestamp of publish time")
    aws_enabled: bool = Field(description="Whether AWS publishing was attempted")


class PublisherAgent:
    """Saves final research reports locally and optionally to AWS S3 + DynamoDB.

    Local save always happens. AWS upload is attempted only when all required
    AWS env vars are set. If AWS calls fail at runtime, the error is logged
    and the agent still returns successfully with local paths.
    """

    def __init__(self, config: PublisherConfig | None = None) -> None:
        self.config = config or PublisherConfig()
        self.aws_enabled = all([
            AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY,
            AWS_REGION,
            S3_BUCKET_NAME,
            DYNAMODB_TABLE,
        ])

    def run(
        self,
        research_id: str,
        topic: str,
        draft_report: str,
        critic_feedback: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish the final report locally and optionally to AWS."""
        published_at = datetime.now(timezone.utc).isoformat()
        slug = self._slugify(topic)

        report_path, metadata_path = self._save_local(
            research_id=research_id,
            slug=slug,
            topic=topic,
            draft_report=draft_report,
            critic_feedback=critic_feedback,
            published_at=published_at,
        )

        s3_url: str | None = None
        dynamodb_id: str | None = None

        if self.aws_enabled:
            s3_url = self._upload_to_s3(research_id, slug, draft_report)
            dynamodb_id = self._write_dynamodb_record(
                research_id=research_id,
                topic=topic,
                slug=slug,
                s3_url=s3_url,
                local_path=str(report_path),
                critic_feedback=critic_feedback,
                published_at=published_at,
            )

        output = PublisherOutputSchema(
            local_report_path=str(report_path),
            local_metadata_path=str(metadata_path),
            s3_url=s3_url,
            dynamodb_record_id=dynamodb_id,
            published_at=published_at,
            aws_enabled=self.aws_enabled,
        )

        return output.model_dump()

    def _slugify(self, text: str) -> str:
        """Convert topic to a filesystem-safe slug."""
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = slug[: self.config.max_title_length].strip("-")
        return slug

    def _save_local(
        self,
        research_id: str,
        slug: str,
        topic: str,
        draft_report: str,
        critic_feedback: dict[str, Any],
        published_at: str,
    ) -> tuple[Path, Path]:
        """Write the report .md and metadata .json to disk."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        topic_slug = re.sub(r"[^a-z0-9]+", "-", slug)[:20].rstrip("-")
        filename = f"report_{research_id[:8]}_{topic_slug}" if topic_slug else f"report_{research_id[:8]}"

        report_path = out_dir / f"{filename}.md"
        report_path.write_text(draft_report, encoding="utf-8")

        metadata = {
            "research_id": research_id,
            "topic": topic,
            "slug": slug,
            "published_at": published_at,
            "report_file": report_path.name,
            "word_count": len(draft_report.split()),
            "critic_passed": critic_feedback.get("passed", False),
            "quality_score": critic_feedback.get("quality_score", 0.0),
        }
        metadata_path = out_dir / f"{filename}_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        logger.info("Saved report to %s", report_path)
        logger.info("Saved metadata to %s", metadata_path)

        return report_path, metadata_path

    def _upload_to_s3(
        self, research_id: str, slug: str, draft_report: str
    ) -> str | None:
        """Upload report to S3. Returns the S3 URL or None on failure."""
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
            )
            key = f"reports/{research_id}_{slug}.md"
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=key,
                Body=draft_report.encode("utf-8"),
                ContentType="text/markdown",
            )
            url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
            logger.info("Uploaded to S3: %s", url)
            return url

        except Exception as e:
            logger.warning("S3 upload failed (graceful degradation): %s", e)
            return None

    def _write_dynamodb_record(
        self,
        research_id: str,
        topic: str,
        slug: str,
        s3_url: str | None,
        local_path: str,
        critic_feedback: dict[str, Any],
        published_at: str,
    ) -> str | None:
        """Write metadata to DynamoDB. Returns research_id or None on failure."""
        try:
            import boto3

            dynamodb = boto3.resource(
                "dynamodb",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
            )
            table = dynamodb.Table(DYNAMODB_TABLE)
            item = {
                "research_id": research_id,
                "topic": topic,
                "slug": slug,
                "published_at": published_at,
                "local_path": local_path,
                "quality_score": str(critic_feedback.get("quality_score", 0.0)),
                "critic_passed": critic_feedback.get("passed", False),
            }
            if s3_url:
                item["s3_url"] = s3_url

            table.put_item(Item=item)
            logger.info("DynamoDB record created: %s", research_id)
            return research_id

        except Exception as e:
            logger.warning("DynamoDB write failed (graceful degradation): %s", e)
            return None


if __name__ == "__main__":
    import uuid

    rid = str(uuid.uuid4())
    agent = PublisherAgent()

    print(f"AWS enabled: {agent.aws_enabled}")

    result = agent.run(
        research_id=rid,
        topic="Impact of AI on Healthcare in 2025",
        draft_report="# AI in Healthcare\n\nThis is a test report.\n\n## Section 1\n\nContent here.",
        critic_feedback={
            "passed": True,
            "quality_score": 0.92,
            "feedback_notes": [],
        },
    )

    print(f"\nPublished at: {result['published_at']}")
    print(f"Report path : {result['local_report_path']}")
    print(f"Metadata    : {result['local_metadata_path']}")
    print(f"S3 URL      : {result['s3_url']}")
    print(f"DynamoDB ID : {result['dynamodb_record_id']}")
    print(f"AWS enabled : {result['aws_enabled']}")
