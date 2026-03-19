"""Pydantic V2 request and response schemas for the NewsForge FastAPI backend.

Every schema defined here maps directly to an API endpoint's input or output.
Nested models (``SubtaskResponse``, ``SearchResultResponse``, ``AgentStatus``)
are embedded inside the top-level response models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid
from typing import Optional

from pydantic import BaseModel, Field, field_validator



class ResearchRequest(BaseModel):
    """Incoming request to start a new research pipeline run.

    If ``research_id`` is not supplied by the caller the backend will
    generate a UUID4 automatically.
    """

    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research topic to investigate",
    )
    research_id: Optional[str] = Field(
        default=None,
        description="Optional caller-supplied run id; auto-generated if omitted",
    )

    @field_validator("topic")
    @classmethod
    def strip_topic_whitespace(cls, v: str) -> str:
        """Remove leading / trailing whitespace from the topic string."""
        return v.strip()

    def resolve_research_id(self) -> str:
        """Return the research_id, generating one if not provided."""
        return self.research_id or str(uuid.uuid4())



class SubtaskResponse(BaseModel):
    """A single subtask produced by the Planner agent."""

    subtask_id: str
    title: str
    search_query: str
    priority: int
    status: str
    reasoning: str


class SearchResultResponse(BaseModel):
    """A single search result returned by the Search agent via Tavily."""

    result_id: str
    subtask_id: str
    title: str
    url: str
    snippet: str
    relevance_score: float
    source_domain: str



class CriticFeedbackResponse(BaseModel):
    """Critic agent evaluation result."""

    passed: bool
    quality_score: float
    feedback_notes: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    """Full output returned after a pipeline run completes.

    Contains the decomposed subtasks, search results, analysis,
    draft report, critic feedback, and publishing metadata.
    """

    research_id: str
    topic: str
    status: str = Field(
        ...,
        description=(
            '"complete" | "partial" | "failed" | '
            '"awaiting_approval" | "rejected"'
        ),
    )
    subtasks: list[SubtaskResponse]
    search_results: list[SearchResultResponse]
    subtask_count: int
    result_count: int

    # Rich pipeline data
    scraped_content: Optional[list[dict]] = Field(
        default=None, description="Scraper output"
    )
    analysis: Optional[dict] = Field(
        default=None, description="Analysis output with themes, key_facts, contradictions"
    )
    draft_report: Optional[str] = Field(
        default=None, description="Full markdown report from Writer"
    )
    critic_feedback: Optional[CriticFeedbackResponse] = Field(
        default=None, description="Critic evaluation"
    )
    revision_count: int = Field(default=0)
    published_url: Optional[str] = Field(default=None)
    published_record_id: Optional[str] = Field(default=None)

    errors: list[str]
    created_at: str = Field(..., description="ISO 8601 timestamp")
    completed_at: str = Field(..., description="ISO 8601 timestamp")


class HealthResponse(BaseModel):
    """Response for the ``/health`` endpoint."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    agents_live: list[str] = Field(
        default=["planner", "search", "scraper", "analysis", "writer", "critic", "publisher"],
        description="Agents with working implementations",
    )
    agents_pending: list[str] = Field(
        default=[],
        description="Agents still in stub / coming-soon state",
    )


class AgentStatus(BaseModel):
    """Status descriptor for a single agent in the pipeline."""

    name: str
    status: str = Field(..., description='"live" | "coming_soon"')
    description: str = Field(..., description="One-sentence purpose")
    phase: str = Field(..., description='"Phase 1" | "Phase 2"')


class PipelineStatusResponse(BaseModel):
    """Response for the ``/pipeline/status`` endpoint.

    Lists every agent in the pipeline with its current implementation
    status and phase.
    """

    pipeline_version: str = Field(default="1.0.0")
    agents: list[AgentStatus]



class ReviewRequest(BaseModel):
    """Incoming request to approve or reject a paused pipeline run."""

    research_id: str = Field(
        ...,
        description="The research_id of the paused pipeline run",
    )


class ReviewStatusResponse(BaseModel):
    """Status of a specific research pipeline run.

    Returned by ``GET /research/{research_id}/status`` so the frontend
    can poll for the current state of a run.
    """

    research_id: str
    status: str = Field(
        ...,
        description=(
            '"running" | "awaiting_approval" | "approved" | '
            '"rejected" | "complete" | "failed"'
        ),
    )
    topic: str
    report_preview: Optional[str] = Field(
        default=None,
        description="First 1000 chars of the draft report (set when awaiting_approval)",
    )
    quality_score: Optional[float] = Field(
        default=None,
        description="Critic quality score (set when awaiting_approval)",
    )
    word_count: Optional[int] = Field(
        default=None,
        description="Report word count (set when awaiting_approval)",
    )
    revision_count: Optional[int] = Field(
        default=None,
        description="Number of revision passes (set when awaiting_approval)",
    )
