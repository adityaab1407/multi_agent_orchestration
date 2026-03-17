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


# ═══════════════════════════════════════════════════════════════════════════
# Request schemas
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Nested response models
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Top-level response schemas
# ═══════════════════════════════════════════════════════════════════════════


class ResearchResponse(BaseModel):
    """Full output returned after a pipeline run completes.

    Contains the decomposed subtasks, search results, error log, and
    timing metadata.
    """

    research_id: str
    topic: str
    status: str = Field(
        ...,
        description='"complete" | "partial" | "failed"',
    )
    subtasks: list[SubtaskResponse]
    search_results: list[SearchResultResponse]
    subtask_count: int
    result_count: int
    errors: list[str]
    created_at: str = Field(..., description="ISO 8601 timestamp")
    completed_at: str = Field(..., description="ISO 8601 timestamp")


class HealthResponse(BaseModel):
    """Response for the ``/health`` endpoint."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    agents_live: list[str] = Field(
        default=["planner", "search"],
        description="Agents with working implementations",
    )
    agents_pending: list[str] = Field(
        default=["scraper", "analysis", "visual", "writer", "critic"],
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
