"""Shared state schema for the NewsForge 7-agent pipeline."""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class Subtask(TypedDict):
    subtask_id: str
    title: str
    search_query: str
    priority: int
    status: str
    reasoning: str


class SearchResult(TypedDict):
    result_id: str
    subtask_id: str
    title: str
    url: str
    snippet: str
    relevance_score: float
    source_domain: str


class ScrapedContent(TypedDict):
    result_id: str
    url: str
    raw_text: str
    chunks: list[str]
    scrape_status: str


class AnalysisOutput(TypedDict):
    themes: list[str]
    key_facts: list[str]
    contradictions: list[str]
    confidence_score: float


class CriticFeedback(TypedDict):
    passed: bool
    quality_score: float
    feedback_notes: list[str]


class NewsForgeState(TypedDict):
    """The single shared state object that flows through the entire LangGraph pipeline.

    Fields that ACCUMULATE across nodes use Annotated[list, operator.add].
    Fields that get REPLACED use plain types.
    """

    # Input (set once at pipeline entry)
    research_id: str
    topic: str

    # Planner Agent output
    subtasks: Annotated[list[Subtask], operator.add]

    # Search Agent output
    search_results: Annotated[list[SearchResult], operator.add]

    # Scraper Agent output
    scraped_content: Annotated[list[ScrapedContent], operator.add]

    # Analysis Agent output
    analysis: Optional[AnalysisOutput]

    # Writer Agent output
    draft_report: Optional[str]

    # Critic Agent output
    critic_feedback: Optional[CriticFeedback]
    revision_count: int

    # Human-in-the-Loop review
    human_decision: Optional[str]

    # Publisher Agent output
    published_url: Optional[str]
    published_record_id: Optional[str]

    # Pipeline metadata
    pipeline_status: str
    errors: Annotated[list[str], operator.add]
    created_at: str
    completed_at: Optional[str]
