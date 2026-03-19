"""Shared state schema for the NewsForge 7-agent pipeline.

State Design Philosophy
-----------------------
This TypedDict is the single source of truth for all data flowing through
the pipeline.  It was designed for all 7 agents *before* any agent code
was written, so every field is an explicit contract between a producer
and a consumer.

Two field styles coexist:

  Annotated[list, operator.add]  — ACCUMULATING fields.
      Returning {"subtasks": [new]} *appends* to the existing list.
      Used for subtasks, search_results, scraped_content, errors
      because multiple nodes contribute to these collections.

  Optional[T]  — REPLACED fields.
      Returning {"draft_report": new_text} *overwrites* the previous value.
      Used for analysis, draft_report, critic_feedback, etc. where only
      the latest value matters (e.g. Writer revisions replace the old draft).
"""

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

    # ── Input (set once at pipeline entry) ──────────────────────────────
    research_id: str
    topic: str

    # ── Planner → Search ─────────────────────────────────────────────────
    # Planner writes, Search reads to build per-subtask queries
    subtasks: Annotated[list[Subtask], operator.add]

    # ── Search → Scraper ─────────────────────────────────────────────────
    # Search writes, Scraper reads URLs to fetch
    search_results: Annotated[list[SearchResult], operator.add]

    # ── Scraper → Analysis ───────────────────────────────────────────────
    # Scraper writes, Analysis reads raw text corpus
    scraped_content: Annotated[list[ScrapedContent], operator.add]

    # ── Analysis → Writer ────────────────────────────────────────────────
    # Analysis writes, Writer reads themes/facts to compose report
    analysis: Optional[AnalysisOutput]

    # ── Writer → Critic (replaced on each revision) ─────────────────────
    draft_report: Optional[str]

    # ── Critic → Writer (feedback loop) / Human Review ───────────────────
    # Replaced each pass; revision_count tracks loop iterations
    critic_feedback: Optional[CriticFeedback]
    revision_count: int

    # ── Human-in-the-Loop → Publisher ────────────────────────────────────
    human_decision: Optional[str]  # "approve" or "reject"

    # ── Publisher output ─────────────────────────────────────────────────
    published_url: Optional[str]
    published_record_id: Optional[str]

    # ── Pipeline metadata (read by all, written by graph runner) ─────────
    pipeline_status: str
    errors: Annotated[list[str], operator.add]  # Accumulated across all nodes
    created_at: str
    completed_at: Optional[str]
