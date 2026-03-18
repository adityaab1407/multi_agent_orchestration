"""
orchestrator/state.py

Shared state schema for the full NewsForge 7-agent pipeline.
This TypedDict is the single source of truth for all inter-agent data contracts.
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Sub-schemas — these are plain dicts (not TypedDicts) for LangGraph compat
# ---------------------------------------------------------------------------

class Subtask(TypedDict):
    """
    Produced by: Planner Agent
    Consumed by: Search Agent, Analysis Agent
    """
    subtask_id: str          # unique id e.g. "subtask_001"
    title: str               # human-readable title e.g. "AI diagnostics adoption"
    search_query: str        # exact query string to pass to Tavily
    priority: int            # 1 (highest) to 5 (lowest)
    status: str              # "pending" | "searching" | "done" | "failed"
    reasoning: str           # why the Planner decided this subtask matters


class SearchResult(TypedDict):
    """
    Produced by: Search Agent
    Consumed by: Scraper Agent, Analysis Agent
    """
    result_id: str           # unique id e.g. "result_001"
    subtask_id: str          # links back to the Subtask that generated this
    title: str               # article/page title
    url: str                 # source URL
    snippet: str             # summary snippet returned by Tavily
    relevance_score: float   # 0.0 to 1.0, as returned by Tavily
    source_domain: str       # e.g. "nature.com", "arxiv.org"


class ScrapedContent(TypedDict):
    """
    Produced by: Scraper Agent (Phase 2)
    Consumed by: Analysis Agent
    """
    result_id: str           # links back to SearchResult
    url: str
    raw_text: str            # full cleaned page text
    chunks: list[str]        # text split into chunks for embedding
    scrape_status: str       # "success" | "failed" | "blocked"


class AnalysisOutput(TypedDict):
    """
    Produced by: Analysis Agent (Phase 2)
    Consumed by: Visual Agent, Writer Agent
    """
    themes: list[str]        # major themes identified across all content
    key_facts: list[str]     # bullet-point facts with source attribution
    contradictions: list[str] # conflicting claims found across sources
    confidence_score: float  # overall confidence in analysis quality


class VisualOutput(TypedDict):
    """
    Produced by: Visual Agent (Phase 2)
    Consumed by: Writer Agent
    """
    visual_id: str
    description: str         # text description of what the visual shows
    visual_type: str         # "chart" | "diagram" | "image"
    data: Optional[str]      # base64 encoded image or chart JSON


class CriticFeedback(TypedDict):
    """
    Produced by: Critic Agent (Phase 2)
    Consumed by: Writer Agent (revision loop) and Publisher Agent
    """
    passed: bool             # True if quality_score >= threshold
    quality_score: float     # 0.0 to 1.0
    feedback_notes: list[str] # specific revision instructions if passed=False


# ---------------------------------------------------------------------------
# Master Pipeline State
# ---------------------------------------------------------------------------

class NewsForgeState(TypedDict):
    """
    The single shared state object that flows through the entire LangGraph pipeline.

    Design rules:
    - Fields that ACCUMULATE across nodes use Annotated[list, operator.add]
    - Fields that get REPLACED use plain types
    - Every field is Optional — no node should crash if an upstream agent hasn't run yet
    - Phase 2 fields are included now so the schema never needs restructuring later
    """

    # --- Input (set once at pipeline entry, never modified) ---
    research_id: str                          # unique run identifier
    topic: str                                # user's research topic

    # --- Planner Agent output ---
    subtasks: Annotated[list[Subtask], operator.add]

    # --- Search Agent output ---
    search_results: Annotated[list[SearchResult], operator.add]

    # --- Scraper Agent output (Phase 2) ---
    scraped_content: Annotated[list[ScrapedContent], operator.add]

    # --- Analysis Agent output (Phase 2) ---
    analysis: Optional[AnalysisOutput]

    # --- Visual Agent output (Phase 2) ---
    visuals: Annotated[list[VisualOutput], operator.add]

    # --- Writer Agent output (Phase 2) ---
    draft_report: Optional[str]

    # --- Critic Agent output (Phase 2) ---
    critic_feedback: Optional[CriticFeedback]
    revision_count: int                       # incremented each Writer→Critic loop

    # --- Human-in-the-Loop (HITL) review ---
    human_decision: Optional[str]            # "approved" | "rejected" | None

    # --- Publisher Agent output ---
    published_url: Optional[str]
    published_record_id: Optional[str]

    # --- Pipeline metadata (written by orchestrator/graph.py) ---
    pipeline_status: str                      # current active stage name
    errors: Annotated[list[str], operator.add]  # non-fatal errors accumulate
    created_at: str                           # ISO timestamp, set at pipeline start
    completed_at: Optional[str]              # ISO timestamp, set at pipeline end