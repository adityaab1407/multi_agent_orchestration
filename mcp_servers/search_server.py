"""MCP server exposing web search capabilities as tool endpoints.

Phase 2 — Model Context Protocol server that wraps Tavily (and optionally
other search APIs) behind a standardised tool interface so that any
MCP-compatible LLM client can invoke web search as a tool call.

Planned tools:
    search_web(query, max_results) -> list[SearchResult]
    search_news(query, max_results, days_back) -> list[SearchResult]
    search_academic(query, max_results) -> list[SearchResult]

Protocol reference: https://modelcontextprotocol.io
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SearchServerConfig:
    """Configuration for the MCP search server."""

    host: str = "0.0.0.0"
    port: int = 8090
    tavily_api_key: str = ""  # injected from .env
    max_results_default: int = 5
    enable_news_search: bool = True
    enable_academic_search: bool = False


class SearchMCPServer:
    """MCP-compliant server for web search tools."""

    def __init__(self, config: SearchServerConfig | None = None) -> None:
        self.config = config or SearchServerConfig()

    # ------------------------------------------------------------------
    # Tool definitions (MCP tool schema)
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool-definition dicts for registration."""
        return [
            {
                "name": "search_web",
                "description": "Search the web using Tavily API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_news",
                "description": "Search recent news articles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                        "days_back": {"type": "integer", "default": 7},
                    },
                    "required": ["query"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Execute a Tavily web search."""
        raise NotImplementedError("Phase 2: wire up TavilyClient here.")

    def search_news(self, query: str, max_results: int = 5, days_back: int = 7) -> list[dict]:
        """Execute a Tavily news search."""
        raise NotImplementedError("Phase 2: wire up TavilyClient news endpoint.")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the MCP server (HTTP or stdio transport)."""
        raise NotImplementedError("Phase 2: implement MCP server transport.")
