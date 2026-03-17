"""MCP server providing persistent storage and retrieval for research artifacts.

Phase 2 — Model Context Protocol server that wraps a vector store
(Qdrant) and SQLite behind standardised tool endpoints so that any
MCP-compatible client can store/retrieve embeddings and structured data.

Planned tools:
    store_document(doc_id, text, metadata) -> {status}
    query_similar(query_text, top_k) -> list[{doc_id, score, text}]
    store_artifact(key, value) -> {status}
    get_artifact(key) -> {value}

Protocol reference: https://modelcontextprotocol.io
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StorageServerConfig:
    """Configuration for the MCP storage server."""

    host: str = "0.0.0.0"
    port: int = 8091
    qdrant_path: str = "data/qdrant"  # local file-based Qdrant
    collection_name: str = "newsforge_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    sqlite_path: str = "data/artifacts.db"


class StorageMCPServer:
    """MCP-compliant server for document storage and retrieval."""

    def __init__(self, config: StorageServerConfig | None = None) -> None:
        self.config = config or StorageServerConfig()

    # ------------------------------------------------------------------
    # Tool definitions (MCP tool schema)
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool-definition dicts for registration."""
        return [
            {
                "name": "store_document",
                "description": "Store a document with embeddings in vector store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["doc_id", "text"],
                },
            },
            {
                "name": "query_similar",
                "description": "Find similar documents by semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query_text"],
                },
            },
            {
                "name": "store_artifact",
                "description": "Store a key-value artifact in SQLite.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "get_artifact",
                "description": "Retrieve a stored artifact by key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                    },
                    "required": ["key"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def store_document(self, doc_id: str, text: str, metadata: dict | None = None) -> dict:
        """Embed and store a document in Qdrant."""
        raise NotImplementedError("Phase 2: wire up Qdrant + sentence-transformers.")

    def query_similar(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Semantic similarity search."""
        raise NotImplementedError("Phase 2: wire up Qdrant query.")

    def store_artifact(self, key: str, value: str) -> dict:
        """Store a key-value pair in SQLite."""
        raise NotImplementedError("Phase 2: wire up SQLite storage.")

    def get_artifact(self, key: str) -> dict:
        """Retrieve an artifact by key."""
        raise NotImplementedError("Phase 2: wire up SQLite retrieval.")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the MCP server (HTTP or stdio transport)."""
        raise NotImplementedError("Phase 2: implement MCP server transport.")
