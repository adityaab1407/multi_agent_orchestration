"""Centralized configuration loaded from environment variables and .env files.

All settings are read once at import time from a ``.env`` file in the project
root (via python-dotenv) and exposed as plain module-level constants so that
every other module can do::

    from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from config/settings.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# LLM — Groq
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Embeddings — local (no API key)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ---------------------------------------------------------------------------
# Vector Store — Qdrant (local folder)
# ---------------------------------------------------------------------------

QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_storage")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_benchmark")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K: int = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Search — Tavily
# ---------------------------------------------------------------------------

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SEARCH_RESULTS_PER_QUERY: int = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))

# ---------------------------------------------------------------------------
# Observability — Langfuse
# ---------------------------------------------------------------------------

LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv(
    "LANGFUSE_BASE_URL",
    os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

# ---------------------------------------------------------------------------
# Pipeline tuning
# ---------------------------------------------------------------------------

MAX_REACT_ITERATIONS: int = int(os.getenv("MAX_REACT_ITERATIONS", "3"))
PLANNING_QUALITY_THRESHOLD: float = float(
    os.getenv("PLANNING_QUALITY_THRESHOLD", "0.7")
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/newsforge_checkpoints.db")
