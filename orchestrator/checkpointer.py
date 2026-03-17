"""
orchestrator/checkpointer.py

SQLite-backed checkpointer for LangGraph state persistence.

This enables resume capability — if the pipeline crashes mid-run, it restarts
from the last completed node, not from scratch.

HOW IT WORKS:
    LangGraph's SqliteSaver serialises the full state dict after every node
    execution and writes it to a local SQLite database keyed by (thread_id,
    checkpoint_id).  When the graph is invoked again with the same thread_id,
    it reads the latest checkpoint and resumes from where it left off.

WHY IT MATTERS:
    - Long research pipelines (7 agents) can take minutes.  Without
      checkpointing, any transient failure (rate-limit, network blip) means
      rerunning the entire pipeline.
    - With checkpointing, only the failed node and its successors are re-executed.
    - It also enables "pause / resume" workflows where a human reviewer can
      inspect intermediate state before continuing.
"""

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "newsforge_checkpoints.db"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_checkpointer() -> SqliteSaver:
    """Return a configured SqliteSaver instance backed by a local SQLite file.

    The database is stored at ``data/newsforge_checkpoints.db`` relative to
    the project root.  The ``data/`` directory is created automatically if it
    does not already exist.

    Returns:
        SqliteSaver: A LangGraph-compatible checkpointer ready to be passed
        to ``StateGraph.compile(checkpointer=...)``.
    """
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    return SqliteSaver(conn=conn)
