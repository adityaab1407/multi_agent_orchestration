"""SQLite-backed checkpointer for LangGraph state persistence."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "newsforge_checkpoints.db"



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
