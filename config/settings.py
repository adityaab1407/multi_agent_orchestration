"""Centralized configuration loaded from environment variables and .env files."""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# LLM — Groq
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# Search — Tavily
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SEARCH_RESULTS_PER_QUERY: int = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))

# Observability — Langfuse
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv(
    "LANGFUSE_BASE_URL",
    os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

# Pipeline tuning
MAX_REACT_ITERATIONS: int = int(os.getenv("MAX_REACT_ITERATIONS", "3"))
PLANNING_QUALITY_THRESHOLD: float = float(
    os.getenv("PLANNING_QUALITY_THRESHOLD", "0.7")
)

# Storage
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/newsforge_checkpoints.db")

# AWS — optional (Publisher Agent uses S3 + DynamoDB when all vars are set)
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
DYNAMODB_TABLE: str = os.getenv("DYNAMODB_TABLE", "")
