"""Centralized configuration loaded from environment variables and .env files."""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# LLM — Groq
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# =============================================================
# MODEL ROUTING STRATEGY
# Two pools for rate dispersion:
#
# Pool A — Reasoning (llama-4-scout, 30K TPM):
#   Planner, Analysis, Critic
#   Rationale: 30K TPM handles large ReAct prompts.
#   Analysis sends ~6500 tokens per iteration.
#   Critic moved here because Writer(5500t) +
#   Critic(2500t) exceeded 8B's 6K TPM limit.
#   Scout is purpose-built for agentic workloads.
#
# Pool B — Execution (llama-3.1-8b, 6K TPM):
#   Writer, Judge
#   Rationale: Structured generation and rubric
#   scoring do not require large model reasoning.
#   Writer alone (5500t) fits under 6K TPM.
#   Separate pool = independent rate limits.
#
# Token budget per 10-topic benchmark:
#   Pool A: ~102K / 500K TPD = 20% used
#   Pool B: ~80K / 500K TPD = 16% used
# =============================================================

# --- Model Routing ---
# Pool A: Reasoning/Agentic tasks
# High TPM (30K) for large ReAct loop prompts
GROQ_REASONING_MODEL: str = os.getenv(
    "GROQ_REASONING_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

# Pool B: Execution/Evaluation tasks
# Separate rate limit pool for rate dispersion
# Sufficient for structured generation + scoring
GROQ_EXECUTION_MODEL: str = os.getenv(
    "GROQ_EXECUTION_MODEL",
    "llama-3.1-8b-instant",
)

# Judge uses execution pool — evaluation = structured scoring
GROQ_JUDGE_MODEL: str = os.getenv(
    "GROQ_JUDGE_MODEL",
    "llama-3.1-8b-instant",
)

# Keep for backwards compatibility
GROQ_MODEL_NAME: str = os.getenv(
    "GROQ_MODEL_NAME",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

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

# API Authentication — optional (empty = disabled, set value = required)
API_KEY: str = os.getenv("API_KEY", "")

# AWS — optional (Publisher Agent uses S3 + DynamoDB when all vars are set)
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
DYNAMODB_TABLE: str = os.getenv("DYNAMODB_TABLE", "")
