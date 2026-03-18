"""Streamlit frontend for the NewsForge multi-agent research system.

Connects to the FastAPI backend at http://localhost:8080 and provides:
- Topic input + Run Research button
- Real-time loading indicator
- Summary metrics (subtasks, results, time, status)
- Pipeline status showing live vs. coming-soon agents
- Subtask cards with nested search-result cards
- Architecture diagram and tech stack table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time

import requests
import streamlit as st

# ─── Backend URL ──────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

# ═══════════════════════════════════════════════════════════════════════════
# Page config (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NewsForge — Multi-Agent Research",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔬",
)

# ═══════════════════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "error" not in st.session_state:
    st.session_state.error = None
if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = None

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .card {
        background: #1e2329; border-radius: 8px;
        padding: 16px; margin-bottom: 12px;
    }
    .metric-box {
        background: #1e2329; border-radius: 8px;
        padding: 20px 16px; text-align: center;
    }
    .metric-box .number {
        font-size: 2rem; font-weight: 700; color: #fff; margin: 0;
    }
    .metric-box .label {
        font-size: 0.85rem; color: #9ca3af; margin: 4px 0 0 0;
    }
    .agent-card {
        background: #1e2329; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
        transition: transform 0.15s ease;
    }
    .agent-card:hover { transform: translateX(4px); }
    .agent-card-live   { border-left: 4px solid #22c55e; }
    .agent-card-pending { border-left: 4px solid #6b7280; }
    .agent-card .agent-name {
        font-weight: 600; color: #fff; font-size: 0.95rem; margin: 0;
    }
    .agent-card .agent-desc {
        color: #9ca3af; font-size: 0.82rem; margin: 2px 0 0 0;
    }
    .result-card {
        background: #1e2329; border: 1px solid #2d333b;
        border-radius: 8px; padding: 16px; margin-bottom: 10px;
        transition: border-color 0.15s ease;
    }
    .result-card:hover { border-color: #58a6ff; }
    .status-badge {
        display: inline-block; padding: 2px 8px;
        border-radius: 4px; font-size: 0.75rem; color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🔬 NewsForge")
st.markdown("### Multi-Agent Research System")
st.caption("Powered by LangGraph · Groq · Tavily · Langfuse")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Input Area
# ═══════════════════════════════════════════════════════════════════════════

topic = st.text_input(
    "Research Topic",
    placeholder="e.g. Impact of AI on healthcare in 2025",
    max_chars=500,
)

run_clicked = st.button("Run Research", type="primary")

if run_clicked:
    if not topic or not topic.strip():
        st.warning("Please enter a topic")
    else:
        st.session_state.is_loading = True
        st.session_state.error = None
        start_time = time.time()

        try:
            with st.spinner("Running pipeline — Planner → Search ..."):
                response = requests.post(
                    f"{BACKEND_URL}/research",
                    json={"topic": topic.strip()},
                    timeout=180,
                )
                response.raise_for_status()
                st.session_state.results = response.json()
                st.session_state.elapsed_time = time.time() - start_time
                st.session_state.error = None

        except requests.exceptions.ConnectionError:
            st.session_state.error = (
                "Cannot connect to backend. Is the API running?\n"
                f"Run: py -m uvicorn backend.main:app --port 8080"
            )
        except requests.exceptions.Timeout:
            st.session_state.error = "Request timed out (180 s). Try a narrower topic."
        except requests.exceptions.HTTPError as e:
            st.session_state.error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
        except Exception as e:
            st.session_state.error = f"Error: {e}"

        st.session_state.is_loading = False

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Error Display
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.error is not None:
    st.error(st.session_state.error)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Results Summary Metrics
# ═══════════════════════════════════════════════════════════════════════════

results = st.session_state.results

if results is not None:
    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""<div class="metric-box">
                <p class="number">{len(results.get("subtasks", []))}</p>
                <p class="label">📋 Subtasks</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-box">
                <p class="number">{len(results.get("search_results", []))}</p>
                <p class="label">🔍 Results</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        elapsed = st.session_state.elapsed_time or 0
        st.markdown(
            f"""<div class="metric-box">
                <p class="number">{elapsed:.1f}s</p>
                <p class="label">⏱️ Time</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        status_text = results.get("status", "unknown").capitalize()
        color = "#22c55e" if status_text == "Complete" else "#f59e0b"
        st.markdown(
            f"""<div class="metric-box">
                <p class="number" style="color:{color}">{status_text}</p>
                <p class="label">✅ Status</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Show pipeline errors if any
    errors = results.get("errors", [])
    if errors:
        with st.expander(f"⚠️ Pipeline Errors ({len(errors)})", expanded=False):
            for err in errors:
                st.warning(err)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Pipeline Status (always visible)
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### ⚙️ Pipeline Status")

col_live, col_pending = st.columns(2)

with col_live:
    st.markdown("#### ✅ Live Agents (Phase 1)")

    for icon, name, desc in [
        ("🧠", "Planner Agent", "Deep ReAct loop — breaks topic into subtasks"),
        ("🔍", "Search Agent", "Tavily web search — 5 results per subtask"),
    ]:
        st.markdown(
            f"""<div class="agent-card agent-card-live">
                <p class="agent-name">{icon} {name}</p>
                <p class="agent-desc">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

with col_pending:
    st.markdown("#### 🔜 Coming Soon (Phase 2)")

    for icon, name, desc in [
        ("🕷️", "Scraper Agent", "Full page content extraction"),
        ("📊", "Analysis Agent", "Theme extraction and fact analysis"),
        ("🎨", "Visual Agent", "Chart and diagram generation"),
        ("✍️", "Writer Agent", "Structured report composition"),
        ("🔎", "Critic Agent", "Quality scoring and revision loop"),
    ]:
        st.markdown(
            f"""<div class="agent-card agent-card-pending">
                <p class="agent-name">{icon} {name}</p>
                <p class="agent-desc">{desc}</p>
            </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Research Results (only if results exist)
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.results is not None:
    res = st.session_state.results

    st.divider()
    st.markdown("### 📋 Research Plan")
    st.markdown(f"**Topic:** {res.get('topic', '')}")

    subtasks = res.get("subtasks", [])
    search_results = res.get("search_results", [])

    for subtask in subtasks:
        sid = subtask.get("subtask_id", "")
        priority = subtask.get("priority", 5)
        title = subtask.get("title", "Untitled")
        query = subtask.get("search_query", "")
        reasoning = subtask.get("reasoning", "")
        status = subtask.get("status", "pending")

        badge_color = "#22c55e" if status == "done" else "#6b7280"

        with st.expander(
            f"[Priority {priority}] {title}",
            expanded=(priority == 1),
        ):
            st.markdown(f"🔍 Query: `{query}`")
            st.markdown(f"*{reasoning}*")
            st.markdown(
                f'<span class="status-badge" style="background:{badge_color}">'
                f"{status}</span>",
                unsafe_allow_html=True,
            )

            matched = [r for r in search_results if r.get("subtask_id") == sid]

            if matched:
                st.markdown("**Search Results**")
                for i, r in enumerate(matched):
                    r_title = r.get("title", "No title")
                    r_url = r.get("url", "#")
                    r_domain = r.get("source_domain", "")
                    r_score = float(r.get("relevance_score", 0))
                    r_snippet = r.get("snippet", "")[:200]

                    st.markdown(f"[**{r_title}**]({r_url})")
                    st.caption(r_domain)
                    st.progress(min(r_score, 1.0))
                    st.markdown(f"*{r_snippet}*")

                    if i < len(matched) - 1:
                        st.markdown("---")
            else:
                st.info("No search results for this subtask.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Architecture Overview
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("### 🏗️ Architecture")

col_flow, col_stack = st.columns(2)

with col_flow:
    st.markdown("#### Pipeline Flow")
    st.code(
        """
User Topic
    |
[Planner Node]   <- ReAct Loop (Groq LLM)        LIVE
    | subtasks
[Search Node]    <- Tool Use (Tavily API)         LIVE
    | search_results
[Scraper Node]   <- BeautifulSoup + Playwright    Phase 2
    | scraped_content
[Analysis Node]  <- Deep ReAct (Groq LLM)        Phase 2
    | analysis
[Visual Node]    <- Groq Vision                   Phase 2
    | visuals
[Writer Node]    <- Structured Generation         Phase 2
    | draft_report
[Critic Node]    <- Quality Check + Revision      Phase 2
    |
Final Report
""",
        language=None,
    )

with col_stack:
    st.markdown("#### Tech Stack")
    st.markdown(
        """
| Component | Technology |
|---|---|
| Orchestration | LangGraph StateGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Search | Tavily API |
| Observability | Langfuse v3 |
| Backend | FastAPI (port 8080) |
| State | SQLite Checkpointer |
| Frontend | Streamlit (port 8501) |
"""
    )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Footer
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.caption("Built by Aditya | NewsForge Multi-Agent Research System")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — Footer
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    "<div style='text-align:center;color:#9ca3af;'>"
    "Built by Aditya | LangGraph + Groq + Tavily | "
    "View on GitHub → <b>newsforge-multi-agent</b>"
    "</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Phase 1: Planner + Search live. Phase 2: 5 agents coming soon."
)
