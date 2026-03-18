"""NewsForge — Streamlit frontend for the 7-agent LangGraph research pipeline.

Design decisions and interview talking points:
─────────────────────────────────────────────────────────────────────────────

1. PIPELINE VISUALIZER as the centrepiece.  Each agent is a card in a
   horizontal flow.  Arrows between cards carry the state-field name that
   flows between agents (e.g. "subtasks[]", "search_results[]").  This
   makes the LangGraph state graph tangible to non-technical viewers.

2. STATUS PROGRESSION.  Cards start grey ("waiting"), turn blue while
   running, green on success, red on failure.  Streamlit rerenders the
   whole page on each interaction, so we colour-code from the result
   dict after the pipeline returns.

3. REVISION LOOP.  If revision_count > 0 we render a visible "loop"
   indicator between Critic and Writer showing each round's score delta.
   This is the most impressive feature — it demonstrates self-improving
   AI.

4. HITL MOMENT.  When status == "awaiting_approval" a distinct amber
   panel appears with the report preview, quality score, and two large
   Approve / Reject buttons.  The visual contrast signals "human action
   required" and differentiates this from the automated steps.

5. REPORT as payoff.  The full markdown report is rendered beautifully
   with metadata badges (word count, quality score, revisions) and a
   download button.  After watching 7 agents work, seeing the polished
   output feels earned.

6. ANALYSIS DEEP DIVE.  Collapsible section with theme cards, confidence
   bars, contradictions, and coverage gaps.  Default closed so the page
   isn't overwhelming, but available for those who want the detail.

Tech: Streamlit + custom HTML/CSS via st.markdown(unsafe_allow_html=True).
Backend: FastAPI at http://localhost:8080.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
from typing import Any, Optional

import requests
import streamlit as st

# ─── Backend URL ──────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NewsForge — AI Research Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔬",
)

# ═══════════════════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "results": None,
    "is_loading": False,
    "error": None,
    "elapsed_time": None,
    "research_id": None,
    "awaiting_approval": False,
    "report_preview": None,
    "quality_score_preview": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════════════════
# Global CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ───────────────────────────────────── */
.stApp { background-color: #0e1117; }
.block-container { max-width: 1200px; }

/* ── Agent pipeline cards ───────────────────── */
.agent-flow { display: flex; align-items: flex-start; gap: 6px;
              overflow-x: auto; padding: 8px 0; }
.agent-card-v { flex: 0 0 130px; border-radius: 10px; padding: 14px 10px;
                text-align: center; position: relative; min-height: 160px;
                transition: all 0.2s ease; }
.agent-card-v .icon { font-size: 1.6rem; margin-bottom: 4px; }
.agent-card-v .name { font-weight: 700; font-size: 0.78rem; color: #fff;
                      margin: 2px 0; line-height: 1.2; }
.agent-card-v .metric { font-size: 0.72rem; color: #d1d5db;
                        margin-top: 6px; line-height: 1.3; }
.agent-card-v .metric b { color: #fff; }

/* Status colours */
.agent-waiting  { background: #1e2329; border: 2px solid #374151; }
.agent-complete { background: #0d2818; border: 2px solid #22c55e; }
.agent-failed   { background: #2d1215; border: 2px solid #ef4444; }
.agent-active   { background: #0c1929; border: 2px solid #3b82f6;
                  box-shadow: 0 0 12px rgba(59,130,246,0.3); }
.agent-hitl     { background: #2d2305; border: 2px solid #f59e0b;
                  box-shadow: 0 0 12px rgba(245,158,11,0.25); }

/* ── Arrow between agents ───────────────────── */
.arrow-label { flex: 0 0 auto; display: flex; flex-direction: column;
               align-items: center; justify-content: center;
               padding-top: 28px; min-width: 56px; }
.arrow-label .arrow { color: #6b7280; font-size: 1.1rem; }
.arrow-label .field { font-size: 0.6rem; color: #9ca3af;
                      font-family: monospace; margin-top: 2px;
                      white-space: nowrap; }

/* ── Revision loop badge ────────────────────── */
.revision-badge { display: inline-block; background: #1e2329;
                  border: 1px solid #f59e0b; border-radius: 6px;
                  padding: 6px 12px; margin: 4px 2px; font-size: 0.78rem;
                  color: #fbbf24; }

/* ── HITL panel ─────────────────────────────── */
.hitl-panel { background: linear-gradient(135deg, #2d2305 0%, #1e2329 100%);
              border: 2px solid #f59e0b; border-radius: 12px;
              padding: 28px; margin: 20px 0; }
.hitl-panel h2 { color: #fbbf24; margin: 0 0 4px 0; font-size: 1.4rem; }
.hitl-panel .subtitle { color: #d1d5db; font-size: 0.9rem; margin-bottom: 16px; }

/* ── Metric boxes ───────────────────────────── */
.metric-box { background: #1e2329; border-radius: 10px;
              padding: 18px 14px; text-align: center; }
.metric-box .number { font-size: 1.8rem; font-weight: 700; color: #fff;
                      margin: 0; line-height: 1.2; }
.metric-box .label  { font-size: 0.8rem; color: #9ca3af; margin: 4px 0 0 0; }

/* ── Report metadata badges ─────────────────── */
.report-meta { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
.meta-badge { display: inline-flex; align-items: center; gap: 6px;
              background: #1e2329; border-radius: 6px; padding: 6px 12px;
              font-size: 0.82rem; color: #d1d5db; }
.meta-badge .val { font-weight: 700; color: #fff; }

/* ── Theme cards ────────────────────────────── */
.theme-card { background: #1e2329; border-radius: 8px; padding: 14px;
              margin-bottom: 10px; border-left: 3px solid #3b82f6; }
.theme-card .theme-name { font-weight: 600; color: #fff;
                          font-size: 0.92rem; margin: 0 0 6px 0; }
.theme-card .theme-facts { color: #9ca3af; font-size: 0.8rem;
                           margin: 0; line-height: 1.5; }

/* ── Quality score colour helpers ───────────── */
.q-green  { color: #22c55e; }
.q-yellow { color: #f59e0b; }
.q-red    { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "q-green"
    if score >= 0.6:
        return "q-yellow"
    return "q-red"


def _score_css(score: float) -> str:
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.6:
        return "#f59e0b"
    return "#ef4444"


def _agent_status_class(agent_key: str, results: Optional[dict]) -> str:
    """Determine the CSS class for a pipeline agent card."""
    if results is None:
        return "agent-waiting"

    status = results.get("status", "")

    # HITL node
    if agent_key == "human_review":
        if status == "awaiting_approval":
            return "agent-hitl"
        if status in ("complete", "rejected"):
            return "agent-complete"
        return "agent-waiting"

    # Publisher
    if agent_key == "publisher":
        if results.get("published_url"):
            return "agent-complete"
        if status == "rejected":
            return "agent-waiting"
        return "agent-waiting"

    # Map agent keys to the data they produce
    _completion_checks = {
        "planner": lambda r: len(r.get("subtasks", [])) > 0,
        "search": lambda r: len(r.get("search_results", [])) > 0,
        "scraper": lambda r: r.get("scraped_content") is not None,
        "analysis": lambda r: r.get("analysis") is not None,
        "writer": lambda r: r.get("draft_report") is not None,
        "critic": lambda r: r.get("critic_feedback") is not None,
    }

    check = _completion_checks.get(agent_key)
    if check and check(results):
        return "agent-complete"

    errors = results.get("errors", [])
    if any(agent_key in e for e in errors):
        return "agent-failed"

    return "agent-waiting"


def _agent_metric(agent_key: str, results: Optional[dict]) -> str:
    """Return a short metric string for the agent card."""
    if results is None:
        return ""

    r = results
    if agent_key == "planner":
        n = len(r.get("subtasks", []))
        return f"<b>{n}</b> subtasks" if n else ""
    if agent_key == "search":
        n = len(r.get("search_results", []))
        return f"<b>{n}</b> results" if n else ""
    if agent_key == "scraper":
        sc = r.get("scraped_content")
        if sc:
            ok = sum(1 for s in sc if s.get("scrape_status") == "success")
            return f"<b>{ok}/{len(sc)}</b> pages"
        return ""
    if agent_key == "analysis":
        a = r.get("analysis")
        if a:
            return f"<b>{len(a.get('themes', []))}</b> themes"
        return ""
    if agent_key == "writer":
        dr = r.get("draft_report")
        if dr:
            return f"<b>{len(dr.split())}</b> words"
        return ""
    if agent_key == "critic":
        cf = r.get("critic_feedback")
        if cf:
            sc = cf.get("quality_score", 0)
            cls = _score_color(sc)
            p = "passed" if cf.get("passed") else "failed"
            return f'<span class="{cls}"><b>{sc:.2f}</b></span> {p}'
        return ""
    if agent_key == "human_review":
        s = r.get("status", "")
        if s == "awaiting_approval":
            return "<b>Waiting</b>"
        if s == "rejected":
            return "<b>Rejected</b>"
        if s in ("complete",):
            return "<b>Approved</b>"
        return ""
    if agent_key == "publisher":
        if r.get("published_url"):
            return "<b>Saved</b>"
        return ""
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🔬 NewsForge")
st.markdown(
    "<p style='color:#9ca3af; margin-top:-10px; font-size:1.05rem;'>"
    "7-Agent AI Research Pipeline &nbsp;·&nbsp; "
    "LangGraph &nbsp;·&nbsp; Groq &nbsp;·&nbsp; Tavily &nbsp;·&nbsp; Langfuse"
    "</p>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Input
# ═══════════════════════════════════════════════════════════════════════════

col_input, col_btn = st.columns([5, 1])
with col_input:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Impact of AI on healthcare in 2025",
        max_chars=500,
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("Run Research", type="primary", use_container_width=True)

if run_clicked:
    if not topic or not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        # Reset state
        st.session_state.results = None
        st.session_state.error = None
        st.session_state.awaiting_approval = False
        st.session_state.is_loading = True
        start_time = time.time()

        try:
            with st.spinner("Pipeline running — 7 agents working on your topic..."):
                resp = requests.post(
                    f"{BACKEND_URL}/research",
                    json={"topic": topic.strip()},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.results = data
                st.session_state.elapsed_time = time.time() - start_time
                st.session_state.research_id = data.get("research_id")

                if data.get("status") == "awaiting_approval":
                    st.session_state.awaiting_approval = True

        except requests.exceptions.ConnectionError:
            st.session_state.error = (
                "Cannot connect to backend. Is the API running at "
                f"{BACKEND_URL}?\n\nRun: uvicorn backend.main:app "
                "--host 0.0.0.0 --port 8080"
            )
        except requests.exceptions.Timeout:
            st.session_state.error = (
                "Request timed out (120s). Try a narrower topic."
            )
        except requests.exceptions.HTTPError as e:
            st.session_state.error = (
                f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            )
        except Exception as e:
            st.session_state.error = f"Unexpected error: {e}"

        st.session_state.is_loading = False

# ── Error display ─────────────────────────────────────────────────────────

if st.session_state.error:
    st.error(st.session_state.error)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Pipeline Execution Visualizer
# ═══════════════════════════════════════════════════════════════════════════

results: Optional[dict] = st.session_state.results

st.markdown("### Pipeline")

# Agent definitions: (key, icon, display_name)
AGENTS = [
    ("planner",      "🧠", "Planner"),
    ("search",       "🔍", "Search"),
    ("scraper",      "🕷️", "Scraper"),
    ("analysis",     "📊", "Analysis"),
    ("writer",       "✍️",  "Writer"),
    ("critic",       "🔎", "Critic"),
    ("human_review", "👤", "Review"),
    ("publisher",    "📦", "Publisher"),
]

# State fields flowing between agents
ARROWS = [
    "subtasks[]",
    "search_results[]",
    "scraped_content[]",
    "analysis{}",
    "draft_report",
    "critic_feedback{}",
    "decision",
    None,  # no arrow after publisher
]

# Build the horizontal flow as HTML
flow_html = '<div class="agent-flow">'
for i, (key, icon, name) in enumerate(AGENTS):
    css_class = _agent_status_class(key, results)
    metric = _agent_metric(key, results)

    flow_html += f'''
    <div class="agent-card-v {css_class}">
        <div class="icon">{icon}</div>
        <div class="name">{name}</div>
        <div class="metric">{metric}</div>
    </div>'''

    # Arrow between agents
    if i < len(AGENTS) - 1 and i < len(ARROWS) and ARROWS[i]:
        flow_html += f'''
        <div class="arrow-label">
            <span class="arrow">→</span>
            <span class="field">{ARROWS[i]}</span>
        </div>'''

flow_html += '</div>'
st.markdown(flow_html, unsafe_allow_html=True)

# ── Revision loop indicator ───────────────────────────────────────────────

if results and results.get("revision_count", 0) > 0:
    rev_count = results["revision_count"]
    cf = results.get("critic_feedback")
    score = cf.get("quality_score", 0) if cf else 0

    rev_html = '<div style="text-align:center; margin: 8px 0 16px 0;">'
    if rev_count == 1 and cf and cf.get("passed"):
        rev_html += (
            '<span class="revision-badge">'
            f'🔄 1 revision loop · Final score: '
            f'<b style="color:{_score_css(score)}">{score:.2f}</b> · Passed'
            '</span>'
        )
    else:
        rev_html += (
            '<span class="revision-badge">'
            f'🔄 {rev_count} revision loop{"s" if rev_count > 1 else ""} · '
            f'Final score: <b style="color:{_score_css(score)}">{score:.2f}</b>'
            '</span>'
        )
    rev_html += '</div>'
    st.markdown(rev_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3b — Summary Metrics (after pipeline completes)
# ═══════════════════════════════════════════════════════════════════════════

if results is not None:
    c1, c2, c3, c4, c5 = st.columns(5)

    subtask_count = len(results.get("subtasks", []))
    result_count = len(results.get("search_results", []))
    elapsed = st.session_state.elapsed_time or 0
    status_text = results.get("status", "unknown")
    rev = results.get("revision_count", 0)

    with c1:
        st.markdown(
            f'<div class="metric-box"><p class="number">{subtask_count}</p>'
            f'<p class="label">Subtasks</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-box"><p class="number">{result_count}</p>'
            f'<p class="label">Sources Found</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        a = results.get("analysis")
        theme_count = len(a.get("themes", [])) if a else 0
        st.markdown(
            f'<div class="metric-box"><p class="number">{theme_count}</p>'
            f'<p class="label">Themes</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-box"><p class="number">{elapsed:.0f}s</p>'
            f'<p class="label">Pipeline Time</p></div>',
            unsafe_allow_html=True,
        )
    with c5:
        s_display = status_text.replace("_", " ").title()
        s_color = {
            "complete": "#22c55e", "awaiting_approval": "#f59e0b",
            "rejected": "#ef4444", "failed": "#ef4444",
        }.get(status_text, "#9ca3af")
        st.markdown(
            f'<div class="metric-box">'
            f'<p class="number" style="color:{s_color};font-size:1.2rem">'
            f'{s_display}</p>'
            f'<p class="label">Status</p></div>',
            unsafe_allow_html=True,
        )

    # Errors
    errors = results.get("errors", [])
    if errors:
        with st.expander(f"Pipeline Errors ({len(errors)})", expanded=False):
            for err in errors:
                st.warning(err)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Human-in-the-Loop Review
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.awaiting_approval and results:
    research_id = st.session_state.research_id
    draft = results.get("draft_report", "")
    cf = results.get("critic_feedback") or {}
    q_score = cf.get("quality_score", 0.0)
    rev = results.get("revision_count", 0)
    word_count = len(draft.split()) if draft else 0

    st.markdown(
        f'''<div class="hitl-panel">
            <h2>👤 Your Review Required</h2>
            <p class="subtitle">
                The pipeline has completed all automated steps and is waiting
                for your decision before publishing.
            </p>
        </div>''',
        unsafe_allow_html=True,
    )

    # Review metadata
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        cls = _score_color(q_score)
        st.markdown(
            f'<div class="metric-box"><p class="number {cls}">'
            f'{q_score:.2f}</p>'
            f'<p class="label">Quality Score</p></div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f'<div class="metric-box"><p class="number">{word_count:,}</p>'
            f'<p class="label">Words</p></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        rev_text = "First pass" if rev <= 1 else f"{rev - 1} revision{'s' if rev > 2 else ''}"
        st.markdown(
            f'<div class="metric-box"><p class="number">{rev_text}</p>'
            f'<p class="label">Revisions</p></div>',
            unsafe_allow_html=True,
        )

    # Report preview
    if draft:
        st.markdown("**Report Preview**")
        preview = draft[:800] + ("..." if len(draft) > 800 else "")
        st.markdown(preview)

    # Decision buttons
    st.markdown("---")
    bcol1, bcol2, bcol3 = st.columns([2, 1, 2])

    with bcol1:
        if st.button(
            "✅ Approve & Publish",
            type="primary",
            use_container_width=True,
            key="approve_btn",
        ):
            with st.spinner("Publishing report..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/research/{research_id}/approve",
                        timeout=30,
                    )
                    resp.raise_for_status()
                    st.session_state.results = resp.json()
                    st.session_state.awaiting_approval = False
                except requests.exceptions.ConnectionError:
                    st.session_state.error = "Cannot connect to backend."
                except Exception as e:
                    st.session_state.error = f"Approve failed: {e}"

    with bcol3:
        if st.button(
            "❌ Reject",
            use_container_width=True,
            key="reject_btn",
        ):
            with st.spinner("Rejecting..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/research/{research_id}/reject",
                        timeout=30,
                    )
                    resp.raise_for_status()
                    st.session_state.results = resp.json()
                    st.session_state.awaiting_approval = False
                except requests.exceptions.ConnectionError:
                    st.session_state.error = "Cannot connect to backend."
                except Exception as e:
                    st.session_state.error = f"Reject failed: {e}"

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Research Report
# ═══════════════════════════════════════════════════════════════════════════

if results and results.get("draft_report") and not st.session_state.awaiting_approval:
    draft = results["draft_report"]
    cf = results.get("critic_feedback") or {}
    q_score = cf.get("quality_score", 0.0)
    rev = results.get("revision_count", 0)
    word_count = len(draft.split())
    section_count = draft.count("\n## ") + draft.count("\n# ")

    st.markdown("---")
    st.markdown("### 📝 Research Report")

    # Metadata badges
    score_cls = _score_color(q_score)
    rev_text = "Passed on first review" if rev <= 1 else f"Improved over {rev - 1} revision loop{'s' if rev > 2 else ''}"

    badges_html = '<div class="report-meta">'
    badges_html += f'<span class="meta-badge">📄 <span class="val">{word_count:,}</span> words</span>'
    badges_html += f'<span class="meta-badge">📑 <span class="val">{section_count}</span> sections</span>'
    badges_html += (
        f'<span class="meta-badge">Quality '
        f'<span class="val {score_cls}">{q_score:.2f}</span></span>'
    )
    badges_html += f'<span class="meta-badge">🔄 {rev_text}</span>'
    if results.get("published_url"):
        badges_html += '<span class="meta-badge">📦 <span class="val">Published</span></span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

    # Model info
    st.caption(
        "Written by llama-3.1-8b-instant · "
        "Reviewed by llama-3.3-70b-versatile · "
        "Orchestrated by LangGraph"
    )

    # Full report
    st.markdown(draft)

    # Download button
    st.download_button(
        label="Download Report (.md)",
        data=draft,
        file_name=f"newsforge_report_{results.get('research_id', 'report')}.md",
        mime="text/markdown",
    )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Analysis Deep Dive
# ═══════════════════════════════════════════════════════════════════════════

if results and results.get("analysis"):
    analysis = results["analysis"]

    with st.expander("📊 Analysis Deep Dive", expanded=False):
        # Themes
        themes = analysis.get("themes", [])
        if themes:
            st.markdown("#### Themes Identified")
            for theme_item in themes:
                if isinstance(theme_item, dict):
                    theme_name = theme_item.get("theme", "Unknown")
                    confidence = theme_item.get("confidence", 0)
                    facts = theme_item.get("key_facts", [])

                    facts_html = ""
                    for f in facts[:3]:
                        facts_html += f"• {f}<br>"

                    st.markdown(
                        f'''<div class="theme-card">
                            <p class="theme-name">{theme_name}
                                <span style="color:{_score_css(confidence)};
                                font-size:0.8rem; margin-left:8px;">
                                {confidence:.0%} confidence</span>
                            </p>
                            <p class="theme-facts">{facts_html}</p>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                elif isinstance(theme_item, str):
                    st.markdown(f"- {theme_item}")

        # Key facts
        key_facts = analysis.get("key_facts", [])
        if key_facts:
            st.markdown("#### Key Facts")
            for fact in key_facts[:10]:
                st.markdown(f"- {fact}")

        # Contradictions
        contradictions = analysis.get("contradictions", [])
        if contradictions:
            st.markdown("#### Contradictions Found")
            for c in contradictions:
                st.warning(c)

        # Coverage gaps
        gaps = analysis.get("coverage_gaps", [])
        if gaps:
            st.markdown("#### Coverage Gaps")
            for g in gaps:
                st.info(g)

        # Sources
        sources_count = analysis.get("sources_analysed", 0)
        conf = analysis.get("confidence_score", 0)
        st.markdown(
            f"**{sources_count} sources analysed** · "
            f"Overall confidence: **{conf:.2f}**"
        )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Architecture (collapsible)
# ═══════════════════════════════════════════════════════════════════════════

with st.expander("🏗️ Architecture & Tech Stack", expanded=False):
    col_flow, col_stack = st.columns(2)

    with col_flow:
        st.markdown("#### Pipeline Flow")
        st.code("""
User Topic
    │
[Planner]    ← ReAct loop (Groq 70B)
    │ subtasks[]
[Search]     ← Tavily API (parallel)
    │ search_results[]
[Scraper]    ← httpx + BeautifulSoup
    │ scraped_content[]
[Analysis]   ← ReAct loop (Groq 70B)
    │ analysis{}
[Writer]     ← Single-call (Groq 8B)
    │ draft_report
[Critic]     ← Rubric scoring (Groq 70B)
    │ ↺ revision loop if score < 0.75
[Human]      ← LangGraph interrupt()
    │ approve / reject
[Publisher]  ← Local + optional AWS S3
    │
Final Report
""", language=None)

    with col_stack:
        st.markdown("#### Tech Stack")
        st.markdown("""
| Component | Technology |
|---|---|
| Orchestration | LangGraph StateGraph |
| Reasoning LLM | Groq llama-3.3-70b-versatile |
| Writing LLM | Groq llama-3.1-8b-instant |
| Search | Tavily API (parallel async) |
| Scraping | httpx + BeautifulSoup |
| Observability | Langfuse v3 |
| Backend | FastAPI |
| State | SQLite Checkpointer |
| HITL | LangGraph interrupt + Command |
| Frontend | Streamlit |
| Storage | Local + optional AWS S3/DynamoDB |
""")

# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    "Built by Aditya &nbsp;·&nbsp; NewsForge Multi-Agent Research System"
    "</div>",
    unsafe_allow_html=True,
)
