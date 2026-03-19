"""
NewsForge Frontend — Design Decisions
======================================

POLLING ARCHITECTURE (Bug 1 fix)
POST /research now returns immediately with a research_id.
The frontend polls GET /research/{id}/status every 2 seconds.
While polling, the pipeline visualiser advances agent cards
through waiting → running → complete based on elapsed time.
When status becomes "awaiting_approval", the HITL UI appears.
No background threads needed — Streamlit's rerun() cycle IS
the polling loop.

LAYOUT
Single-page vertical scroll: input → pipeline → HITL → results.
Sidebar holds system health (model routing, API keys, history).
No clutter above the fold — metric boxes removed (Bug 2 fix),
replaced with a single status line after completion.
"""

import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark_results"

# FIX 3 — Configurable agent timing (seconds from pipeline start)
AGENT_TIMINGS = {
    "planner":      (0,  12),
    "search":       (12, 16),
    "scraper":      (16, 38),
    "analysis":     (38, 50),
    "writer":       (50, 60),
    "critic":       (60, 72),
    "human_review": (72, 78),
    "publisher":    (78, 85),
}

# FIX 7 — Loading messages matching agent timings
LOADING_MESSAGES = [
    (0,  "Planning research subtasks..."),
    (12, "Searching the web via Tavily..."),
    (16, "Scraping article content..."),
    (38, "Analysing themes and facts..."),
    (50, "Writing your research report..."),
    (60, "Critic reviewing quality..."),
    (72, "Awaiting human review..."),
]

AGENT_DEFS = [
    ("planner",      "Planner",   "Decomposes topic into subtasks"),
    ("search",       "Search",    "Web search via Tavily API"),
    ("scraper",      "Scraper",   "Fetches full page content"),
    ("analysis",     "Analysis",  "Extracts themes & facts"),
    ("writer",       "Writer",    "Composes research report"),
    ("critic",       "Critic",    "Reviews & scores quality"),
    ("human_review", "Review",    "Human approval gate"),
    ("publisher",    "Publisher",  "Saves final report"),
]

DATA_LABELS = [
    "subtasks[]", "results[]", "content[]", "analysis{}",
    "report", "feedback{}", "decision", None,
]


CREDIBLE_ACADEMIC = {
    ".edu", ".gov", ".mil", "nature.com", "science.org", "sciencedirect.com",
    "thelancet.com", "nejm.org", "bmj.com", "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org", "ieee.org", "acm.org", "springer.com", "wiley.com",
    "nih.gov", "who.int", "worldbank.org",
}

CREDIBLE_NEWS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "wsj.com", "ft.com",
    "economist.com", "bloomberg.com", "cnbc.com", "aljazeera.com",
    "npr.org", "pbs.org", "cnn.com", "wired.com", "arstechnica.com",
    "techcrunch.com", "theverge.com", "statnews.com", "healthaffairs.org",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsForge — AI Research Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="N",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "results": None,
    "is_loading": False,
    "error": None,
    "elapsed_time": None,
    "research_id": None,
    "awaiting_approval": False,
    "report_preview": None,
    "quality_score_preview": None,
    "topic_history": [],
    "pipeline_start_time": None,
    "current_topic": None,
    "show_evaluations": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ──────────────────────────────────── */
.stApp { background-color: #0a0e14; }
.block-container { max-width: 1280px; padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0d1117; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Header ────────────────────────────────── */
.nf-header { margin-bottom: 0.5rem; }
.nf-header h1 { font-size: 2rem; font-weight: 800; color: #e6edf3;
    margin: 0; letter-spacing: -0.5px; }
.nf-header .tagline { color: #7d8590; font-size: 0.92rem; margin-top: 2px; }
.nf-status-dot { display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.nf-status-healthy { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.nf-status-down { background: #ef4444; box-shadow: 0 0 6px #ef444488; }

/* ── Topic input area ──────────────────────── */

/* ── Pipeline cards ────────────────────────── */
.pipeline-row { display: flex; align-items: stretch; gap: 0;
    overflow-x: auto; padding: 12px 0; }
.pipe-card { flex: 1 1 0; min-width: 110px; max-width: 160px;
    border-radius: 10px; padding: 14px 10px; text-align: center;
    position: relative; transition: all 0.3s ease; }
.pipe-card .p-name { font-weight: 700; font-size: 0.78rem; color: #e6edf3;
    margin: 4px 0 2px 0; line-height: 1.2; }
.pipe-card .p-desc { font-size: 0.65rem; color: #7d8590; margin: 0;
    line-height: 1.3; }
.pipe-card .p-metric { font-size: 0.72rem; color: #c9d1d9; margin-top: 8px;
    line-height: 1.3; min-height: 1.4em; }
.pipe-card .p-metric b { color: #fff; }
.pipe-card .p-time { font-size: 0.62rem; color: #484f58; margin-top: 4px; }

/* Card states */
.pc-waiting  { background: #161b22; border: 1.5px solid #21262d; }
.pc-running  { background: #0c1929; border: 1.5px solid #1f6feb;
    box-shadow: 0 0 16px rgba(31,111,235,0.2); }
.pc-complete { background: #0d2818; border: 1.5px solid #238636; }
.pc-failed   { background: #2d1215; border: 1.5px solid #da3633; }
.pc-hitl     { background: #2d2305; border: 1.5px solid #d29922;
    box-shadow: 0 0 14px rgba(210,153,34,0.2); }

/* Status indicator dots */
.pc-dot { width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; margin-right: 4px; vertical-align: middle; }
.pc-dot-waiting  { background: #484f58; }
.pc-dot-running  { background: #58a6ff; animation: pulse-blue 1.5s infinite; }
.pc-dot-complete { background: #3fb950; }
.pc-dot-failed   { background: #f85149; }
.pc-dot-hitl     { background: #d29922; animation: pulse-amber 1.5s infinite; }

@keyframes pulse-blue {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px #58a6ff; }
    50% { opacity: 0.5; box-shadow: 0 0 8px #58a6ff; }
}
@keyframes pulse-amber {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px #d29922; }
    50% { opacity: 0.5; box-shadow: 0 0 8px #d29922; }
}

/* Arrow connectors */
.pipe-arrow { flex: 0 0 28px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding-top: 10px; }
.pipe-arrow .arr { color: #30363d; font-size: 0.9rem; }
.pipe-arrow .arr-active { color: #58a6ff; }
.pipe-arrow .arr-done { color: #3fb950; }
.pipe-arrow .lbl { font-size: 0.55rem; color: #484f58; font-family: monospace;
    margin-top: 1px; white-space: nowrap; }

/* ── Metric boxes ──────────────────────────── */
.m-box { background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 16px 12px; text-align: center; }
.m-box .m-num { font-size: 1.7rem; font-weight: 700; color: #e6edf3;
    margin: 0; line-height: 1.2; }
.m-box .m-label { font-size: 0.78rem; color: #7d8590; margin: 4px 0 0 0; }

/* ── Status bar ────────────────────────────── */
.status-bar { background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 16px; margin: 8px 0 12px 0;
    font-size: 0.82rem; color: #8b949e; }
.status-bar b { color: #e6edf3; }

/* ── HITL panel ────────────────────────────── */
.hitl-panel { background: linear-gradient(135deg, #2d2305 0%, #161b22 100%);
    border: 2px solid #d29922; border-radius: 14px; padding: 28px;
    margin: 20px 0; }
.hitl-panel h2 { color: #d29922; margin: 0 0 4px 0; font-size: 1.3rem; }
.hitl-panel .sub { color: #8b949e; font-size: 0.88rem; margin-bottom: 16px; }

/* ── Report section ────────────────────────── */
.report-meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px 0; }
.r-badge { display: inline-flex; align-items: center; gap: 5px;
    background: #161b22; border: 1px solid #21262d; border-radius: 6px;
    padding: 5px 12px; font-size: 0.8rem; color: #8b949e; }
.r-badge .val { font-weight: 700; color: #e6edf3; }

/* ── Quality gauge ─────────────────────────── */
.gauge-container { position: relative; width: 120px; height: 120px;
    margin: 0 auto; }
.gauge-ring { transform: rotate(-90deg); }
.gauge-text { position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%); text-align: center; }
.gauge-score { font-size: 1.6rem; font-weight: 800; line-height: 1; }
.gauge-label { font-size: 0.7rem; color: #7d8590; margin-top: 2px; }

/* ── Theme cards ───────────────────────────── */
.theme-card { background: #161b22; border-radius: 8px; padding: 14px;
    margin-bottom: 8px; border-left: 3px solid #1f6feb; }
.theme-card .t-name { font-weight: 600; color: #e6edf3; font-size: 0.9rem;
    margin: 0 0 6px 0; }
.theme-card .t-facts { color: #8b949e; font-size: 0.8rem; margin: 0;
    line-height: 1.5; }

/* ── Source credibility badges ─────────────── */
.cred-academic { color: #3fb950; }
.cred-news     { color: #58a6ff; }
.cred-other    { color: #8b949e; }

/* ── Benchmark ─────────────────────────────── */
.bm-hero { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d; border-radius: 14px; padding: 24px;
    text-align: center; margin-bottom: 16px; }
.bm-hero .bm-score { font-size: 3rem; font-weight: 800; margin: 0; }
.bm-hero .bm-label { font-size: 0.85rem; color: #7d8590; }

/* ── Revision timeline ─────────────────────── */
.rev-timeline { display: flex; align-items: center; gap: 0;
    margin: 12px 0; flex-wrap: wrap; }
.rev-node { background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 6px 14px; font-size: 0.78rem;
    color: #c9d1d9; white-space: nowrap; }
.rev-arrow { color: #484f58; padding: 0 6px; font-size: 0.85rem; }
.rev-node-pass { border-color: #238636; color: #3fb950; }
.rev-node-fail { border-color: #da3633; color: #f85149; }

/* ── Sidebar ───────────────────────────────── */
.sb-section { background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.sb-title { font-size: 0.82rem; font-weight: 700; color: #e6edf3;
    margin: 0 0 8px 0; }
.sb-row { display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; font-size: 0.78rem; }
.sb-key { color: #7d8590; }
.sb-val { color: #c9d1d9; font-weight: 600; }
.sb-ok  { color: #3fb950; }
.sb-err { color: #f85149; }

/* ── Progress bar ──────────────────────────── */
.progress-outer { width: 100%; height: 4px; background: #21262d;
    border-radius: 2px; overflow: hidden; margin: 8px 0 4px 0; }
.progress-inner { height: 100%; background: linear-gradient(90deg, #1f6feb, #58a6ff);
    border-radius: 2px; transition: width 0.5s ease; }

/* ── Color helpers ─────────────────────────── */
.c-green  { color: #3fb950; }
.c-yellow { color: #d29922; }
.c-red    { color: #f85149; }
.c-blue   { color: #58a6ff; }
.c-muted  { color: #7d8590; }

/* ── Eval panel ───────────────────────────── */
.eval-panel-header { background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
.eval-panel-header .ep-title { font-size: 0.92rem; font-weight: 700;
    color: #e6edf3; margin: 0 0 6px 0; }
.eval-panel-header .ep-desc { font-size: 0.78rem; color: #8b949e;
    line-height: 1.5; margin: 0; }
.eval-divider { border: 0; border-top: 1px solid #21262d; margin: 10px 0; }

/* ── Right eval column — sticky + scrollable ─ */
div[data-testid="stColumn"]:last-child .eval-sticky-wrap {
    position: sticky; top: 1rem; max-height: 95vh;
    overflow-y: auto; padding-right: 4px; }
div[data-testid="stColumn"]:last-child .eval-sticky-wrap::-webkit-scrollbar {
    width: 4px; }
div[data-testid="stColumn"]:last-child .eval-sticky-wrap::-webkit-scrollbar-thumb {
    background: #30363d; border-radius: 2px; }
div[data-testid="stColumn"]:last-child .eval-sticky-wrap::-webkit-scrollbar-track {
    background: transparent; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def score_color(score: float) -> str:
    if score >= 0.8: return "#3fb950"
    if score >= 0.6: return "#d29922"
    return "#f85149"


def score_class(score: float) -> str:
    if score >= 0.8: return "c-green"
    if score >= 0.6: return "c-yellow"
    return "c-red"


def source_credibility(domain: str) -> tuple[str, str]:
    """Return (label, css_class) for a domain."""
    d = domain.lower().removeprefix("www.")
    for pattern in CREDIBLE_ACADEMIC:
        if d.endswith(pattern) or pattern in d:
            return ("Academic", "cred-academic")
    for pattern in CREDIBLE_NEWS:
        if d.endswith(pattern) or pattern in d:
            return ("News", "cred-news")
    return ("Web", "cred-other")


def check_backend_health() -> dict:
    """Quick health check — returns dict with status info."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "unreachable"}


@st.cache_data(ttl=300)
def load_benchmark_data() -> list[dict]:
    """Load and deduplicate benchmark results. Cached for 5 minutes."""
    results = []
    if not BENCHMARK_DIR.exists():
        return results
    incremental_dirs = sorted(BENCHMARK_DIR.glob("*_incremental"), reverse=True)
    if incremental_dirs:
        latest = incremental_dirs[0]
        for f in sorted(latest.glob("topic_*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                continue
    if not results:
        for f in sorted(BENCHMARK_DIR.glob("run_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    results = data
                elif isinstance(data, dict) and "results" in data:
                    results = data["results"]
                break
            except Exception:
                continue
    return results


def load_report_metadata() -> list[dict]:
    """Load all report metadata files from data/reports/."""
    metas = []
    if REPORTS_DIR.exists():
        for f in sorted(REPORTS_DIR.glob("*_metadata.json"), reverse=True):
            try:
                metas.append(json.loads(f.read_text()))
            except Exception:
                continue
    return metas


def make_gauge_svg(score: float, size: int = 120) -> str:
    """Create an SVG circular gauge for a quality score."""
    r = (size - 12) / 2
    cx = cy = size / 2
    circumference = 2 * 3.14159 * r
    fill_pct = min(max(score, 0), 1.0)
    offset = circumference * (1 - fill_pct)
    color = score_color(score)

    return f"""
    <div class="gauge-container" style="width:{size}px;height:{size}px;">
        <svg class="gauge-ring" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle cx="{cx}" cy="{cy}" r="{r}" stroke="#21262d"
                stroke-width="8" fill="none"/>
            <circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color}"
                stroke-width="8" fill="none" stroke-linecap="round"
                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                style="transition: stroke-dashoffset 0.8s ease;"/>
        </svg>
        <div class="gauge-text">
            <div class="gauge-score" style="color:{color}">{score:.2f}</div>
            <div class="gauge-label">Quality</div>
        </div>
    </div>"""


def topic_slug(topic: str) -> str:
    """Create a filesystem-safe slug from a topic string."""
    return re.sub(r"[^a-z0-9]+", "-", topic[:40].lower()).strip("-")


def get_loading_message(elapsed: float) -> str:
    """Return the appropriate loading message for the current elapsed time."""
    msg = LOADING_MESSAGES[0][1]
    for threshold, text in LOADING_MESSAGES:
        if elapsed >= threshold:
            msg = text
    return msg


def _sim_agent_state(elapsed: float, agent_key: str) -> str:
    """Determine simulated state based on elapsed time."""
    start, end = AGENT_TIMINGS.get(agent_key, (999, 999))
    if elapsed < start:
        return "waiting"
    if elapsed < end:
        return "running"
    return "complete"


def _get_real_agent_state(agent_key: str, results: dict) -> str:
    """Determine actual agent state from results."""
    status = results.get("status", "")
    checks = {
        "planner":      lambda r: len(r.get("subtasks", [])) > 0,
        "search":       lambda r: len(r.get("search_results", [])) > 0,
        "scraper":      lambda r: r.get("scraped_content") is not None,
        "analysis":     lambda r: r.get("analysis") is not None,
        "writer":       lambda r: r.get("draft_report") is not None,
        "critic":       lambda r: r.get("critic_feedback") is not None,
        "human_review": lambda r: status in ("awaiting_approval", "complete", "rejected"),
        "publisher":    lambda r: r.get("published_url") is not None,
    }
    if agent_key == "human_review" and status == "awaiting_approval":
        return "hitl"
    check = checks.get(agent_key)
    if check and check(results):
        return "complete"
    errors = results.get("errors", [])
    if any(agent_key in str(e) for e in errors):
        return "failed"
    return "waiting"


def _agent_metric_text(agent_key: str, results: Optional[dict]) -> str:
    """Short metric string for a completed agent card."""
    if not results:
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
            pct = int(ok / len(sc) * 100) if sc else 0
            return f"<b>{ok}/{len(sc)}</b> pages ({pct}%)"
        return ""
    if agent_key == "analysis":
        a = r.get("analysis")
        if a:
            themes = len(a.get("themes", []))
            conf = a.get("confidence_score", 0)
            return f"<b>{themes}</b> themes, {conf:.2f}"
        return ""
    if agent_key == "writer":
        dr = r.get("draft_report")
        if dr:
            words = len(dr.split())
            sections = dr.count("\n## ") + dr.count("\n# ")
            return f"<b>{words:,}</b> words, {sections} sections"
        return ""
    if agent_key == "critic":
        cf = r.get("critic_feedback")
        if cf:
            sc = cf.get("quality_score", 0)
            p = "passed" if cf.get("passed") else "revision"
            cls = score_class(sc)
            return f'<span class="{cls}"><b>{sc:.2f}</b></span> {p}'
        return ""
    if agent_key == "human_review":
        s = r.get("status", "")
        if s == "awaiting_approval":
            return "<b>Awaiting</b>"
        if s == "rejected":
            return "<b>Rejected</b>"
        if s == "complete":
            return "<b>Approved</b>"
        return ""
    if agent_key == "publisher":
        if r.get("published_url"):
            return "<b>Saved</b> to reports/"
        return ""
    return ""


def render_pipeline_cards(
    results: Optional[dict],
    elapsed: float = 0.0,
    is_simulating: bool = False,
    loading_msg: str = "",
) -> str:
    """Build HTML for the pipeline visualiser."""
    html = '<div class="pipeline-row">'

    for i, (key, name, desc) in enumerate(AGENT_DEFS):
        if results and not is_simulating:
            state = _get_real_agent_state(key, results)
        elif is_simulating:
            state = _sim_agent_state(elapsed, key)
        else:
            state = "waiting"

        card_cls = f"pc-{state}"
        dot_cls = f"pc-dot-{state}"

        if results and state == "complete":
            metric = _agent_metric_text(key, results)
        elif state == "hitl":
            metric = "<b>Awaiting review</b>"
        elif state == "running":
            metric = '<span class="c-muted">Processing...</span>'
        else:
            metric = ""

        time_text = ""
        if is_simulating and state == "running":
            start_t = AGENT_TIMINGS[key][0]
            running_for = max(0, elapsed - start_t)
            time_text = f"{running_for:.0f}s"
        elif results and state == "complete" and not is_simulating:
            start_t, end_t = AGENT_TIMINGS[key]
            time_text = f"~{end_t - start_t}s"

        html += f'''
        <div class="pipe-card {card_cls}">
            <div style="margin-bottom:4px;">
                <span class="pc-dot {dot_cls}"></span>
                <span style="font-size:0.65rem;color:#7d8590;">
                    {state.replace("hitl","review").title()}</span>
            </div>
            <div class="p-name">{name}</div>
            <div class="p-desc">{desc}</div>
            <div class="p-metric">{metric}</div>
            <div class="p-time">{time_text}</div>
        </div>'''

        if i < len(AGENT_DEFS) - 1 and i < len(DATA_LABELS) and DATA_LABELS[i]:
            arr_cls = "arr"
            if results and not is_simulating:
                next_key = AGENT_DEFS[i + 1][0]
                next_state = _get_real_agent_state(next_key, results)
                if next_state in ("complete", "hitl"):
                    arr_cls = "arr-done"
                elif next_state == "running":
                    arr_cls = "arr-active"
            elif is_simulating:
                next_key = AGENT_DEFS[i + 1][0]
                next_state = _sim_agent_state(elapsed, next_key)
                if next_state == "complete":
                    arr_cls = "arr-done"
                elif next_state == "running":
                    arr_cls = "arr-active"

            html += f'''
            <div class="pipe-arrow">
                <span class="{arr_cls}">&#x2192;</span>
                <span class="lbl">{DATA_LABELS[i]}</span>
            </div>'''

    html += '</div>'

    if is_simulating:
        pct = min(elapsed / 80 * 100, 95)
        html += f'''
        <div class="progress-outer">
            <div class="progress-inner" style="width:{pct:.0f}%"></div>
        </div>
        <div style="text-align:center;font-size:0.78rem;color:#8b949e;margin-top:4px;">
            {loading_msg} &nbsp;&mdash;&nbsp; {elapsed:.0f}s elapsed
        </div>'''

    return html


# ═══════════════════════════════════════════════════════════════════════════
# API helpers
# ═══════════════════════════════════════════════════════════════════════════

def start_research(topic_text: str) -> str:
    """POST /research — returns research_id immediately."""
    resp = requests.post(
        f"{BACKEND_URL}/research",
        json={"topic": topic_text},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["research_id"]


def poll_status(research_id: str) -> dict:
    """GET /research/{id}/status — returns current status dict."""
    resp = requests.get(
        f"{BACKEND_URL}/research/{research_id}/status",
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark evaluation panel (renders inside the right column)
# ═══════════════════════════════════════════════════════════════════════════

def render_benchmark_panel():
    """Render the benchmark evaluation panel. Called inside the sidebar."""

    # ── Section A — Context header ───────────────────────────────────────
    st.markdown('''
    <div class="eval-panel-header">
        <p class="ep-title">Evaluation Results</p>
        <p class="ep-desc">
            Tested across pre-defined research topics using an independent
            LLM judge. <span style="color:#d29922;">Not related to your
            current query.</span>
        </p>
    </div>''', unsafe_allow_html=True)

    benchmark_data = load_benchmark_data()

    if not benchmark_data:
        st.info(
            "**No evaluation data yet.**\n\n"
            "Run the benchmark:\n\n"
            "```\npython evaluation/benchmark_runner.py --topics 3\n```\n\n"
            "Results appear here automatically."
        )
        return

    successful = [d for d in benchmark_data if d.get("pipeline_success")]
    scores = [d.get("judge_score", 0) for d in successful]
    all_elapsed = [d.get("elapsed_seconds", 0) for d in successful]
    pass_count = len(successful)
    total_count = len(benchmark_data)
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_time = sum(all_elapsed) / len(all_elapsed) if all_elapsed else 0
    scrape_ok_total = sum(d.get("scrape_success_count", 0) for d in successful)
    scrape_total = sum(d.get("scrape_total_count", 0) for d in successful)
    scrape_pct = int(scrape_ok_total / scrape_total * 100) if scrape_total else 0

    # ── Section B — 2×2 headline metrics ─────────────────────────────────
    bm_r1c1, bm_r1c2 = st.columns(2)
    with bm_r1c1:
        st.markdown(
            f'<div class="m-box"><p class="m-num c-green">{pass_count}/{total_count}</p>'
            f'<p class="m-label">Passing</p></div>',
            unsafe_allow_html=True,
        )
    with bm_r1c2:
        sc_c = score_color(avg_score / 10)
        st.markdown(
            f'<div class="m-box"><p class="m-num" style="color:{sc_c}">'
            f'{avg_score:.1f}/10</p>'
            f'<p class="m-label">Avg Score</p></div>',
            unsafe_allow_html=True,
        )
    bm_r2c1, bm_r2c2 = st.columns(2)
    with bm_r2c1:
        st.markdown(
            f'<div class="m-box"><p class="m-num">{avg_time:.0f}s</p>'
            f'<p class="m-label">Avg Time</p></div>',
            unsafe_allow_html=True,
        )
    with bm_r2c2:
        st.markdown(
            f'<div class="m-box"><p class="m-num">{scrape_pct}%</p>'
            f'<p class="m-label">Scrape Rate</p></div>',
            unsafe_allow_html=True,
        )

    # ── Section C — Score bars by difficulty ──────────────────────────────
    diff_groups: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    for d in successful:
        diff = d.get("expected_difficulty", "medium")
        if diff in diff_groups:
            diff_groups[diff].append(d.get("judge_score", 0))
    diff_colors = {"easy": "#3fb950", "medium": "#d29922", "hard": "#f85149"}

    bars_html = '<div style="margin:12px 0;">'
    for diff_label in ["easy", "medium", "hard"]:
        group_scores = diff_groups[diff_label]
        if group_scores:
            avg_g = sum(group_scores) / len(group_scores)
            bar_pct = min(avg_g / 10 * 100, 100)
            color = diff_colors[diff_label]
            bars_html += (
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<span style="width:52px;font-size:0.72rem;color:{color};'
                f'text-align:right;">{diff_label.title()}</span>'
                f'<div style="flex:1;height:8px;background:#21262d;border-radius:4px;">'
                f'<div style="width:{bar_pct}%;height:100%;background:{color};'
                f'border-radius:4px;"></div></div>'
                f'<span style="font-size:0.72rem;color:#c9d1d9;width:32px;">'
                f'{avg_g:.1f}</span></div>'
            )
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    # ── Section D — Compact results list with expandable details ─────────
    st.markdown('<hr class="eval-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82rem;font-weight:700;color:#e6edf3;'
        'margin:4px 0 8px 0;">Topic Results</div>',
        unsafe_allow_html=True,
    )

    sorted_results = sorted(
        successful, key=lambda x: x.get("judge_score", 0), reverse=True,
    )
    for idx, d in enumerate(sorted_results):
        topic_name = d.get("topic", "Unknown")
        js = d.get("judge_score", 0)
        elapsed_s = d.get("elapsed_seconds", 0)
        diff = d.get("expected_difficulty", "")
        diff_c = diff_colors.get(diff, "#7d8590")
        js_c = score_color(js / 10)
        icon = "✅" if js >= 6 else "❌"

        with st.expander(
            f"{icon} {topic_name[:28]}  {js:.1f}  {elapsed_s:.0f}s  {diff}",
            expanded=False,
        ):
            st.markdown(f"**{topic_name}**")

            # 5-dimension score breakdown
            judge_output = d.get("judge_output", {}) or {}
            dim_scores = judge_output.get("dimension_scores", judge_output.get("scores", {}))

            dimensions = [
                ("Research Depth", "research_depth"),
                ("Source Diversity", "source_diversity"),
                ("Topic Coverage", "topic_coverage"),
                ("Factual Coherence", "factual_coherence"),
                ("Report Quality", "report_quality"),
            ]

            has_dims = dim_scores and any(dim_scores.get(k, 0) for _, k in dimensions)
            if has_dims:
                dim_html = ''
                for label, key in dimensions:
                    val = dim_scores.get(key, 0)
                    filled = int(val)
                    empty = 10 - filled
                    bar = (
                        f'<span style="color:{score_color(val / 10)};">'
                        f'{"█" * filled}{"░" * empty}</span>'
                    )
                    dim_html += (
                        f'<div style="font-size:0.72rem;display:flex;gap:6px;padding:1px 0;">'
                        f'<span style="color:#7d8590;min-width:108px;">{label}</span>'
                        f'{bar} '
                        f'<span style="color:#c9d1d9;font-weight:600;">{val:.1f}</span></div>'
                    )
                st.markdown(dim_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#8b949e;">Overall: '
                    f'<b style="color:{js_c}">{js:.1f}</b>/10</div>',
                    unsafe_allow_html=True,
                )

            # Strengths and weaknesses
            strengths = judge_output.get("strengths", [])
            weaknesses = judge_output.get("weaknesses", [])
            sw_html = ''
            for s in strengths[:3]:
                sw_html += f'<div style="font-size:0.72rem;color:#3fb950;">✓ {s}</div>'
            for w in weaknesses[:3]:
                sw_html += f'<div style="font-size:0.72rem;color:#d29922;">⚠ {w}</div>'
            if sw_html:
                st.markdown(sw_html, unsafe_allow_html=True)

            # Revision info
            rev_n = d.get("revision_count", 0)
            if rev_n and rev_n > 1:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#58a6ff;">'
                    f'🔄 Revised {rev_n - 1} time{"s" if rev_n > 2 else ""}</div>',
                    unsafe_allow_html=True,
                )

            # Report download link
            report_file = d.get("report_file", "")
            if report_file:
                rpath = REPORTS_DIR / report_file
                if rpath.exists():
                    st.download_button(
                        "View Report",
                        data=rpath.read_text(encoding="utf-8"),
                        file_name=report_file,
                        mime="text/markdown",
                        key=f"dl_ep_{idx}",
                    )

    # ── Section E — Key Findings ─────────────────────────────────────────
    if len(successful) >= 2:
        st.markdown('<hr class="eval-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.82rem;font-weight:700;color:#e6edf3;'
            'margin:4px 0 6px 0;">Key Findings</div>',
            unsafe_allow_html=True,
        )

        insights: list[str] = []

        easy_scores = diff_groups.get("easy", [])
        hard_scores = diff_groups.get("hard", [])
        if easy_scores and hard_scores:
            avg_e = sum(easy_scores) / len(easy_scores)
            avg_h = sum(hard_scores) / len(hard_scores)
            insights.append(f"Hard topics scored {avg_h:.1f} vs easy {avg_e:.1f}")

        revised = [d for d in successful if (d.get("revision_count") or 0) > 1]
        if revised:
            insights.append(
                f"{len(revised)} report{'s' if len(revised) > 1 else ''} "
                f"improved via revision loop"
            )

        sorted_time = sorted(successful, key=lambda x: x.get("elapsed_seconds", 0))
        if len(sorted_time) >= 2:
            insights.append(
                f"Fastest: {sorted_time[0].get('elapsed_seconds', 0):.0f}s | "
                f"Slowest: {sorted_time[-1].get('elapsed_seconds', 0):.0f}s"
            )

        analysis_ok = sum(
            1 for d in successful
            if (d.get("analysis_confidence") or d.get("critic_score") or 0) > 0.5
        )
        if analysis_ok:
            insights.append(f"Analysis succeeded on {analysis_ok}/{len(successful)} topics")

        for ins in insights:
            st.markdown(
                f'<div style="background:#0c1929;border:1px solid #1f6feb;'
                f'border-radius:6px;padding:8px 10px;margin-bottom:4px;'
                f'font-size:0.72rem;color:#c9d1d9;">{ins}</div>',
                unsafe_allow_html=True,
            )

    # ── Section F — Known Limitation ─────────────────────────────────────
    with st.expander("Known Limitation", expanded=False):
        st.markdown(
            '<div style="font-size:0.75rem;color:#8b949e;line-height:1.6;">'
            'Analysis agent had binary encoding failures on some benchmark '
            'topics when processing PDF/academic content. The Writer + Critic '
            'loop compensated — all topics still passed the judge threshold.'
            '<br><br>'
            '<b style="color:#c9d1d9;">Root cause:</b> Llama 4 Scout returns '
            'compressed binary data on prompts containing mixed-encoding content.'
            '<br>'
            '<b style="color:#c9d1d9;">Fix:</b> 8K char corpus limit + binary '
            'response detection added to Analysis agent.</div>',
            unsafe_allow_html=True,
        )

    # ── Section G — Evaluation methodology ───────────────────────────────
    with st.expander("Evaluation Methodology", expanded=False):
        st.markdown(
            '<div style="font-size:0.75rem;color:#8b949e;line-height:1.6;">'
            '<b style="color:#c9d1d9;">Judge model:</b> llama-3.1-8b-instant '
            '(temp 0.1)<br>'
            '<b style="color:#c9d1d9;">Dimensions:</b> Research Depth, Source '
            'Diversity, Topic Coverage, Factual Coherence, Report Quality<br>'
            '<b style="color:#c9d1d9;">Pass threshold:</b> 6.0/10<br>'
            '<b style="color:#c9d1d9;">Run separately</b> from live pipeline.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Section H — Report Comparison ────────────────────────────────────
    render_report_comparison()


def render_report_comparison():
    """Render report comparison inside the sidebar evaluation panel."""
    report_metas_all = load_report_metadata()
    real_reports = [m for m in report_metas_all if m.get("word_count", 0) > 200]

    if len(real_reports) < 2:
        return

    st.markdown('<hr class="eval-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.82rem;font-weight:700;color:#e6edf3;'
        'margin:4px 0 8px 0;">Report Comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="font-size:0.72rem;color:#7d8590;">'
        'Compare two reports side by side</span>',
        unsafe_allow_html=True,
    )

    report_labels = [
        f"{m.get('topic', 'Unknown')[:30]} ({m.get('quality_score', 0):.2f})"
        for m in real_reports[:20]
    ]

    sel1 = st.selectbox("Report A", report_labels, index=0, key="cmp_a")
    sel2 = st.selectbox(
        "Report B", report_labels,
        index=min(1, len(report_labels) - 1), key="cmp_b",
    )

    if sel1 and sel2:
        idx1 = report_labels.index(sel1)
        idx2 = report_labels.index(sel2)

        for meta, label in [(real_reports[idx1], "A"), (real_reports[idx2], "B")]:
            qs = meta.get("quality_score", 0)
            st.markdown(
                f'<div class="m-box" style="margin-bottom:8px;">'
                f'<p class="m-label">Report {label}</p>'
                f'<p style="color:#c9d1d9;font-size:0.78rem;margin:4px 0;">'
                f'{meta.get("topic", "Unknown")[:40]}</p>'
                f'<p class="m-num" style="color:{score_color(qs)};font-size:1.2rem;">'
                f'{qs:.2f}</p>'
                f'<p class="m-label">{meta.get("word_count", 0):,} words</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            report_file = REPORTS_DIR / meta.get("report_file", "")
            if report_file.exists():
                content = report_file.read_text(encoding="utf-8")
                with st.expander(f"Report {label} Preview", expanded=False):
                    st.markdown(
                        content[:2000] + ("\n\n*... truncated*" if len(content) > 2000 else "")
                    )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — System Health
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sb-title" style="font-size:1rem;margin-bottom:12px;">'
        'NewsForge</div>',
        unsafe_allow_html=True,
    )

    health = check_backend_health()
    is_healthy = health.get("status") == "healthy"
    status_dot = "nf-status-healthy" if is_healthy else "nf-status-down"
    status_text = "Operational" if is_healthy else "Unreachable"

    st.markdown(f'''
    <div class="sb-section">
        <div class="sb-title">
            <span class="nf-status-dot {status_dot}"></span> System Status
        </div>
        <div class="sb-row">
            <span class="sb-key">Backend</span>
            <span class="{"sb-ok" if is_healthy else "sb-err"}">{status_text}</span>
        </div>
        <div class="sb-row">
            <span class="sb-key">Endpoint</span>
            <span class="sb-val" style="font-size:0.7rem;">{BACKEND_URL}</span>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown('''
    <div class="sb-section">
        <div class="sb-title">Model Routing</div>
        <div style="font-size:0.72rem;color:#7d8590;margin-bottom:6px;">
            Two-pool rate dispersion strategy</div>
        <div class="sb-row">
            <span class="sb-key">Pool A (Reasoning)</span>
            <span class="sb-val" style="font-size:0.7rem;">Scout 17B</span>
        </div>
        <div style="font-size:0.68rem;color:#484f58;padding-left:8px;margin-bottom:4px;">
            Planner, Analysis, Critic &middot; 30K TPM</div>
        <div class="sb-row">
            <span class="sb-key">Pool B (Execution)</span>
            <span class="sb-val" style="font-size:0.7rem;">Llama 8B</span>
        </div>
        <div style="font-size:0.68rem;color:#484f58;padding-left:8px;">
            Writer, Judge &middot; 6K TPM</div>
    </div>''', unsafe_allow_html=True)

    from config.settings import (
        GROQ_API_KEY, TAVILY_API_KEY,
        LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    )
    groq_ok = bool(GROQ_API_KEY)
    tavily_ok = bool(TAVILY_API_KEY)
    langfuse_ok = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

    def _api_badge(ok: bool) -> str:
        return f'<span class="{"sb-ok" if ok else "sb-err"}">{"Connected" if ok else "Missing key"}</span>'

    st.markdown(f'''
    <div class="sb-section">
        <div class="sb-title">API Keys</div>
        <div class="sb-row">
            <span class="sb-key">Groq</span>
            {_api_badge(groq_ok)}
        </div>
        <div class="sb-row">
            <span class="sb-key">Tavily</span>
            {_api_badge(tavily_ok)}
        </div>
        <div class="sb-row">
            <span class="sb-key">Langfuse</span>
            {_api_badge(langfuse_ok)}
        </div>
    </div>''', unsafe_allow_html=True)

    report_metas = load_report_metadata()
    report_count = len(report_metas)
    last_run = ""
    if report_metas:
        try:
            last_run = report_metas[0].get("published_at", "")[:19].replace("T", " ")
        except Exception:
            pass

    st.markdown(f'''
    <div class="sb-section">
        <div class="sb-title">History</div>
        <div class="sb-row">
            <span class="sb-key">Total Reports</span>
            <span class="sb-val">{report_count}</span>
        </div>
        <div class="sb-row">
            <span class="sb-key">Last Run</span>
            <span class="sb-val" style="font-size:0.68rem;">
                {last_run or "N/A"}</span>
        </div>
    </div>''', unsafe_allow_html=True)

    if st.session_state.topic_history:
        st.markdown(
            '<div class="sb-section"><div class="sb-title">Recent Topics</div>',
            unsafe_allow_html=True,
        )
        for t in st.session_state.topic_history[-5:]:
            st.markdown(
                f'<div style="font-size:0.72rem;color:#8b949e;'
                f'padding:2px 0;border-bottom:1px solid #21262d;">'
                f'{t[:50]}{"..." if len(t) > 50 else ""}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Layout — Two-column: main content + right evaluation panel
# ═══════════════════════════════════════════════════════════════════════════
_show_evals = st.session_state.show_evaluations

if _show_evals:
    _main_col, _eval_col = st.columns([55, 45])
else:
    _main_col, _eval_col = st.columns([97, 3])

with _eval_col:
    if st.button(
        "Close" if _show_evals else "Evaluations",
        key="eval_toggle",
        use_container_width=True,
        help="Toggle benchmark evaluation results panel",
    ):
        st.session_state.show_evaluations = not _show_evals
        st.rerun()

    if _show_evals:
        st.markdown('<div class="eval-sticky-wrap">', unsafe_allow_html=True)
        render_benchmark_panel()
        st.markdown('</div>', unsafe_allow_html=True)

# Enter main column context so all subsequent st.xxx() calls render there.
# Using __enter__ directly avoids re-indenting 800+ lines of existing code.
# Safe because st.rerun() / st.stop() restart the script from scratch,
# resetting the container stack — no cleanup needed.
_main_col.__enter__()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Header
# ═══════════════════════════════════════════════════════════════════════════
health_dot = "nf-status-healthy" if is_healthy else "nf-status-down"
st.markdown(f'''
<div class="nf-header">
    <h1><span class="nf-status-dot {health_dot}"></span>NewsForge</h1>
    <div class="tagline">
        7-agent autonomous research pipeline &nbsp;&middot;&nbsp;
        LangGraph &middot; Groq &middot; Tavily &middot; Langfuse
    </div>
</div>''', unsafe_allow_html=True)

# FIX 5 — Stop if backend is unreachable
if not is_healthy:
    st.error(
        "**Backend not running.** Start it with:\n\n"
        "```\nuvicorn backend.main:app --reload --port 8080\n```"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Research Input
# ═══════════════════════════════════════════════════════════════════════════

col_input, col_run = st.columns([7, 1])
with col_input:
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Impact of AI on healthcare, CRISPR breakthroughs, Semiconductor supply chain risks",
        max_chars=500,
        label_visibility="collapsed",
        key="topic_input",
    )
with col_run:
    run_clicked = st.button(
        "Run Research",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_loading,
        key="run_btn",
    )

if topic:
    st.markdown(
        f'<div style="text-align:right;font-size:0.7rem;color:#484f58;'
        f'margin-top:-10px;">{len(topic)}/500</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Handle Run button — non-blocking POST /research
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    if not topic or not topic.strip():
        st.warning("Enter a research topic to begin.")
    else:
        st.session_state.results = None
        st.session_state.error = None
        st.session_state.awaiting_approval = False
        st.session_state.report_preview = None
        st.session_state.quality_score_preview = None
        st.session_state.current_topic = topic.strip()

        # Add to history
        history = st.session_state.topic_history
        clean_topic = topic.strip()
        if clean_topic not in history:
            history.append(clean_topic)
            st.session_state.topic_history = history[-5:]

        try:
            rid = start_research(clean_topic)
            st.session_state.research_id = rid
            st.session_state.is_loading = True
            st.session_state.pipeline_start_time = time.time()
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.session_state.error = (
                f"Cannot connect to backend at {BACKEND_URL}.\n\n"
                "Start the API server:\n"
                "  `uvicorn backend.main:app --host 0.0.0.0 --port 8080`"
            )
        except requests.exceptions.HTTPError as e:
            st.session_state.error = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
        except Exception as e:
            st.session_state.error = f"Failed to start research: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Live Pipeline Visualiser with Polling
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Pipeline")

results: Optional[dict] = st.session_state.results

if st.session_state.is_loading and st.session_state.research_id:
    # ── Polling loop ─────────────────────────────────────────────────────
    elapsed = time.time() - (st.session_state.pipeline_start_time or time.time())
    loading_msg = get_loading_message(elapsed)

    # Show simulated pipeline animation
    st.markdown(
        render_pipeline_cards(None, elapsed=elapsed, is_simulating=True, loading_msg=loading_msg),
        unsafe_allow_html=True,
    )

    # Poll the backend for actual status
    try:
        status_data = poll_status(st.session_state.research_id)
        poll_status_val = status_data.get("status", "running")

        if poll_status_val == "awaiting_approval":
            # Pipeline paused at HITL — stop polling, show review UI
            st.session_state.is_loading = False
            st.session_state.awaiting_approval = True
            st.session_state.elapsed_time = elapsed
            st.session_state.report_preview = status_data.get("report_preview")
            st.session_state.quality_score_preview = status_data.get("quality_score")
            # Store preview data so HITL UI can render
            st.session_state.results = {
                "status": "awaiting_approval",
                "research_id": st.session_state.research_id,
                "topic": status_data.get("topic", ""),
                "report_preview": status_data.get("report_preview"),
                "quality_score": status_data.get("quality_score"),
                "word_count": status_data.get("word_count"),
                "revision_count": status_data.get("revision_count"),
            }
            st.rerun()

        elif poll_status_val == "complete":
            # Pipeline completed without HITL pause (edge case)
            st.session_state.is_loading = False
            st.session_state.elapsed_time = elapsed
            st.rerun()

        elif poll_status_val == "failed":
            st.session_state.is_loading = False
            st.session_state.error = status_data.get("error", "Pipeline failed — check server logs.")
            st.rerun()

        else:
            # Still running — wait and poll again
            time.sleep(2)
            st.rerun()

    except requests.exceptions.ConnectionError:
        st.session_state.is_loading = False
        st.session_state.error = "Lost connection to backend during pipeline execution."
        st.rerun()
    except Exception as e:
        st.session_state.is_loading = False
        st.session_state.error = f"Polling error: {e}"
        st.rerun()

else:
    # Static view — show results or empty pipeline
    st.markdown(
        render_pipeline_cards(results),
        unsafe_allow_html=True,
    )

# Error display
if st.session_state.error:
    st.error(st.session_state.error)

# ── BUG 2 FIX — Single status bar instead of metric boxes ───────────────
if results and results.get("status") in ("complete", "rejected") and not st.session_state.is_loading:
    r = results
    topic_display = r.get("topic", "")[:50]
    subtask_count = len(r.get("subtasks", []))
    result_count = len(r.get("search_results", []))
    cf = r.get("critic_feedback") or {}
    q_score = cf.get("quality_score", 0.0)
    elapsed_display = st.session_state.elapsed_time or 0
    status_val = r.get("status", "")
    status_color = "#3fb950" if status_val == "complete" else "#f85149"

    st.markdown(
        f'<div class="status-bar">'
        f'<b>{topic_display}</b> &nbsp;&middot;&nbsp; '
        f'{subtask_count} subtasks &nbsp;&middot;&nbsp; '
        f'{result_count} sources &nbsp;&middot;&nbsp; '
        f'Quality <b style="color:{score_color(q_score)}">{q_score:.2f}</b> &nbsp;&middot;&nbsp; '
        f'{elapsed_display:.0f}s &nbsp;&middot;&nbsp; '
        f'<span style="color:{status_color}">{status_val.replace("_"," ").title()}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    errors = r.get("errors", [])
    if errors:
        with st.expander(f"Pipeline Errors ({len(errors)})", expanded=True):
            for err in errors:
                st.error(err)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — HITL Review Panel
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.awaiting_approval and results:
    research_id = st.session_state.research_id
    report_preview = results.get("report_preview") or ""
    q_score = results.get("quality_score", 0.0) or 0.0
    word_count = results.get("word_count", 0) or 0
    rev = results.get("revision_count", 0) or 0
    elapsed_display = st.session_state.elapsed_time or 0

    st.markdown(f'''
    <div class="hitl-panel">
        <h2>Pipeline Paused — Your Review Required</h2>
        <p class="sub">
            The pipeline ran for {elapsed_display:.0f}s and is waiting for your decision.
            Review the report preview below and approve or reject.
        </p>
    </div>''', unsafe_allow_html=True)

    # Review metrics with gauge
    mc1, mc2, mc3 = st.columns([1.5, 1, 1])
    with mc1:
        st.markdown(make_gauge_svg(q_score, 130), unsafe_allow_html=True)
    with mc2:
        st.markdown(
            f'<div class="m-box"><p class="m-num">{word_count:,}</p>'
            f'<p class="m-label">Words</p></div>',
            unsafe_allow_html=True,
        )
    with mc3:
        rev_text = "First pass" if rev <= 1 else f"{rev - 1} revision{'s' if rev > 2 else ''}"
        st.markdown(
            f'<div class="m-box"><p class="m-num">{rev_text}</p>'
            f'<p class="m-label">Revisions</p></div>',
            unsafe_allow_html=True,
        )

    # Report preview
    if report_preview:
        st.markdown("**Report Preview**")
        preview = report_preview[:1200] + (
            "\n\n*... (truncated for preview)*" if len(report_preview) > 1200 else ""
        )
        st.markdown(preview)

    # Decision buttons
    st.markdown("---")
    bcol1, bcol2, bcol3 = st.columns([2, 1, 2])

    with bcol1:
        if st.button(
            "Approve & Publish",
            type="primary",
            use_container_width=True,
            key="approve_btn",
        ):
            with st.spinner("Publishing report..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/research/{research_id}/approve",
                        timeout=60,
                    )
                    resp.raise_for_status()
                    st.session_state.results = resp.json()
                    st.session_state.awaiting_approval = False
                    st.session_state.elapsed_time = (
                        time.time() - (st.session_state.pipeline_start_time or time.time())
                    )
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.session_state.error = "Cannot connect to backend."
                except Exception as e:
                    st.session_state.error = f"Approve failed: {e}"

    with bcol3:
        if st.button(
            "Reject",
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
                    st.session_state.error = "Report rejected — not published."
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.session_state.error = "Cannot connect to backend."
                except Exception as e:
                    st.session_state.error = f"Reject failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Research Report (after completion)
# ═══════════════════════════════════════════════════════════════════════════

def extract_executive_summary(report: str) -> str:
    """Extract text between ## Executive Summary and next ## heading."""
    lines = report.split('\n')
    in_summary = False
    summary_lines: list[str] = []
    for line in lines:
        if '## Executive Summary' in line or '## executive summary' in line.lower():
            in_summary = True
            continue
        if in_summary and line.startswith('## '):
            break
        if in_summary:
            summary_lines.append(line)
    return '\n'.join(summary_lines).strip()


def extract_key_findings_preview(report: str) -> list[str]:
    """Extract first 3 bullet points from ## Key Findings section."""
    lines = report.split('\n')
    in_findings = False
    bullets: list[str] = []
    for line in lines:
        if '## Key Findings' in line or '## key findings' in line.lower():
            in_findings = True
            continue
        if in_findings and line.startswith('## '):
            break
        if in_findings and line.strip().startswith(('-', '*', '•')):
            bullets.append(line.strip().lstrip('-*• '))
            if len(bullets) >= 3:
                break
    return bullets


if results and results.get("draft_report") and not st.session_state.awaiting_approval:
    draft = results["draft_report"]
    cf = results.get("critic_feedback") or {}
    q_score = cf.get("quality_score", 0.0)
    rev = results.get("revision_count", 0)
    word_count = len(draft.split())
    section_count = draft.count("\n## ") + draft.count("\n# ")
    status = results.get("status", "")

    st.markdown("---")
    st.markdown("### Research Report")

    # ── Metadata badges ──────────────────────────────────────────────────
    rev_text = (
        "First review pass" if rev <= 1
        else f"{rev - 1} revision{'s' if rev > 2 else ''}"
    )
    badges = [
        f'<span class="r-badge">Quality <span class="val" style="color:{score_color(q_score)}">{q_score:.2f}</span></span>',
        f'<span class="r-badge">Words <span class="val">{word_count:,}</span></span>',
        f'<span class="r-badge">Sections <span class="val">{section_count}</span></span>',
    ]
    if results.get("published_url"):
        badges.append('<span class="r-badge"><span class="val c-green">Published</span></span>')
    elif status == "rejected":
        badges.append('<span class="r-badge"><span class="val c-red">Rejected</span></span>')

    st.markdown(
        f'<div class="report-meta">{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Written by Llama 3.1 8B (Pool B) · "
        "Reviewed by Llama 4 Scout 17B (Pool A) · "
        "Orchestrated by LangGraph"
    )

    # ── Revision timeline (inline) ───────────────────────────────────────
    if rev and rev > 0:
        cf_data = results.get("critic_feedback") or {}
        passed = cf_data.get("passed", False)
        rev_score = cf_data.get("quality_score", 0)

        rev_html = '<div class="rev-timeline" style="margin:4px 0 12px 0;">'
        if rev == 1 and passed:
            rev_html += '<span class="rev-node">Draft v1</span>'
            rev_html += '<span class="rev-arrow">&rarr;</span>'
            rev_html += f'<span class="rev-node rev-node-pass">Critic: {rev_score:.2f} Passed</span>'
        else:
            rev_html += '<span class="rev-node">Draft v1</span>'
            rev_html += '<span class="rev-arrow">&rarr;</span>'
            rev_html += '<span class="rev-node rev-node-fail">Revision needed</span>'
            for r in range(1, rev):
                rev_html += '<span class="rev-arrow">&rarr;</span>'
                rev_html += f'<span class="rev-node">Draft v{r + 1}</span>'
            rev_html += '<span class="rev-arrow">&rarr;</span>'
            cls = "rev-node-pass" if passed else "rev-node-fail"
            rev_html += f'<span class="rev-node {cls}">Critic: {rev_score:.2f} {"Passed" if passed else "Failed"}</span>'
        rev_html += '</div>'
        st.markdown(rev_html, unsafe_allow_html=True)

    # ── Executive Summary (always visible) ───────────────────────────────
    exec_summary = extract_executive_summary(draft)
    key_findings = extract_key_findings_preview(draft)

    if exec_summary:
        st.markdown("#### Executive Summary")
        st.markdown(exec_summary)
    else:
        # Fallback: show first 500 chars of the report
        st.markdown("#### Summary")
        fallback = draft[:500].rsplit('\n', 1)[0]
        st.markdown(fallback + "...")

    if key_findings:
        st.markdown("**Key Findings:**")
        for bullet in key_findings:
            st.markdown(f"- {bullet}")

    # ── Download + Copy buttons (always visible) ─────────────────────────
    t_slug = topic_slug(results.get("topic", "report"))
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns([1, 1, 1, 1])
    with dl_col1:
        st.download_button(
            label="Download Markdown",
            data=draft,
            file_name=f"newsforge-{t_slug}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl_col2:
        plain = re.sub(r'[#*_`\[\]()]', '', draft)
        plain = re.sub(r'\n{3,}', '\n\n', plain)
        st.download_button(
            label="Download Plain Text",
            data=plain,
            file_name=f"newsforge-{t_slug}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl_col3:
        search_res = results.get("search_results", [])
        if search_res:
            buf = io.StringIO()
            csv_writer = csv.DictWriter(buf, fieldnames=[
                "result_id", "subtask_id", "title", "url",
                "snippet", "relevance_score", "source_domain",
            ])
            csv_writer.writeheader()
            for sr in search_res:
                csv_writer.writerow({
                    "result_id": sr.get("result_id", ""),
                    "subtask_id": sr.get("subtask_id", ""),
                    "title": sr.get("title", ""),
                    "url": sr.get("url", ""),
                    "snippet": sr.get("snippet", "")[:200],
                    "relevance_score": sr.get("relevance_score", 0),
                    "source_domain": sr.get("source_domain", ""),
                })
            st.download_button(
                label="Download Sources CSV",
                data=buf.getvalue(),
                file_name=f"newsforge-{t_slug}-sources.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("No Sources", disabled=True, use_container_width=True)
    with dl_col4:
        rid = results.get("research_id", "")
        if rid:
            from config.settings import LANGFUSE_HOST
            trace_url = f"{LANGFUSE_HOST}/trace/{rid}"
            st.link_button("View Langfuse Trace", trace_url, use_container_width=True)

    # ── Full Report (collapsed by default) ───────────────────────────────
    with st.expander("Read Full Report", expanded=False):
        st.markdown(draft)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Deep Dive (three separate expanders, all collapsed)
# ═══════════════════════════════════════════════════════════════════════════
if results and (results.get("analysis") or results.get("search_results")):
    st.markdown("---")

    # ── EXPANDER 1 — Analysis Deep Dive ──────────────────────────────────
    with st.expander("Analysis Deep Dive", expanded=False):
        analysis = results.get("analysis")
        if analysis:
            dd_left, dd_right = st.columns([3, 2])

            with dd_left:
                themes = analysis.get("themes", [])
                if themes:
                    st.markdown("#### Themes Identified")
                    for theme_item in themes:
                        if isinstance(theme_item, dict):
                            theme_name = theme_item.get("theme", "Unknown")
                            confidence = theme_item.get("confidence", 0)
                            facts = theme_item.get("key_facts", [])
                            facts_html = "".join(f"&bull; {f}<br>" for f in facts[:3])
                            source_n = theme_item.get("source_count", "")
                            source_line = (
                                f'<div style="font-size:0.68rem;color:#484f58;margin-top:4px;">'
                                f'Supported by {source_n} sources</div>'
                                if source_n else ""
                            )
                            bar_pct = confidence * 100
                            bar_color = score_color(confidence)

                            st.markdown(f'''
                            <div class="theme-card">
                                <p class="t-name">{theme_name}
                                    <span style="color:{bar_color};font-size:0.78rem;
                                    margin-left:8px;">{confidence:.0%}</span>
                                </p>
                                <div style="height:3px;background:#21262d;border-radius:2px;
                                    margin:4px 0 8px 0;">
                                    <div style="height:100%;width:{bar_pct}%;
                                        background:{bar_color};border-radius:2px;"></div>
                                </div>
                                <p class="t-facts">{facts_html}</p>
                                {source_line}
                            </div>''', unsafe_allow_html=True)
                        elif isinstance(theme_item, str):
                            st.markdown(f"- {theme_item}")

                # Contradictions
                contradictions = analysis.get("contradictions", [])
                if contradictions:
                    st.markdown("#### Contradictions Found")
                    for c in contradictions:
                        if isinstance(c, dict):
                            claim_a = c.get("claim_a", c.get("claim", str(c)))
                            claim_b = c.get("claim_b", "")
                            severity = c.get("severity", "minor")
                            sev_color = "#f85149" if severity == "major" else "#d29922"
                            st.markdown(
                                f'<div style="background:#161b22;border-left:3px solid {sev_color};'
                                f'border-radius:4px;padding:10px;margin-bottom:6px;">'
                                f'<div style="font-size:0.78rem;color:#c9d1d9;">{claim_a}</div>'
                                + (f'<div style="font-size:0.72rem;color:#7d8590;margin-top:4px;">'
                                   f'vs. {claim_b}</div>' if claim_b else '')
                                + f'<span style="font-size:0.65rem;color:{sev_color};'
                                f'text-transform:uppercase;font-weight:700;">{severity}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.warning(str(c))

            with dd_right:
                # Stats box
                themes_count = len(analysis.get("themes", []))
                key_facts_count = len(analysis.get("key_facts", []))
                contra_count = len(analysis.get("contradictions", []))
                conf = analysis.get("confidence_score", 0)
                sources_count = analysis.get("sources_analysed", 0)

                st.markdown(f'''
                <div style="background:#161b22;border:1px solid #21262d;
                    border-radius:10px;padding:16px;">
                    <div style="font-size:0.82rem;font-weight:700;color:#e6edf3;
                        margin-bottom:10px;">Analysis Stats</div>
                    <div class="sb-row"><span class="sb-key">Themes</span>
                        <span class="sb-val">{themes_count}</span></div>
                    <div class="sb-row"><span class="sb-key">Key Facts</span>
                        <span class="sb-val">{key_facts_count}</span></div>
                    <div class="sb-row"><span class="sb-key">Contradictions</span>
                        <span class="sb-val">{contra_count}</span></div>
                    <div class="sb-row"><span class="sb-key">Confidence</span>
                        <span class="sb-val" style="color:{score_color(conf)}">{conf:.2f}</span></div>
                    <div class="sb-row"><span class="sb-key">Sources Analysed</span>
                        <span class="sb-val">{sources_count}</span></div>
                </div>''', unsafe_allow_html=True)

                # Coverage gaps
                gaps = analysis.get("coverage_gaps", analysis.get("coverage_notes", []))
                if gaps:
                    st.markdown(
                        '<div style="margin-top:12px;font-size:0.82rem;'
                        'font-weight:700;color:#d29922;">Coverage Gaps</div>',
                        unsafe_allow_html=True,
                    )
                    for g in gaps:
                        st.markdown(f"- {g}")
        else:
            st.info("No analysis data available.")

    # ── EXPANDER 2 — Sources & Search Results ────────────────────────────
    with st.expander("Sources & Search Results", expanded=False):
        search_results = results.get("search_results", [])
        scraped = results.get("scraped_content", []) or []
        subtasks_list = results.get("subtasks", [])

        if search_results:
            # Summary line
            scrape_ok = sum(1 for s in scraped if s.get("scrape_status") == "success")
            st.markdown(
                f"**{len(search_results)}** sources searched across "
                f"**{len(subtasks_list)}** subtasks. "
                f"**{scrape_ok}/{len(scraped)}** pages successfully scraped."
            )

            # Build a lookup: subtask_id -> subtask info
            subtask_map: dict[str, dict] = {}
            for st_item in subtasks_list:
                subtask_map[st_item.get("subtask_id", "")] = st_item

            # Build scraped status lookup: url -> scrape info
            scrape_map: dict[str, dict] = {}
            for sc_item in scraped:
                scrape_map[sc_item.get("url", "")] = sc_item

            # Group results by subtask
            by_subtask: dict[str, list[dict]] = {}
            for sr in search_results:
                sid = sr.get("subtask_id", "ungrouped")
                by_subtask.setdefault(sid, []).append(sr)

            for sid, group in by_subtask.items():
                st_info = subtask_map.get(sid, {})
                st_title = st_info.get("title", sid)
                st_priority = st_info.get("priority", "?")
                pri_color = {1: "#f85149", 2: "#d29922", 3: "#58a6ff"}.get(
                    st_priority if isinstance(st_priority, int) else 0, "#7d8590"
                )

                st.markdown(
                    f'<div style="margin-top:12px;font-size:0.88rem;font-weight:600;'
                    f'color:#e6edf3;">{st_title} '
                    f'<span style="font-size:0.68rem;color:{pri_color};'
                    f'border:1px solid {pri_color};border-radius:4px;padding:1px 6px;'
                    f'margin-left:6px;">P{st_priority}</span></div>',
                    unsafe_allow_html=True,
                )

                for sr in group:
                    domain = sr.get("source_domain", "")
                    cred_label, cred_cls = source_credibility(domain)
                    rel_score = sr.get("relevance_score", 0)
                    rel_color = score_color(rel_score)
                    cred_icon = {"Academic": "&#x1F7E2;", "News": "&#x1F535;", "Web": "&#x26AA;"}.get(
                        cred_label, "&#x26AA;"
                    )

                    url = sr.get("url", "")
                    sc_info = scrape_map.get(url, {})
                    scrape_ok_flag = sc_info.get("scrape_status") == "success"
                    scrape_badge = (
                        f'<span style="color:#3fb950;font-size:0.68rem;">scraped '
                        f'({sc_info.get("word_count", 0)} words)</span>'
                        if scrape_ok_flag
                        else '<span style="color:#f85149;font-size:0.68rem;">not scraped</span>'
                        if sc_info
                        else ""
                    )

                    rel_bar_pct = min(rel_score * 100, 100)
                    st.markdown(f'''
                    <div style="background:#161b22;border:1px solid #21262d;
                        border-radius:8px;padding:10px;margin:4px 0;">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div style="flex:1;">
                                <a href="{url}" target="_blank"
                                    style="font-size:0.85rem;font-weight:600;color:#58a6ff;
                                    text-decoration:none;">{sr.get("title", "Untitled")}</a>
                                <div style="font-size:0.72rem;margin-top:3px;">
                                    <span style="color:#7d8590;">{domain}</span>
                                    <span style="margin:0 4px;color:#484f58;">&middot;</span>
                                    <span class="{cred_cls}">{cred_icon} {cred_label}</span>
                                    <span style="margin:0 4px;color:#484f58;">&middot;</span>
                                    {scrape_badge}
                                </div>
                            </div>
                            <div style="text-align:right;min-width:50px;">
                                <span style="font-size:0.88rem;font-weight:700;color:{rel_color};">
                                    {rel_score:.2f}</span>
                            </div>
                        </div>
                        <div style="height:2px;background:#21262d;border-radius:1px;margin-top:6px;">
                            <div style="height:100%;width:{rel_bar_pct}%;background:{rel_color};
                                border-radius:1px;"></div>
                        </div>
                    </div>''', unsafe_allow_html=True)
        else:
            st.info("No search results available.")

    # ── EXPANDER 3 — Pipeline Execution Log ──────────────────────────────
    with st.expander("Pipeline Execution Log", expanded=False):
        r = results
        subtask_n = len(r.get("subtasks", []))
        search_n = len(r.get("search_results", []))
        scraped_content = r.get("scraped_content", []) or []
        scraped_n = len(scraped_content)
        scrape_ok_n = sum(1 for s in scraped_content if s.get("scrape_status") == "success")
        analysis_data = r.get("analysis") or {}
        theme_n = len(analysis_data.get("themes", []))
        draft_text = r.get("draft_report", "")
        draft_words = len(draft_text.split()) if draft_text else 0
        critic_data = r.get("critic_feedback") or {}
        critic_score = critic_data.get("quality_score", 0)
        critic_passed = critic_data.get("passed", False)
        rev_count = r.get("revision_count", 0)
        published = r.get("published_url")
        errors_list = r.get("errors", [])

        # Build the table rows
        pipe_log = [
            ("Planner", subtask_n > 0, f"Topic", f"{subtask_n} subtasks", "planner"),
            ("Search", search_n > 0, f"{subtask_n} subtasks", f"{search_n} results", "search"),
            ("Scraper", scraped_n > 0, f"{search_n} URLs", f"{scrape_ok_n}/{scraped_n} scraped", "scraper"),
            ("Analysis", bool(analysis_data.get("themes")), f"{scrape_ok_n} articles", f"{theme_n} themes", "analysis"),
            ("Writer", bool(draft_text), "Analysis data", f"{draft_words} words", "writer"),
        ]

        # If revision happened, add Writer row again
        if rev_count > 1:
            pipe_log.append(
                ("Writer (rev)", bool(draft_text), "Critic feedback", f"{draft_words} words (v{rev_count})", "writer"),
            )

        pipe_log.extend([
            ("Critic", bool(critic_data), "Draft report",
             f'Score: {critic_score:.2f}',
             "critic"),
            ("Publisher", published is not None, "Approved",
             "Saved locally" if published else "Skipped",
             "publisher"),
        ])

        # Build table HTML
        log_table = '''
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">
        <thead>
            <tr style="border-bottom:2px solid #21262d;">
                <th style="text-align:left;padding:8px;color:#7d8590;">Agent</th>
                <th style="text-align:center;padding:8px;color:#7d8590;">Status</th>
                <th style="text-align:left;padding:8px;color:#7d8590;">Input</th>
                <th style="text-align:left;padding:8px;color:#7d8590;">Output</th>
                <th style="text-align:center;padding:8px;color:#7d8590;">Time</th>
            </tr>
        </thead><tbody>'''

        for agent_name, has_data, input_desc, output_desc, agent_key in pipe_log:
            status_icon = '<span style="color:#3fb950;">Done</span>' if has_data else '<span style="color:#484f58;">&#x2014;</span>'
            # Check for agent-specific errors
            agent_errors = [e for e in errors_list if agent_key in str(e).lower()]
            if agent_errors:
                status_icon = '<span style="color:#f85149;">Error</span>'
                output_desc = f'<span style="color:#f85149;">{agent_errors[0][:60]}</span>'

            start_t, end_t = AGENT_TIMINGS.get(agent_key, (0, 0))
            time_est = f"~{end_t - start_t}s" if has_data else "—"

            log_table += f'''
            <tr style="border-bottom:1px solid #161b22;">
                <td style="padding:8px;color:#e6edf3;font-weight:600;">{agent_name}</td>
                <td style="padding:8px;text-align:center;">{status_icon}</td>
                <td style="padding:8px;color:#8b949e;">{input_desc}</td>
                <td style="padding:8px;color:#c9d1d9;">{output_desc}</td>
                <td style="padding:8px;text-align:center;color:#7d8590;">{time_est}</td>
            </tr>'''

        log_table += '</tbody></table></div>'
        st.markdown(log_table, unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.72rem;color:#484f58;margin-top:12px;">'
            f'Created: {r.get("created_at", "N/A")} &middot; '
            f'Completed: {r.get("completed_at", "N/A")}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — (Benchmark moved to evaluation side panel)
# ═══════════════════════════════════════════════════════════════════════════


# (Report Comparison moved to right evaluation panel)


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#484f58;font-size:0.82rem;padding:8px 0;">'
    'Built by Aditya &nbsp;&middot;&nbsp; '
    'NewsForge Multi-Agent Research System &nbsp;&middot;&nbsp; '
    'LangGraph &middot; Groq &middot; Tavily &middot; Langfuse'
    '</div>',
    unsafe_allow_html=True,
)

# Exit the main column context (matches __enter__ above)
_main_col.__exit__(None, None, None)
