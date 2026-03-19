"""Benchmark runner for the NewsForge pipeline.

Runs the full pipeline on all benchmark topics, judges results,
and produces summary statistics with resume capability.

Usage:
    python evaluation/benchmark_runner.py --dry-run
    python evaluation/benchmark_runner.py --topics 3
    python evaluation/benchmark_runner.py --topics 10
    python evaluation/benchmark_runner.py --start-from 5
    python evaluation/benchmark_runner.py --list-reports
    python evaluation/benchmark_runner.py --show-report 1
"""

import argparse
import csv
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GROQ_EXECUTION_MODEL, GROQ_JUDGE_MODEL, GROQ_REASONING_MODEL
from evaluation.benchmark_topics import BENCHMARK_TOPICS
from evaluation.judge import LLMJudge
from langgraph.types import Command
from orchestrator.graph import build_pipeline
from orchestrator.state import NewsForgeState

DELAY_BETWEEN_TOPICS = 5  # seconds — brief pause to avoid per-minute burst limits


class BenchmarkRunner:
    """Runs the full NewsForge pipeline against benchmark topics and scores results."""

    def __init__(self) -> None:
        self.topics = BENCHMARK_TOPICS
        self.judge = LLMJudge()
        self.results_dir = Path("data/benchmark_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def run_all(self, max_topics: int = 10, start_from: int = 0) -> dict:
        """Run the pipeline on benchmark topics and collect results.

        Args:
            max_topics: Maximum number of topics to run.
            start_from: Index to start from (0-based).

        Returns:
            Summary dict with all results and aggregate statistics.
        """
        topics_to_run = self.topics[start_from : start_from + max_topics]
        completed_ids = self._load_completed_ids()
        results: list[dict] = []

        print(f"\n{'=' * 60}")
        print(f"  NewsForge Benchmark Run")
        print(f"  Topics: {len(topics_to_run)} | Start: {start_from}")
        print(f"  Timestamp: {self.run_timestamp}")
        print(f"{'=' * 60}")
        print(f"\n[Benchmark] Model routing:")
        print(f"  Pool A — Scout (30K TPM) : {GROQ_REASONING_MODEL}")
        print(f"    └── Planner, Analysis, Critic")
        print(f"  Pool B — 8B (6K TPM)     : {GROQ_EXECUTION_MODEL}")
        print(f"    └── Writer, Judge")
        print()

        remaining_topics = [
            t for t in topics_to_run
            if t["topic_id"] not in completed_ids
        ]

        for i, topic_data in enumerate(remaining_topics):
            if i > 0:
                print(f"\n[Benchmark] Pausing {DELAY_BETWEEN_TOPICS}s...")
                time.sleep(DELAY_BETWEEN_TOPICS)

            print(f"\n[Benchmark] ({i + 1}/{len(remaining_topics)}) Running: {topic_data['topic']}")
            print(f"  Difficulty: {topic_data['expected_difficulty']} | Category: {topic_data['category']}")

            try:
                result = self.run_single(topic_data)
                results.append(result)
                self._save_incremental(result)
            except RuntimeError as e:
                if "Rate limit" in str(e):
                    print(f"\n[Benchmark] Stopped at topic {i + 1}.")
                    print(
                        f"[Benchmark] Resume tomorrow with: "
                        f"--start-from {start_from + i}"
                    )
                    break
                raise

            status = "PASS" if result.get("judge_score") and result["judge_score"] >= 6.0 else "FAIL"
            if not result["pipeline_success"]:
                status = "ERROR"
            print(f"  Result: {status} | Score: {result.get('judge_score', 'N/A')} | Time: {result['elapsed_seconds']:.0f}s")

        # Save full run results
        if results:
            results_path = self.save_results(results)
            print(f"\n[Benchmark] Results saved to: {results_path}")

            html_path = self.generate_html_report(results)
            print(f"[Benchmark] HTML report: {html_path}")
            print(f"[Benchmark] Open in browser: file://{html_path.absolute()}")

        self.print_summary(results)
        return {"results": results, "run_timestamp": self.run_timestamp}

    def run_single(self, topic_data: dict) -> dict:
        """Run the full pipeline on one topic and judge the output.

        Raises RuntimeError on rate-limit (429) so the caller can stop
        the benchmark cleanly. All other errors are recorded and returned.

        Args:
            topic_data: Benchmark topic dict.

        Returns:
            Result dict with pipeline output, judge scores, and timing.
        """
        result: dict = {
            "topic_id": topic_data["topic_id"],
            "topic": topic_data["topic"],
            "category": topic_data["category"],
            "expected_difficulty": topic_data["expected_difficulty"],
            "pipeline_result": None,
            "judge_output": None,
            "pipeline_success": False,
            "error": None,
            "elapsed_seconds": 0.0,
            **self._empty_metrics(),
            "judge_score": None,
        }

        start_time = time.time()

        try:
            # Build a fresh pipeline for each topic to avoid state leaks
            pipeline = build_pipeline()
            research_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": research_id}}

            initial_state: NewsForgeState = {
                "research_id": research_id,
                "topic": topic_data["topic"],
                "subtasks": [],
                "search_results": [],
                "scraped_content": [],
                "analysis": None,
                "draft_report": None,
                "critic_feedback": None,
                "revision_count": 0,
                "human_decision": None,
                "published_url": None,
                "published_record_id": None,
                "pipeline_status": "starting",
                "errors": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
            }

            # First invoke — runs until human_review_node interrupt
            pipeline_output = pipeline.invoke(initial_state, config=config)

            # Auto-approve for benchmark — bypass HITL
            pipeline_output = pipeline.invoke(
                Command(resume={"decision": "approve"}),
                config=config,
            )

            result["pipeline_success"] = True
            result["pipeline_result"] = self._serialize_state(pipeline_output)
            result.update(self._extract_metrics(pipeline_output))

            # Judge the output
            try:
                judge_output = self.judge.judge(topic_data, pipeline_output)
                result["judge_output"] = judge_output.model_dump()
                result["judge_score"] = judge_output.overall_score
            except Exception as judge_err:
                result["error"] = f"Judge failed: {judge_err}"
                print(f"  [WARNING] Judge failed: {judge_err}")

        except Exception as pipeline_err:
            error_str = str(pipeline_err)
            if "429" in error_str or "rate_limit_exceeded" in error_str:
                result["elapsed_seconds"] = round(time.time() - start_time, 2)
                print(f"\n[Benchmark] ⛔ Rate limit hit — daily quota exhausted.")
                print(f"[Benchmark] Rerun tomorrow after UTC midnight.")
                raise RuntimeError(
                    "Rate limit exceeded — stopping benchmark"
                ) from pipeline_err

            result["pipeline_success"] = False
            result["error"] = error_str
            print(f"  [ERROR] Pipeline failed: {pipeline_err}")

        result["elapsed_seconds"] = round(time.time() - start_time, 2)
        return result

    def save_results(self, results: list[dict]) -> str:
        """Save results as JSON and CSV.

        Args:
            results: List of result dicts from run_single.

        Returns:
            Path to the JSON results file.
        """
        json_path = self.results_dir / f"run_{self.run_timestamp}.json"
        csv_path = self.results_dir / f"run_{self.run_timestamp}.csv"

        # Save JSON (full data)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV (summary metrics)
        csv_fields = [
            "topic_id", "topic", "category", "expected_difficulty",
            "pipeline_success", "elapsed_seconds", "subtask_count",
            "search_results_count", "scrape_success_count", "scrape_total_count",
            "analysis_themes", "analysis_confidence", "report_word_count",
            "critic_score", "revision_count", "judge_score", "error",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        return str(json_path)

    def print_summary(self, results: list[dict]) -> None:
        """Print a formatted summary table of benchmark results."""
        if not results:
            print("\n[Benchmark] No results to display.")
            return

        # Header
        print(f"\n{'╔' + '═' * 72 + '╗'}")
        print(f"{'║'} {'NewsForge Benchmark Results':^70} {'║'}")
        print(f"{'╠' + '═' * 72 + '╣'}")
        print(f"{'║'} {'Topic':<35} │ {'Score':>5} │ {'Time':>5} │ {'Status':>6} │ {'Diff':>6} {'║'}")
        print(f"{'╠' + '═' * 72 + '╣'}")

        # Rows
        pass_count = 0
        total_score = 0.0
        total_time = 0.0
        scored_count = 0

        for r in results:
            name = r["topic"][:33] + ".." if len(r["topic"]) > 35 else r["topic"]
            score = r.get("judge_score")
            elapsed = r["elapsed_seconds"]
            difficulty = r.get("expected_difficulty", "?")[:6]
            total_time += elapsed

            if not r["pipeline_success"]:
                status = " ERR "
            elif score is not None and score >= 6.0:
                status = "  ✅ "
                pass_count += 1
                total_score += score
                scored_count += 1
            elif score is not None:
                status = "  ❌ "
                total_score += score
                scored_count += 1
            else:
                status = "  ⚠️ "

            score_str = f"{score:.1f}" if score is not None else " N/A"
            time_str = f"{elapsed:.0f}s"

            print(f"{'║'} {name:<35} │ {score_str:>5} │ {time_str:>5} │ {status:>6} │ {difficulty:>6} {'║'}")

        # Footer
        avg_score = total_score / scored_count if scored_count else 0.0
        avg_time = total_time / len(results) if results else 0.0

        print(f"{'╠' + '═' * 72 + '╣'}")
        print(
            f"{'║'} {'AVERAGE':<35} │ {avg_score:>5.1f} │ {avg_time:>4.0f}s │ "
            f"{pass_count}/{len(results):>3} │ {'':>6} {'║'}"
        )
        print(f"{'╚' + '═' * 72 + '╝'}")

        # Additional stats
        successful = [r for r in results if r["pipeline_success"]]
        if successful:
            avg_scrape_rate = (
                sum(r["scrape_success_count"] for r in successful)
                / max(sum(r["scrape_total_count"] for r in successful), 1)
                * 100
            )
            avg_confidence = sum(r["analysis_confidence"] for r in successful) / len(successful)
            avg_critic = sum(r["critic_score"] for r in successful) / len(successful)
            avg_words = sum(r["report_word_count"] for r in successful) / len(successful)

            print(f"\n  Pipeline Statistics:")
            print(f"  ├─ Avg scrape success rate: {avg_scrape_rate:.0f}%")
            print(f"  ├─ Avg analysis confidence: {avg_confidence:.2f}")
            print(f"  ├─ Avg critic score:        {avg_critic:.2f}")
            print(f"  ├─ Avg report word count:   {avg_words:.0f}")
            print(f"  └─ Total benchmark time:    {total_time:.0f}s")

        # Failed topics
        failed = [r for r in results if not r["pipeline_success"]]
        if failed:
            print(f"\n  Failed topics:")
            for r in failed:
                print(f"  ├─ {r['topic_id']}: {r['topic']}")
                print(f"  │  Error: {r['error']}")

    def _extract_metrics(self, result: dict) -> dict:
        """Safely extract metrics from pipeline result.

        Returns zeros for any missing or None fields.
        """
        if not result:
            return self._empty_metrics()
        return {
            "subtask_count": len(result.get("subtasks") or []),
            "search_results_count": len(result.get("search_results") or []),
            "scrape_success_count": sum(
                1 for s in (result.get("scraped_content") or [])
                if s.get("scrape_status") == "success"
            ),
            "scrape_total_count": len(result.get("scraped_content") or []),
            "analysis_themes": len(
                (result.get("analysis") or {}).get("themes") or []
            ),
            "analysis_confidence": (
                result.get("analysis") or {}
            ).get("confidence_score", 0.0),
            "report_word_count": len(
                (result.get("draft_report") or "").split()
            ),
            "critic_score": (
                result.get("critic_feedback") or {}
            ).get("quality_score", 0.0),
            "revision_count": result.get("revision_count", 0),
        }

    def _empty_metrics(self) -> dict:
        """Return zeroed-out metrics dict."""
        return {
            "subtask_count": 0,
            "search_results_count": 0,
            "scrape_success_count": 0,
            "scrape_total_count": 0,
            "analysis_themes": 0,
            "analysis_confidence": 0.0,
            "report_word_count": 0,
            "critic_score": 0.0,
            "revision_count": 0,
        }

    def load_and_print_summary(self) -> None:
        """Load existing results from disk and print summary."""
        all_results: list[dict] = []
        for f in sorted(self.results_dir.glob("*.json")):
            with open(f) as file:
                data = json.load(file)
                if isinstance(data, list):
                    all_results.extend(data)
        if all_results:
            self.print_summary(all_results)
        else:
            print("No results found in data/benchmark_results/")

    def _save_incremental(self, result: dict) -> None:
        """Save a single topic result for crash-recovery resume."""
        incremental_dir = self.results_dir / f"run_{self.run_timestamp}_incremental"
        incremental_dir.mkdir(parents=True, exist_ok=True)
        path = incremental_dir / f"{result['topic_id']}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    def _load_completed_ids(self) -> set[str]:
        """Load topic IDs from any incremental results in the results directory."""
        completed: set[str] = set()
        for incremental_dir in self.results_dir.glob("run_*_incremental"):
            for result_file in incremental_dir.glob("topic_*.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    if data.get("pipeline_success") or data.get("judge_score") is not None:
                        completed.add(data["topic_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed

    @staticmethod
    def _serialize_state(state: dict) -> dict:
        """Convert pipeline state to JSON-serializable dict."""
        serializable = {}
        for key, value in state.items():
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                serializable[key] = str(value)
        return serializable

    def _load_latest_results(self) -> list[dict]:
        """Load results from the most recent benchmark run JSON."""
        json_files = sorted(self.results_dir.glob("run_*.json"))
        # Exclude incremental dirs — only full run files
        json_files = [f for f in json_files if f.is_file()]
        if not json_files:
            return []
        with open(json_files[-1]) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def show_best_report(self, n: int = 1) -> None:
        """Show the Nth best report from the latest benchmark run."""
        results = self._load_latest_results()
        if not results:
            print("No benchmark results found.")
            return

        scored = [r for r in results if r.get("judge_score") is not None]
        scored.sort(key=lambda r: r["judge_score"], reverse=True)

        if n < 1 or n > len(scored):
            print(f"Only {len(scored)} scored reports available. Requested #{n}.")
            return

        r = scored[n - 1]
        topic = r["topic"]
        score = r.get("judge_score", 0.0)
        critic = r.get("critic_score", 0.0)
        words = r.get("report_word_count", 0)

        # Try to find the report file
        pipeline_result = r.get("pipeline_result") or {}
        local_path = pipeline_result.get("published_url") or ""
        # Also check for report files by research_id
        if not local_path or not Path(local_path).exists():
            research_id = pipeline_result.get("research_id", "")
            if research_id:
                matches = list(Path("data/reports").glob(f"report_{research_id[:8]}_*.md"))
                if matches:
                    local_path = str(matches[0])

        print(f"\n{'=' * 56}")
        print(f"  Report #{n}: {topic}")
        print(f"  Judge Score: {score:.1f}/10")
        print(f"  Critic Score: {critic:.2f}")
        print(f"  Words: {words}")
        print(f"  File: {local_path or 'N/A'}")
        print(f"{'=' * 56}\n")

        if local_path and Path(local_path).exists():
            content = Path(local_path).read_text(encoding="utf-8")
            print(content)
        else:
            print("Report file not found")

    def list_reports(self) -> None:
        """List all generated reports with scores from the latest benchmark run."""
        results = self._load_latest_results()
        if not results:
            print("No benchmark results found.")
            return

        scored = [r for r in results if r.get("judge_score") is not None]
        scored.sort(key=lambda r: r["judge_score"], reverse=True)

        if not scored:
            print("No scored reports found.")
            return

        # Table header
        print(f"\n{'─' * 57}")
        print(f"  {'NewsForge Generated Reports':^53}")
        print(f"{'─' * 57}")
        print(f"  {'#':>3}  {'Topic':<30} {'Score':>5}  {'Words':>5}")
        print(f"{'─' * 57}")

        for i, r in enumerate(scored, 1):
            topic = r["topic"]
            if len(topic) > 28:
                topic = topic[:27] + ".."
            score = r.get("judge_score", 0.0)
            words = r.get("report_word_count", 0)
            print(f"  {i:>3}  {topic:<30} {score:>5.1f}  {words:>5}")

        print(f"{'─' * 57}")

    def generate_html_report(self, results: list[dict]) -> Path:
        """Generate a self-contained HTML summary of benchmark results."""
        html_path = self.results_dir / "summary.html"

        # Compute aggregates
        scored = [r for r in results if r.get("judge_score") is not None]
        pass_count = sum(1 for r in scored if r["judge_score"] >= 6.0)
        avg_score = sum(r["judge_score"] for r in scored) / len(scored) if scored else 0.0
        successful = [r for r in results if r["pipeline_success"]]
        total_time = sum(r["elapsed_seconds"] for r in results)

        avg_scrape = 0.0
        avg_confidence = 0.0
        avg_critic = 0.0
        if successful:
            total_scraped = sum(r["scrape_total_count"] for r in successful)
            total_success = sum(r["scrape_success_count"] for r in successful)
            avg_scrape = (total_success / max(total_scraped, 1)) * 100
            avg_confidence = sum(r["analysis_confidence"] for r in successful) / len(successful)
            avg_critic = sum(r["critic_score"] for r in successful) / len(successful)

        # Score color
        if avg_score >= 7:
            score_color = "#4ade80"
        elif avg_score >= 5:
            score_color = "#facc15"
        else:
            score_color = "#f87171"

        # Score distribution
        buckets = {"9-10": 0, "7-8": 0, "5-6": 0, "3-4": 0, "1-2": 0, "0": 0}
        for r in scored:
            s = r["judge_score"]
            if s >= 9:
                buckets["9-10"] += 1
            elif s >= 7:
                buckets["7-8"] += 1
            elif s >= 5:
                buckets["5-6"] += 1
            elif s >= 3:
                buckets["3-4"] += 1
            elif s >= 1:
                buckets["1-2"] += 1
            else:
                buckets["0"] += 1
        max_bucket = max(buckets.values()) if buckets.values() else 1

        # Build results table rows
        sorted_results = sorted(results, key=lambda r: r.get("judge_score") or 0, reverse=True)
        table_rows = ""
        for rank, r in enumerate(sorted_results, 1):
            score = r.get("judge_score")
            passed = score is not None and score >= 6.0
            row_bg = "#1a2e1a" if passed else ("#2e1a1a" if r["pipeline_success"] else "#2e2a1a")
            score_str = f"{score:.1f}" if score is not None else "N/A"
            critic_str = f"{r.get('critic_score', 0):.2f}"
            status = "PASS" if passed else ("FAIL" if r["pipeline_success"] else "ERROR")
            table_rows += f"""<tr style="background:{row_bg}">
                <td>{rank}</td><td>{r['topic']}</td><td>{r.get('category','')}</td>
                <td>{r.get('expected_difficulty','')}</td><td><b>{score_str}</b></td>
                <td>{critic_str}</td><td>{r.get('report_word_count',0)}</td>
                <td>{r.get('scrape_success_count',0)}/{r.get('scrape_total_count',0)}</td>
                <td>{r['elapsed_seconds']:.0f}s</td><td>{status}</td></tr>\n"""

        # Build per-topic detail cards
        topic_cards = ""
        for r in sorted_results:
            judge = r.get("judge_output") or {}
            strengths = judge.get("strengths", [])
            weaknesses = judge.get("weaknesses", [])
            reasoning = judge.get("judge_reasoning", "")
            diff = r.get("expected_difficulty", "unknown")
            diff_color = {"easy": "#4ade80", "medium": "#facc15", "hard": "#f87171"}.get(diff, "#888")

            strengths_html = "".join(f"<li>{s}</li>" for s in strengths) if strengths else "<li>N/A</li>"
            weaknesses_html = "".join(f"<li>{w}</li>" for w in weaknesses) if weaknesses else "<li>N/A</li>"

            dims_html = ""
            for dim_key, dim_label in [
                ("research_depth", "Research Depth"),
                ("source_diversity", "Source Diversity"),
                ("topic_coverage", "Topic Coverage"),
                ("factual_coherence", "Factual Coherence"),
                ("report_quality", "Report Quality"),
            ]:
                val = judge.get(dim_key, 0)
                bar_width = val * 10  # 0-100%
                dims_html += f"""<div style="margin:4px 0"><span style="display:inline-block;width:140px">{dim_label}</span>
                    <span style="display:inline-block;width:120px;background:#333;border-radius:3px;height:14px;vertical-align:middle">
                    <span style="display:inline-block;width:{bar_width}%;background:{score_color};height:100%;border-radius:3px"></span>
                    </span> <span>{val:.1f}</span></div>"""

            topic_cards += f"""
            <details style="margin:12px 0;background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:0">
                <summary style="padding:12px;cursor:pointer;font-size:15px">
                    <b>{r['topic']}</b>
                    <span style="background:{diff_color};color:#000;padding:2px 8px;border-radius:10px;font-size:11px;margin-left:8px">{diff}</span>
                    <span style="float:right;font-weight:bold">{r.get('judge_score','N/A')}</span>
                </summary>
                <div style="padding:0 16px 16px">
                    {dims_html}
                    <div style="margin-top:12px"><b>Strengths:</b><ul style="margin:4px 0">{strengths_html}</ul></div>
                    <div><b>Weaknesses:</b><ul style="margin:4px 0">{weaknesses_html}</ul></div>
                    <div style="margin-top:8px;color:#aaa;font-style:italic">{reasoning}</div>
                    <div style="margin-top:8px;font-size:12px;color:#666">
                        Scrapes: {r.get('scrape_success_count',0)}/{r.get('scrape_total_count',0)} |
                        Themes: {r.get('analysis_themes',0)} |
                        Words: {r.get('report_word_count',0)} |
                        Time: {r['elapsed_seconds']:.0f}s
                    </div>
                </div>
            </details>"""

        # Build score distribution bars
        dist_html = ""
        for label, count in buckets.items():
            bar_w = int((count / max(max_bucket, 1)) * 200)
            dist_html += f'<div style="margin:2px 0"><code>{label:>4}</code> <span style="display:inline-block;width:{bar_w}px;height:16px;background:#6366f1;border-radius:2px;vertical-align:middle"></span> ({count})</div>'

        # Failure analysis
        failed = [r for r in results if not r["pipeline_success"]]
        failure_html = ""
        if failed:
            failure_html = '<h2 style="color:#f87171">Failure Analysis</h2>'
            for r in failed:
                failure_html += f'<div style="margin:8px 0;padding:8px;background:#2e1a1a;border-radius:4px"><b>{r["topic"]}</b><br><code style="color:#f87171">{r.get("error","Unknown error")}</code></div>'

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NewsForge Benchmark Results</title></head>
<body style="background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:24px">

<h1 style="border-bottom:1px solid #333;padding-bottom:12px">NewsForge Benchmark Results</h1>
<p style="color:#888">Run: {self.run_timestamp} | Topics: {len(results)} | Pass rate: {pass_count}/{len(scored)}</p>

<div style="display:flex;gap:16px;flex-wrap:wrap;margin:20px 0">
    <div style="background:#161b22;border:1px solid #333;border-radius:8px;padding:16px;flex:1;min-width:140px;text-align:center">
        <div style="font-size:28px;font-weight:bold;color:{score_color}">{avg_score:.1f}</div>
        <div style="color:#888;font-size:13px">Avg Judge Score</div>
    </div>
    <div style="background:#161b22;border:1px solid #333;border-radius:8px;padding:16px;flex:1;min-width:140px;text-align:center">
        <div style="font-size:28px;font-weight:bold">{avg_scrape:.0f}%</div>
        <div style="color:#888;font-size:13px">Scrape Success</div>
    </div>
    <div style="background:#161b22;border:1px solid #333;border-radius:8px;padding:16px;flex:1;min-width:140px;text-align:center">
        <div style="font-size:28px;font-weight:bold">{avg_confidence:.2f}</div>
        <div style="color:#888;font-size:13px">Avg Confidence</div>
    </div>
    <div style="background:#161b22;border:1px solid #333;border-radius:8px;padding:16px;flex:1;min-width:140px;text-align:center">
        <div style="font-size:28px;font-weight:bold">{avg_critic:.2f}</div>
        <div style="color:#888;font-size:13px">Avg Critic Score</div>
    </div>
    <div style="background:#161b22;border:1px solid #333;border-radius:8px;padding:16px;flex:1;min-width:140px;text-align:center">
        <div style="font-size:28px;font-weight:bold">{total_time:.0f}s</div>
        <div style="color:#888;font-size:13px">Total Time</div>
    </div>
</div>

<h2>Results</h2>
<table style="width:100%;border-collapse:collapse;font-size:14px">
<thead><tr style="background:#161b22;border-bottom:2px solid #333">
    <th style="padding:8px;text-align:left">#</th><th style="padding:8px;text-align:left">Topic</th>
    <th style="padding:8px">Cat</th><th style="padding:8px">Diff</th><th style="padding:8px">Judge</th>
    <th style="padding:8px">Critic</th><th style="padding:8px">Words</th><th style="padding:8px">Scrapes</th>
    <th style="padding:8px">Time</th><th style="padding:8px">Status</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>

<h2>Score Distribution</h2>
<div style="background:#161b22;border:1px solid #333;border-radius:8px;padding:16px;font-family:monospace">
{dist_html}
</div>

<h2>Per-Topic Details</h2>
{topic_cards}

{failure_html}

<div style="margin-top:40px;padding-top:16px;border-top:1px solid #333;color:#555;font-size:12px;text-align:center">
    Generated by NewsForge Benchmark Runner | {self.run_timestamp}
</div>

</body></html>"""

        html_path.write_text(html, encoding="utf-8")
        return html_path


def main() -> None:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="NewsForge Benchmark Runner")
    parser.add_argument(
        "--topics", type=int, default=10,
        help="Number of topics to run (default: 10)",
    )
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="Topic index to start from, 0-based (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate setup without running the pipeline",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print summary of existing results without running",
    )
    parser.add_argument(
        "--show-report", type=int, default=None, metavar="N",
        help="Show the Nth best report from the latest benchmark run",
    )
    parser.add_argument(
        "--list-reports", action="store_true",
        help="List all generated reports with scores",
    )
    args = parser.parse_args()

    if args.list_reports:
        runner = BenchmarkRunner()
        runner.list_reports()
        sys.exit(0)

    if args.show_report is not None:
        runner = BenchmarkRunner()
        runner.show_best_report(n=args.show_report)
        sys.exit(0)

    if args.summary_only:
        runner = BenchmarkRunner()
        runner.load_and_print_summary()
        sys.exit(0)

    if args.dry_run:
        print("[Benchmark] Dry run — validating setup...\n")
        print(f"  Topics loaded: {len(BENCHMARK_TOPICS)}")
        for t in BENCHMARK_TOPICS:
            print(f"  ├─ {t['topic_id']}: {t['topic']}")
            print(f"  │  Category: {t['category']} | Difficulty: {t['expected_difficulty']}")
            print(f"  │  Criteria: {len(t['quality_criteria'])} | Challenges: {len(t['known_challenges'])}")

        print(f"\n  Results dir: data/benchmark_results/")
        print(f"  Delay between topics: {DELAY_BETWEEN_TOPICS}s")

        # Validate judge can be instantiated
        try:
            judge = LLMJudge()
            print(f"  Judge LLM: {judge.llm.model_name} ✓")
        except Exception as e:
            print(f"  Judge LLM: FAILED — {e}")

        # Validate pipeline can be built
        try:
            pipeline = build_pipeline()
            print(f"  Pipeline: compiled ✓")
        except Exception as e:
            print(f"  Pipeline: FAILED — {e}")

        print("\n[Benchmark] Dry run complete.")
        return

    runner = BenchmarkRunner()
    runner.run_all(max_topics=args.topics, start_from=args.start_from)


if __name__ == "__main__":
    main()
