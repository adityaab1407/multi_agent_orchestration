"""Cleanup script for NewsForge data directories.

Removes stale reports, old benchmark results, and runtime artifacts
that should not persist in the repository.

Usage:
    python scripts/cleanup_data.py
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark_results"
CHECKPOINT_DB = PROJECT_ROOT / "data" / "newsforge_checkpoints.db"

removed: list[str] = []


def clean_reports() -> None:
    """Remove reports older than 7 days OR keep only the 10 most recent.

    Also removes any file in data/reports/ not starting with 'report_'.
    """
    if not REPORTS_DIR.exists():
        print("[reports] Directory does not exist — skipping.")
        return

    # Remove non-report files
    for f in REPORTS_DIR.iterdir():
        if f.is_file() and not f.name.startswith("report_"):
            removed.append(str(f.relative_to(PROJECT_ROOT)))
            f.unlink()

    # Gather all report files (md + metadata json), sorted newest first
    report_files = sorted(
        REPORTS_DIR.glob("report_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    keep_count = 10  # pairs = 10 reports * 2 files each = 20 files max

    # Group by report stem (report_XXXX_topic has .md and _metadata.json)
    seen_stems: set[str] = set()
    kept = 0

    for f in report_files:
        # Derive stem: strip _metadata.json or .md
        stem = f.stem.replace("_metadata", "")
        if stem in seen_stems:
            # Already decided for this report
            continue

        seen_stems.add(stem)
        kept += 1

        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        if kept > keep_count or mtime < cutoff:
            # Remove both .md and _metadata.json for this report
            for companion in REPORTS_DIR.glob(f"{stem}*"):
                removed.append(str(companion.relative_to(PROJECT_ROOT)))
                companion.unlink()

    print(f"[reports] Kept {min(kept, keep_count)} reports, removed {len([r for r in removed if 'reports/' in r])} files.")


def clean_benchmarks() -> None:
    """Remove old benchmark results, keeping only the latest run_*.json and run_*.csv."""
    if not BENCHMARK_DIR.exists():
        print("[benchmark] Directory does not exist — skipping.")
        return

    # Keep only the latest run_* files
    run_jsons = sorted(BENCHMARK_DIR.glob("run_*.json"), reverse=True)
    run_csvs = sorted(BENCHMARK_DIR.glob("run_*.csv"), reverse=True)
    run_htmls = sorted(BENCHMARK_DIR.glob("run_*.html"), reverse=True)

    # Keep the first (latest) of each, remove the rest
    for file_list in [run_jsons, run_csvs, run_htmls]:
        for f in file_list[1:]:
            removed.append(str(f.relative_to(PROJECT_ROOT)))
            f.unlink()

    # Remove old incremental directories except the latest
    incremental_dirs = sorted(BENCHMARK_DIR.glob("*_incremental"), reverse=True)
    for d in incremental_dirs[1:]:
        for f in d.iterdir():
            removed.append(str(f.relative_to(PROJECT_ROOT)))
            f.unlink()
        d.rmdir()
        removed.append(str(d.relative_to(PROJECT_ROOT)))

    kept_count = min(1, len(run_jsons)) + min(1, len(run_csvs)) + min(1, len(run_htmls))
    print(f"[benchmark] Kept {kept_count} latest run files, removed {len([r for r in removed if 'benchmark' in r])} files.")


def clean_checkpoints() -> None:
    """Remove SQLite checkpoint database and WAL/SHM files."""
    for suffix in ["", "-wal", "-shm"]:
        db_path = Path(str(CHECKPOINT_DB) + suffix)
        if db_path.exists():
            removed.append(str(db_path.relative_to(PROJECT_ROOT)))
            db_path.unlink()

    if any("checkpoints" in r for r in removed):
        print("[checkpoints] Removed checkpoint database and WAL files.")
    else:
        print("[checkpoints] No checkpoint files found.")


def main() -> None:
    print("NewsForge Data Cleanup")
    print("=" * 40)

    clean_reports()
    clean_benchmarks()
    clean_checkpoints()

    print()
    if removed:
        print(f"Total files removed: {len(removed)}")
        for r in removed:
            print(f"  - {r}")
    else:
        print("Nothing to clean up.")


if __name__ == "__main__":
    main()
