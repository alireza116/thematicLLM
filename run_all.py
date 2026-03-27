"""
run_all.py — Run Thematic-LM on all substantive free-text columns.

Usage:
    # Test mode: 20 rows per column to validate the pipeline
    python run_all.py --test

    # Full analysis on all columns
    python run_all.py

    # Full analysis, specific columns only
    python run_all.py --columns pq_trustReason pq_surpriseReason

    # Override provider/model
    python run_all.py --test --provider gemini --model gemini-3-flash-preview
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Study context
#
# BASE_STUDY_CONTEXT is injected into every column's analysis.
# COLUMN_CONTEXTS lets you append column-specific text (e.g. the exact survey
# question) on top of the base. If a column has no entry here, only the base
# context is used. Set either to None / empty dict to disable.
# ---------------------------------------------------------------------------

BASE_STUDY_CONTEXT = """This study investigates how people perceive and interpret data visualizations
showing political and social trends. Participants were shown a series of charts
and asked open-ended questions about their reactions, understanding, and attitudes."""

COLUMN_CONTEXTS = {
    "pq_trustReason": (
        'The column contains responses to: "Why do you trust or distrust this visualization?"'
    ),
    "pq_surpriseReason": (
        'The column contains responses to: "What, if anything, surprised you about this visualization?"'
    ),
    "pq_additionalComments": (
        "The column contains open-ended additional comments participants chose to leave at the end of the study."
    ),
    "memTrans_taskDescription": (
        "The column contains participants' free-text descriptions of the task they were asked to perform "
        "during the memorability transfer phase of the study."
    ),
    "eliciTrain_taskDescription": (
        "The column contains participants' free-text descriptions of the task they were asked to perform "
        "during the elicitation training phase of the study."
    ),
    "viz_summary_SPV_violgoals_other_pre": (
        "The column contains participants' written summaries of a visualization showing trends in "
        "support for political violence goals. Collected before any experimental manipulation."
    ),
    "viz_summary_SUP_ban_other_pre": (
        "The column contains participants' written summaries of a visualization showing trends in "
        "support for banning far-right rallies. Collected before any experimental manipulation."
    ),
    "viz_summary_SUP_court_other_pre": (
        "The column contains participants' written summaries of a visualization showing trends in "
        "support for ignoring court rulings. Collected before any experimental manipulation."
    ),
    "viz_summary_SUP_laws_other_pre": (
        "The column contains participants' written summaries of a visualization showing trends in "
        "support for partisan laws. Collected before any experimental manipulation."
    ),
    "viz_summary_feeling_thermometer": (
        "The column contains participants' written summaries of a visualization showing feeling "
        "thermometer trends (affective ratings toward political groups over time)."
    ),
    "viz_summary_trend_pres_approval": (
        "The column contains participants' written summaries of a visualization showing presidential "
        "approval rating trends over time."
    ),
    "viz_summary_trend_economy_right_track": (
        "The column contains participants' written summaries of a visualization showing trends in "
        "public opinion on whether the economy is on the right track."
    ),
    "viz_summary_trend_immigration": (
        "The column contains participants' written summaries of a visualization showing trends in "
        "public attitudes toward immigration."
    ),
}


def build_study_context(column: str) -> str:
    """Combine base context with any column-specific text."""
    parts = [BASE_STUDY_CONTEXT.strip()]
    extra = COLUMN_CONTEXTS.get(column)
    if extra:
        parts.append(extra.strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Columns worth analysing (free-text, avg response length > 70 chars)
# ---------------------------------------------------------------------------

TARGET_COLUMNS = [
    "pq_trustReason",                        # Why they trust/distrust the visualization
    "pq_surpriseReason",                     # What surprised them
    "pq_additionalComments",                 # Open-ended additional comments
    "memTrans_taskDescription",              # Participants' description of the task
    "eliciTrain_taskDescription",            # Elicitation training task descriptions
    "viz_summary_SPV_violgoals_other_pre",   # Summary: support for political violence goals
    "viz_summary_SUP_ban_other_pre",         # Summary: support for banning far-right rallies
    "viz_summary_SUP_court_other_pre",       # Summary: support for ignoring court rulings
    "viz_summary_SUP_laws_other_pre",        # Summary: support for partisan laws
    "viz_summary_feeling_thermometer",       # Summary: feeling thermometer trend
    "viz_summary_trend_pres_approval",       # Summary: presidential approval trend
    "viz_summary_trend_economy_right_track", # Summary: economy right track trend
    "viz_summary_trend_immigration",         # Summary: immigration trend
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch thematic analysis across all text columns.")
    p.add_argument("--test", action="store_true",
                   help="Test mode: analyse only 20 rows per column.")
    p.add_argument("--columns", nargs="+", default=None,
                   help="Subset of columns to run (default: all TARGET_COLUMNS).")
    p.add_argument("--provider", default="gemini")
    p.add_argument("--model", default="gemini-3-flash-preview")
    p.add_argument("--n-coders", type=int, default=2)
    p.add_argument("--n-theme-coders", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_column(col: str, args, limit: int = None) -> dict:
    cmd = [
        sys.executable, "run.py",
        "--column", col,
        "--provider", args.provider,
        "--model", args.model,
        "--n-coders", str(args.n_coders),
        "--n-theme-coders", str(args.n_theme_coders),
        "--batch-size", str(args.batch_size),
        "--results-dir", args.results_dir,
        "--study-context", build_study_context(col),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if args.skip_eval:
        cmd.append("--skip-eval")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    return {
        "column": col,
        "success": result.returncode == 0,
        "elapsed_s": round(elapsed, 1),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    columns = args.columns or TARGET_COLUMNS
    limit = 20 if args.test else None
    mode = "TEST (20 rows)" if args.test else "FULL"

    print(f"\n{'='*65}")
    print(f"  Thematic-LM Batch Run  |  {mode}  |  {args.provider}/{args.model}")
    print(f"  {len(columns)} columns  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}\n")

    summary = []
    for i, col in enumerate(columns, 1):
        print(f"[{i}/{len(columns)}] {col} ...", flush=True)
        result = run_column(col, args, limit=limit)

        status = "OK" if result["success"] else "FAIL"
        print(f"       {status}  ({result['elapsed_s']}s)")

        # Print last few lines of stdout for progress visibility
        lines = [l for l in result["stdout"].splitlines() if l.strip()]
        for line in lines[-4:]:
            print(f"       {line}")

        if not result["success"]:
            err_lines = [l for l in result["stderr"].splitlines() if l.strip()]
            for line in err_lines[-5:]:
                print(f"  ERR: {line}")

        summary.append(result)
        print()

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------

    passed = [r for r in summary if r["success"]]
    failed = [r for r in summary if not r["success"]]
    total_time = sum(r["elapsed_s"] for r in summary)

    print(f"\n{'='*65}")
    print(f"  Results: {len(passed)}/{len(summary)} columns completed successfully")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'='*65}")

    if failed:
        print("\nFailed columns:")
        for r in failed:
            print(f"  - {r['column']}")
            err_lines = [l for l in r["stderr"].splitlines() if l.strip()]
            for line in err_lines[-3:]:
                print(f"      {line}")

    # Write batch summary JSON
    summary_path = Path(args.results_dir) / f"batch_summary_{'test_' if args.test else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "mode": mode,
        "provider": args.provider,
        "model": args.model,
        "columns_run": len(summary),
        "passed": len(passed),
        "failed": len(failed),
        "total_time_s": total_time,
        "results": [{k: v for k, v in r.items() if k != "stdout"} for r in summary],
    }, indent=2))
    print(f"\nBatch summary → {summary_path}\n")

    if args.test and len(passed) == len(summary):
        print("All tests passed. Run without --test for the full analysis.\n")
    elif args.test:
        print("Some columns failed. Fix errors above before running the full analysis.\n")


if __name__ == "__main__":
    main()
