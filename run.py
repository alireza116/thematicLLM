"""
run.py — Run Thematic-LM on a column from the data file.

Usage:
    python run.py --column pq_trustReason
    python run.py --column pq_surpriseReason --provider gemini --model gemini-2.0-flash
    python run.py --column pq_additionalComments --n-coders 3 --batch-size 20

Output (written to results/<column>_<timestamp>/):
    themes.json   — raw themes with quotes
    codebook.json — adaptive codebook built during coding stage
    report.md     — markdown summary: themes, supporting codes, example quotes
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from thematic_lm import LLMClient, ThematicLMPipeline, trustworthiness_report

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Thematic-LM on a column from the dataset."
    )
    p.add_argument(
        "--column", "-c",
        required=True,
        help="Name of the column to analyse (must contain free-text responses).",
    )
    p.add_argument(
        "--data", "-d",
        default="data/qualitative_data_VIS_tidy.xlsx",
        help="Path to the data file (.xlsx or .csv). Default: data/qualitative_data_VIS_tidy.xlsx",
    )
    p.add_argument(
        "--provider",
        default="gemini",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider. Default: gemini",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults: gemini-2.0-flash | gpt-4o | claude-sonnet-4-6"
        ),
    )
    p.add_argument(
        "--n-coders",
        type=int,
        default=2,
        help="Number of coder agents (default: 2).",
    )
    p.add_argument(
        "--n-theme-coders",
        type=int,
        default=2,
        help="Number of theme coder agents (default: 2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Items per coding batch (default: 10).",
    )
    p.add_argument(
        "--coder-batch-size",
        type=int,
        default=20,
        help="Items sent to coder agent in a single API call (default: 20).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of rows analysed (useful for quick tests).",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Sample rows randomly instead of taking the first N (use with --limit).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random sampling (default: 42).",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Root directory for output. Default: results/",
    )
    p.add_argument(
        "--top-k-similar",
        type=int,
        default=10,
        help="Similar codes retrieved per new code for the reviewer (default: 10). "
             "Reducing to 3-5 cuts ~60%% of tokens.",
    )
    p.add_argument(
        "--top-k-quotes",
        type=int,
        default=20,
        help="Max quotes stored per code/theme in the codebook (default: 20). "
             "Reducing to 5-8 shrinks the reviewer and theme coder inputs.",
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Requests per minute cap. Defaults: gemini=15, openai=500, anthropic=50. "
             "Set to 0 to disable.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the trustworthiness evaluation step.",
    )
    p.add_argument(
        "--estimate",
        action="store_true",
        help="Print a token and cost estimate then exit (no API calls made).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, column: str, limit: Optional[int], random: bool = False, seed: int = 42) -> list[dict]:
    path = Path(path)
    if not path.exists():
        sys.exit(f"Error: data file not found at '{path}'")

    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        sys.exit(f"Error: unsupported file type '{path.suffix}' (use .xlsx or .csv)")

    if column not in df.columns:
        close = [c for c in df.columns if column.lower() in c.lower()]
        hint = f"\n  Did you mean one of: {close}" if close else ""
        sys.exit(
            f"Error: column '{column}' not found in {path.name}.{hint}\n"
            f"  Available columns: {df.columns.tolist()}"
        )

    subset = df[["participant_id", column]].copy() if "participant_id" in df.columns \
        else df[[column]].copy()
    subset = subset.dropna(subset=[column])
    subset = subset[subset[column].astype(str).str.strip() != ""]

    if limit:
        if random:
            subset = subset.sample(n=min(limit, len(subset)), random_state=seed)
        else:
            subset = subset.head(limit)

    data = [
        {
            "id": str(row.get("participant_id", idx)),
            "text": str(row[column]).strip(),
        }
        for idx, row in subset.iterrows()
    ]
    return data


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}


def build_report_md(
    column: str,
    provider: str,
    model: str,
    n_rows: int,
    themes: list[dict],
    codebook_path: Path,
    trust_report: Optional[dict],
) -> str:
    import json as _json
    lines = []

    # Header
    lines += [
        f"# Thematic Analysis Report",
        f"",
        f"**Column:** `{column}`  ",
        f"**Responses analysed:** {n_rows}  ",
        f"**Model:** {provider} / {model}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        "---",
        "",
    ]

    # Trustworthiness scores
    if trust_report:
        lines += ["## Trustworthiness", ""]
        cred = trust_report.get("credibility_confirmability")
        if cred is not None:
            lines.append(f"- **Credibility & Confirmability:** {cred:.2f}")
        dep = trust_report.get("dependability", {})
        if dep:
            lines.append(f"- **Dependability (ROUGE):** {dep.get('rouge', 0):.2f}")
        trans = trust_report.get("transferability", {})
        if trans:
            lines.append(f"- **Transferability (ROUGE):** {trans.get('rouge', 0):.2f}")
        lines += ["", "---", ""]

    # Themes
    lines += [f"## Themes ({len(themes)} identified)", ""]

    for i, theme in enumerate(themes, 1):
        title = theme.get("theme", f"Theme {i}")
        description = theme.get("description", "")
        quotes = theme.get("quotes", [])
        quote_ids = theme.get("quote_ids", [])

        lines += [
            f"### {i}. {title}",
            "",
            f"{description}",
            "",
        ]

        if quotes:
            lines.append("**Example quotes:**")
            lines.append("")
            for j, (q, qid) in enumerate(zip(quotes[:3], quote_ids[:3]), 1):
                q_clean = q.strip().replace("\n", " ")
                lines.append(f"> {j}. *\"{q_clean}\"*  ")
                lines.append(f">    — Participant `{qid}`")
                lines.append("")

        lines.append("")

    # Codebook summary
    if codebook_path.exists():
        try:
            cb = _json.loads(codebook_path.read_text())
            lines += [
                "---",
                "",
                f"## Codebook Summary ({len(cb)} codes)",
                "",
                "| Code | Quotes |",
                "|---|---|",
            ]
            for code, data in list(cb.items())[:30]:
                n_quotes = len(data.get("quotes", []))
                lines.append(f"| {code} | {n_quotes} |")
            if len(cb) > 30:
                lines.append(f"| *… {len(cb) - 30} more codes* | |")
            lines.append("")
        except Exception:
            pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve model default
    model = args.model or DEFAULT_MODELS[args.provider]

    # --estimate: print cost breakdown and exit without calling any API
    if args.estimate:
        from thematic_lm.estimator import estimate
        report = estimate(
            data_path=args.data,
            column=args.column,
            n_coders=args.n_coders,
            n_theme_coders=args.n_theme_coders,
            batch_size=args.batch_size,
            coder_batch_size=args.coder_batch_size,
            top_k_similar=args.top_k_similar,
            top_k_quotes=args.top_k_quotes,
            limit=args.limit,
        )
        print(report.summary())
        sys.exit(0)

    # Load data
    print(f"Loading '{args.column}' from {args.data}...")
    data = load_data(args.data, args.column, args.limit, random=args.random, seed=args.seed)
    print(f"  {len(data)} non-empty responses found.")

    if not data:
        sys.exit("Error: no data to analyse after filtering empty rows.")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / f"{args.column}__{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    codebook_path = out_dir / "codebook.json"
    themes_path = out_dir / "themes.json"
    report_path = out_dir / "report.md"

    print(f"Output directory: {out_dir}")
    print(f"Provider: {args.provider} / {model}")
    print(f"Coders: {args.n_coders}  |  Theme coders: {args.n_theme_coders}  |  Batch size: {args.batch_size}  |  Coder batch: {args.coder_batch_size}")

    # Build client
    client_kwargs = {}
    if args.rpm is not None:
        client_kwargs["rpm"] = args.rpm
    client = LLMClient(provider=args.provider, model=model, **client_kwargs)

    # Run pipeline
    pipeline = ThematicLMPipeline(
        client=client,
        n_coders=args.n_coders,
        n_theme_coders=args.n_theme_coders,
        batch_size=args.batch_size,
        coder_batch_size=args.coder_batch_size,
        top_k_similar=args.top_k_similar,
        top_k_quotes=args.top_k_quotes,
        codebook_path=str(codebook_path),
    )

    themes = pipeline.run(data)

    # Save themes JSON
    themes_path.write_text(json.dumps(themes, indent=2, ensure_ascii=False))
    print(f"\nThemes saved → {themes_path}")

    # Trustworthiness evaluation
    trust_report = None
    if not args.skip_eval and themes:
        data_lookup = {item["id"]: item["text"] for item in data}
        trust_report = trustworthiness_report(
            client=client,
            themes=themes,
            data_lookup=data_lookup,
        )
        trust_path = out_dir / "trustworthiness.json"
        trust_path.write_text(json.dumps(trust_report, indent=2))
        print(f"Trustworthiness saved → {trust_path}")

    # Markdown report
    md = build_report_md(
        column=args.column,
        provider=args.provider,
        model=model,
        n_rows=len(data),
        themes=themes,
        codebook_path=codebook_path,
        trust_report=trust_report,
    )
    report_path.write_text(md, encoding="utf-8")
    print(f"Markdown report saved → {report_path}")

    # Print summary to terminal
    print(f"\n{'='*60}")
    print(f"  {len(themes)} themes identified from {len(data)} responses")
    print(f"{'='*60}")
    for i, t in enumerate(themes, 1):
        print(f"  {i}. {t.get('theme', '')}")
        print(f"     {t.get('description', '')[:80]}...")
    print()


if __name__ == "__main__":
    main()
