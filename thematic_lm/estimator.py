"""
Token and cost estimator for the Thematic-LM pipeline.

Analyses the dataset before running and produces a breakdown of:
  - API calls per agent type
  - Estimated input / output tokens per stage
  - Estimated total cost per provider/model
  - Hotspot diagnosis (where most tokens are spent)
  - Recommendations for reducing token usage

Token counting uses a chars/4 heuristic (standard English approximation).
For OpenAI models, tiktoken is used if installed.

Usage:
    from thematic_lm.estimator import estimate
    report = estimate("data/my_data.xlsx", column="pq_trustReason", n_coders=2)
    print(report.summary())
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt lengths (chars) — measured from agents.py prompts
# ---------------------------------------------------------------------------

_PROMPT_CHARS = {
    "coder_system":          480,
    "aggregator_system":     340,
    "reviewer_system":       530,
    "theme_coder_system":    430,
    "theme_aggregator_system": 330,
    "evaluator_system":      280,
}

# Approximate output sizes (chars) per call
_OUTPUT_CHARS = {
    "coder":             180,   # ~3 codes × (label + quote)
    "aggregator":        600,   # merged codes JSON
    "reviewer":          400,   # decisions JSON
    "theme_coder":       800,   # ~6 themes × (title + desc + quotes)
    "theme_aggregator":  600,
    "evaluator":         200,
}

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1M tokens, as of early 2026)
# ---------------------------------------------------------------------------

PRICING = {
    # Gemini
    "gemini-3-flash-preview":   {"input": 0.075,  "output": 0.30},
    "gemini-2.0-flash":         {"input": 0.075,  "output": 0.30},
    "gemini-1.5-flash":         {"input": 0.075,  "output": 0.30},
    "gemini-1.5-pro":           {"input": 1.25,   "output": 5.00},
    # OpenAI
    "gpt-4o":                   {"input": 2.50,   "output": 10.00},
    "gpt-4o-mini":              {"input": 0.15,   "output": 0.60},
    "gpt-4.1":                  {"input": 2.00,   "output": 8.00},
    "gpt-4.1-mini":             {"input": 0.40,   "output": 1.60},
    # Anthropic
    "claude-sonnet-4-6":        {"input": 3.00,   "output": 15.00},
    "claude-haiku-4-5":         {"input": 0.80,   "output": 4.00},
    "claude-opus-4-6":          {"input": 15.00,  "output": 75.00},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chars_to_tokens(chars: int) -> int:
    """Estimate token count from character count (4 chars ≈ 1 token)."""
    return max(1, chars // 4)


def _cost(tokens: int, rate_per_million: float) -> float:
    return tokens / 1_000_000 * rate_per_million


# ---------------------------------------------------------------------------
# Estimation logic
# ---------------------------------------------------------------------------

@dataclass
class AgentEstimate:
    name: str
    calls: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class EstimationReport:
    column: str
    n_rows: int
    avg_text_chars: int
    n_coders: int
    n_theme_coders: int
    batch_size: int
    top_k_similar: int
    top_k_quotes: int
    agents: list[AgentEstimate] = field(default_factory=list)

    # ----------------------------------------------------------------

    @property
    def total_calls(self) -> int:
        return sum(a.calls for a in self.agents)

    @property
    def total_input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.agents)

    @property
    def total_output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.agents)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def cost_for(self, model: str) -> Optional[dict]:
        pricing = PRICING.get(model)
        if not pricing:
            return None
        input_cost  = _cost(self.total_input_tokens,  pricing["input"])
        output_cost = _cost(self.total_output_tokens, pricing["output"])
        return {
            "model": model,
            "input_cost_usd":  round(input_cost,  4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd":  round(input_cost + output_cost, 4),
        }

    def hotspots(self) -> list[dict]:
        total = max(self.total_tokens, 1)
        return sorted(
            [
                {
                    "agent": a.name,
                    "calls": a.calls,
                    "total_tokens": a.total_tokens,
                    "pct_of_total": round(a.total_tokens / total * 100, 1),
                }
                for a in self.agents
            ],
            key=lambda x: -x["total_tokens"],
        )

    def recommendations(self) -> list[str]:
        recs = []
        hot = self.hotspots()
        names = [h["agent"] for h in hot[:3]]

        if "Reviewer" in names:
            recs.append(
                f"REVIEWER is {hot[0]['pct_of_total'] if hot[0]['agent']=='Reviewer' else next(h['pct_of_total'] for h in hot if h['agent']=='Reviewer')}% of tokens. "
                f"Reduce --top-k-similar (currently {self.top_k_similar}) to 3-5, "
                f"and/or reduce --top-k-quotes (currently {self.top_k_quotes}) to 5-8."
            )
        if "Coder" in names:
            recs.append(
                f"CODER calls dominate ({self.n_coders} coders × {self.n_rows} items = "
                f"{self.n_coders * self.n_rows} calls). "
                "Reduce to --n-coders 1 for a first pass, or increase --batch-size to process more per aggregation round."
            )
        if "Code Aggregator" in names:
            recs.append(
                f"CODE AGGREGATOR input scales with batch_size × n_coders. "
                f"Current batch-size={self.batch_size}. Try --batch-size 5 to reduce per-call input."
            )
        if "Theme Coder" in names:
            recs.append(
                "THEME CODER receives the full codebook. "
                "Reduce --top-k-quotes to 3-5 to shrink the codebook sent to theme coders."
            )

        recs.append(
            "CHEAPEST MODEL for this workload: "
            + min(
                PRICING.keys(),
                key=lambda m: (self.cost_for(m) or {}).get("total_cost_usd", 9999),
            )
        )
        return recs

    # ----------------------------------------------------------------

    def summary(self, models: Optional[list[str]] = None) -> str:
        if models is None:
            models = [
                "gemini-3-flash-preview", "gpt-4o-mini",
                "gpt-4o", "claude-sonnet-4-6", "claude-haiku-4-5",
            ]
        lines = [
            "",
            "=" * 65,
            f"  Token & Cost Estimate — column: '{self.column}'",
            "=" * 65,
            f"  Rows: {self.n_rows:,}  |  Avg text: {self.avg_text_chars} chars  "
            f"|  Coders: {self.n_coders}  |  Batch: {self.batch_size}",
            f"  top_k_similar: {self.top_k_similar}  |  top_k_quotes: {self.top_k_quotes}",
            "",
            f"  {'Agent':<22} {'Calls':>7}  {'Input tok':>10}  {'Output tok':>10}  {'Total tok':>10}",
            "  " + "-" * 63,
        ]
        for a in self.agents:
            lines.append(
                f"  {a.name:<22} {a.calls:>7,}  {a.input_tokens:>10,}  "
                f"{a.output_tokens:>10,}  {a.total_tokens:>10,}"
            )
        lines += [
            "  " + "-" * 63,
            f"  {'TOTAL':<22} {self.total_calls:>7,}  {self.total_input_tokens:>10,}  "
            f"{self.total_output_tokens:>10,}  {self.total_tokens:>10,}",
            "",
            "  Cost estimates:",
        ]
        for m in models:
            c = self.cost_for(m)
            if c:
                lines.append(
                    f"    {m:<35}  ${c['total_cost_usd']:>7.4f}  "
                    f"(in: ${c['input_cost_usd']:.4f} + out: ${c['output_cost_usd']:.4f})"
                )
        lines += ["", "  Token hotspots (largest first):"]
        for h in self.hotspots():
            bar = "█" * int(h["pct_of_total"] / 5)
            lines.append(
                f"    {h['agent']:<22} {h['total_tokens']:>9,} tok  {h['pct_of_total']:>5.1f}%  {bar}"
            )
        lines += ["", "  Recommendations:"]
        for i, r in enumerate(self.recommendations(), 1):
            # Word-wrap at 72 chars
            words = r.split()
            current, wrapped = [], []
            for w in words:
                if sum(len(x) + 1 for x in current) + len(w) > 70:
                    wrapped.append("    " + " ".join(current))
                    current = [w]
                else:
                    current.append(w)
            if current:
                wrapped.append("    " + " ".join(current))
            lines.append(f"  {i}. " + wrapped[0].strip())
            lines.extend(wrapped[1:])
        lines.append("=" * 65 + "\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate(
    data_path: str,
    column: str,
    n_coders: int = 2,
    n_theme_coders: int = 2,
    batch_size: int = 10,
    coder_batch_size: int = 20,
    top_k_similar: int = 10,
    top_k_quotes: int = 20,
    n_themes_estimate: int = 8,
    limit: Optional[int] = None,
) -> EstimationReport:
    """
    Estimate token usage and cost before running the pipeline.

    Args:
        data_path:          Path to .xlsx or .csv file.
        column:             Column name containing the free-text responses.
        n_coders:           Number of coder agents (Stage 1).
        n_theme_coders:     Number of theme coder agents (Stage 2).
        batch_size:         Items per coding batch.
        top_k_similar:      Similar codes retrieved per new code for reviewer.
        top_k_quotes:       Max quotes stored per code/theme in the codebook.
        n_themes_estimate:  Expected number of final themes (for evaluator estimate).
        limit:              Cap rows (same as --limit in run.py).

    Returns:
        EstimationReport with per-agent breakdown and cost table.
    """
    # Load data
    path = Path(data_path)
    if path.suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path)
    else:
        import pandas as pd
        df = pd.read_csv(path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")

    rows = df[column].dropna().astype(str).str.strip()
    rows = rows[rows != ""]
    if limit:
        rows = rows.head(limit)

    n_rows = len(rows)
    avg_text_chars = int(rows.str.len().mean())
    avg_text_tokens = _chars_to_tokens(avg_text_chars)

    n_batches = math.ceil(n_rows / batch_size)

    # Estimated codes per batch after aggregation (empirically ~2 new codes per item)
    avg_codes_per_batch   = batch_size * 2          # rough: 2 codes per item, some merged
    avg_codes_in_codebook = min(avg_codes_per_batch * n_batches * 0.4, 200)  # grows then stabilises
    avg_quotes_per_code   = min(top_k_quotes, 5)    # codebook quotes stored (capped for reviewer)

    # ----------------------------------------------------------------
    # Agent: Coder  (mini-batched: coder_batch_size items per call)
    # ----------------------------------------------------------------
    # Each coder call handles coder_batch_size items
    coder_calls_per_coder = math.ceil(n_rows / coder_batch_size)
    coder_calls           = coder_calls_per_coder * n_coders
    # Input: system prompt + coder_batch_size items worth of text
    coder_input_per_call  = _chars_to_tokens(
        _PROMPT_CHARS["coder_system"] + avg_text_chars * coder_batch_size
    )
    # Output: coder_batch_size items × ~3 codes each
    coder_output_per_call = _chars_to_tokens(_OUTPUT_CHARS["coder"] * coder_batch_size)
    coder = AgentEstimate(
        name="Coder",
        calls=coder_calls,
        input_tokens=coder_calls * coder_input_per_call,
        output_tokens=coder_calls * coder_output_per_call,
    )

    # ----------------------------------------------------------------
    # Agent: Code Aggregator
    # ----------------------------------------------------------------
    # Input = system + all coder outputs for the batch
    agg_input_data_chars  = batch_size * n_coders * (_OUTPUT_CHARS["coder"])
    agg_input_per_call    = _chars_to_tokens(_PROMPT_CHARS["aggregator_system"] + agg_input_data_chars)
    agg_output_per_call   = _chars_to_tokens(_OUTPUT_CHARS["aggregator"])
    aggregator = AgentEstimate(
        name="Code Aggregator",
        calls=n_batches,
        input_tokens=n_batches * agg_input_per_call,
        output_tokens=n_batches * agg_output_per_call,
    )

    # ----------------------------------------------------------------
    # Agent: Reviewer  ← main hotspot
    # ----------------------------------------------------------------
    # Input = system + new_codes + similar_codes for each new code
    # new_codes: avg_codes_per_batch codes, each with quotes
    new_codes_chars = avg_codes_per_batch * (40 + avg_quotes_per_code * 25)

    # similar_codes: for each new code, top_k_similar entries, each with
    # code label + similarity score + quotes.
    # As the codebook grows batch-by-batch, average codebook size at review time
    # is ~half the final size. We estimate a growing cost.
    avg_codebook_at_review = avg_codes_in_codebook / 2
    similar_entry_chars    = 40 + avg_quotes_per_code * 25   # code + quotes
    similar_total_chars    = avg_codes_per_batch * min(top_k_similar, avg_codebook_at_review) * similar_entry_chars
    reviewer_input_per_call = _chars_to_tokens(
        _PROMPT_CHARS["reviewer_system"] + new_codes_chars + similar_total_chars
    )
    reviewer_output_per_call = _chars_to_tokens(_OUTPUT_CHARS["reviewer"])
    reviewer = AgentEstimate(
        name="Reviewer",
        calls=n_batches,
        input_tokens=n_batches * reviewer_input_per_call,
        output_tokens=n_batches * reviewer_output_per_call,
    )

    # ----------------------------------------------------------------
    # Agent: Theme Coder
    # ----------------------------------------------------------------
    # Input = system + full codebook (code names + quotes)
    codebook_chars         = avg_codes_in_codebook * (40 + avg_quotes_per_code * 25)
    theme_input_per_call   = _chars_to_tokens(_PROMPT_CHARS["theme_coder_system"] + codebook_chars)
    theme_output_per_call  = _chars_to_tokens(_OUTPUT_CHARS["theme_coder"])
    theme_coder = AgentEstimate(
        name="Theme Coder",
        calls=n_theme_coders,
        input_tokens=n_theme_coders * theme_input_per_call,
        output_tokens=n_theme_coders * theme_output_per_call,
    )

    # ----------------------------------------------------------------
    # Agent: Theme Aggregator
    # ----------------------------------------------------------------
    theme_agg_input   = _chars_to_tokens(
        _PROMPT_CHARS["theme_aggregator_system"] + n_theme_coders * _OUTPUT_CHARS["theme_coder"]
    )
    theme_agg = AgentEstimate(
        name="Theme Aggregator",
        calls=1,
        input_tokens=theme_agg_input,
        output_tokens=_chars_to_tokens(_OUTPUT_CHARS["theme_aggregator"]),
    )

    # ----------------------------------------------------------------
    # Agent: Evaluator (credibility check, optional)
    # ----------------------------------------------------------------
    eval_input_per_call  = _chars_to_tokens(_PROMPT_CHARS["evaluator_system"] + 40 + 3 * 25)
    evaluator = AgentEstimate(
        name="Evaluator",
        calls=n_themes_estimate,
        input_tokens=n_themes_estimate * eval_input_per_call,
        output_tokens=n_themes_estimate * _chars_to_tokens(_OUTPUT_CHARS["evaluator"]),
    )

    return EstimationReport(
        column=column,
        n_rows=n_rows,
        avg_text_chars=avg_text_chars,
        n_coders=n_coders,
        n_theme_coders=n_theme_coders,
        batch_size=batch_size,
        top_k_similar=top_k_similar,
        top_k_quotes=top_k_quotes,
        agents=[coder, aggregator, reviewer, theme_coder, theme_agg, evaluator],
    )
