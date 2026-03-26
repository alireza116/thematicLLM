"""
Trustworthiness evaluation framework (paper Section 3.1).

Implements the three principles:
  1. Credibility & Confirmability  — LLM-as-judge consistency check
  2. Dependability                 — ROUGE overlap across repeated runs
  3. Transferability               — ROUGE overlap between train/val splits
"""

from __future__ import annotations
import json
from typing import Optional
from rouge_score import rouge_scorer

from .llm import LLMClient
from .agents import evaluator_agent


# ---------------------------------------------------------------------------
# 1. Credibility & Confirmability
# ---------------------------------------------------------------------------

def evaluate_credibility(
    client: LLMClient,
    themes: list[dict],
    data_lookup: Optional[dict[str, str]] = None,
) -> float:
    """
    Use an LLM evaluator to check whether each theme is consistent with its quotes.

    Args:
        client: LLMClient instance.
        themes: List of theme dicts {"theme", "description", "quotes", "quote_ids"}.
        data_lookup: Optional {quote_id: original_text} mapping. If provided, the
                     original texts are used instead of the stored quote snippets.

    Returns:
        Overall credibility score (0.0–1.0).
    """
    if not themes:
        return 0.0

    scores = []
    for theme in themes:
        quotes = theme.get("quotes", [])
        quote_ids = theme.get("quote_ids", [])

        # Prefer full original texts when available
        if data_lookup and quote_ids:
            resolved = [data_lookup.get(qid, q) for qid, q in zip(quote_ids, quotes)]
        else:
            resolved = quotes

        score = evaluator_agent(
            client,
            theme=theme.get("theme", ""),
            description=theme.get("description", ""),
            quotes=resolved,
        )
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# 2. Dependability — ROUGE across repeated runs
# ---------------------------------------------------------------------------

def _theme_set_to_text(themes: list[dict]) -> str:
    """Flatten a list of themes into a single string for ROUGE comparison."""
    parts = []
    for t in themes:
        parts.append(t.get("theme", ""))
        parts.append(t.get("description", ""))
    return " ".join(parts)


def _pairwise_rouge(theme_runs: list[list[dict]]) -> dict[str, float]:
    """
    Compute average pairwise ROUGE-1 and ROUGE-2 across all run pairs.
    Follows Equations (1)–(3) from the paper.
    """
    scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    rouge1_scores, rouge2_scores = [], []

    n = len(theme_runs)
    for i in range(n):
        for j in range(i + 1, n):
            text_a = _theme_set_to_text(theme_runs[i])
            text_b = _theme_set_to_text(theme_runs[j])
            if not text_a or not text_b:
                continue

            # A→B
            scores_ab = scorer_obj.score(text_a, text_b)
            # B→A
            scores_ba = scorer_obj.score(text_b, text_a)

            r1 = (scores_ab["rouge1"].fmeasure + scores_ba["rouge1"].fmeasure) / 2
            r2 = (scores_ab["rouge2"].fmeasure + scores_ba["rouge2"].fmeasure) / 2
            rouge1_scores.append(r1)
            rouge2_scores.append(r2)

    if not rouge1_scores:
        return {"rouge1": 0.0, "rouge2": 0.0, "rouge": 0.0}

    r1_avg = sum(rouge1_scores) / len(rouge1_scores)
    r2_avg = sum(rouge2_scores) / len(rouge2_scores)
    return {"rouge1": r1_avg, "rouge2": r2_avg, "rouge": (r1_avg + r2_avg) / 2}


def evaluate_dependability(theme_runs: list[list[dict]]) -> dict[str, float]:
    """
    Measure dependability by computing average pairwise ROUGE across N runs.

    Args:
        theme_runs: List of N theme sets (each from a separate pipeline run).

    Returns:
        {"rouge1": float, "rouge2": float, "rouge": float}
    """
    if len(theme_runs) < 2:
        raise ValueError("Need at least 2 runs to measure dependability.")
    return _pairwise_rouge(theme_runs)


# ---------------------------------------------------------------------------
# 3. Transferability — ROUGE between train/val splits
# ---------------------------------------------------------------------------

def evaluate_transferability(
    train_themes: list[dict],
    val_themes: list[dict],
) -> dict[str, float]:
    """
    Measure transferability by computing ROUGE between themes from two data splits.

    Args:
        train_themes: Themes identified on the training split.
        val_themes:   Themes identified on the validation split.

    Returns:
        {"rouge1": float, "rouge2": float, "rouge": float}
    """
    return _pairwise_rouge([train_themes, val_themes])


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def trustworthiness_report(
    client: LLMClient,
    themes: list[dict],
    theme_runs: Optional[list[list[dict]]] = None,
    train_themes: Optional[list[dict]] = None,
    val_themes: Optional[list[dict]] = None,
    data_lookup: Optional[dict[str, str]] = None,
) -> dict:
    """
    Compute all applicable trustworthiness scores and return as a dict.

    Args:
        client: LLMClient for credibility evaluation.
        themes: The primary set of themes to evaluate.
        theme_runs: Additional runs for dependability (include `themes` as one run).
        train_themes / val_themes: Train-val split themes for transferability.
        data_lookup: {quote_id: text} mapping for credibility evaluation.

    Returns:
        {
          "credibility_confirmability": float,
          "dependability": {"rouge1", "rouge2", "rouge"},  // if theme_runs provided
          "transferability": {"rouge1", "rouge2", "rouge"}, // if splits provided
        }
    """
    report = {}

    print("Evaluating credibility & confirmability...")
    report["credibility_confirmability"] = evaluate_credibility(client, themes, data_lookup)

    if theme_runs and len(theme_runs) >= 2:
        print("Evaluating dependability...")
        report["dependability"] = evaluate_dependability(theme_runs)

    if train_themes and val_themes:
        print("Evaluating transferability...")
        report["transferability"] = evaluate_transferability(train_themes, val_themes)

    return report
