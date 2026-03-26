"""
LLM agent functions implementing the Thematic-LM roles from the paper:
  - coder_agent         : assigns 1-3 codes + quotes to a single data item
  - code_aggregator     : merges codes from multiple coders into JSON
  - reviewer_agent      : compares new codes to similar codes and updates codebook
  - theme_coder_agent   : identifies themes from the codebook
  - theme_aggregator    : merges themes from multiple theme coders
  - evaluator_agent     : checks consistency between a theme and its quotes

All functions accept an LLMClient instance as their first argument.
"""

from __future__ import annotations
import json
from typing import Optional
from .llm import LLMClient


# ---------------------------------------------------------------------------
# Prompts (from Appendix B of the paper, lightly adapted)
# ---------------------------------------------------------------------------

_CODER_SYSTEM = """You are a coder in thematic analysis of social media data.
When given a social media post, write 1-3 codes for the post. The code should \
capture concepts or ideas with the most analytical interest.
For each code, extract a representative quote from the post. The quote must be \
a verbatim extract from the text.
{identity_block}
Respond with a JSON object in this exact format:
{{
  "codes": [
    {{"code": "<short label>", "quote": "<verbatim quote from text>"}},
    ...
  ]
}}"""

_CODER_BATCH_SYSTEM = """You are a coder in thematic analysis of social media data.
You will receive multiple posts, each with a unique ID.
For EACH post independently, write 1-3 codes that capture concepts or ideas \
with the most analytical interest.
For each code, extract a verbatim quote from THAT SPECIFIC post only — never \
mix quotes across posts.
{identity_block}
Respond with a JSON object in this exact format:
{{
  "items": [
    {{
      "id": "<post id exactly as given>",
      "codes": [
        {{"code": "<short label>", "quote": "<verbatim quote from this post only>"}},
        ...
      ]
    }},
    ...
  ]
}}
Include one entry for every post ID provided. Output only valid JSON."""

_AGGREGATOR_SYSTEM = """You are an aggregator in thematic analysis of social media data.
Your job is to take codes and their corresponding quotes from multiple coders, \
merge codes with similar meanings (retaining distinctions where important), \
and keep the top {top_k} most relevant quotes per code.
Respond with a JSON object:
{{
  "codes": [
    {{
      "code": "<merged label>",
      "quotes": ["<quote1>", ...],
      "quote_ids": ["<id1>", ...]
    }},
    ...
  ]
}}
Output only valid JSON. Nothing else."""

_REVIEWER_SYSTEM = """You are a reviewer in thematic analysis of social media data.
You will receive two items:
1. "new_codes": fresh codes and quotes just produced by a coder.
2. "similar_codes": existing codes from the codebook that are semantically close.

Your job:
- Decide whether each new code can be merged with or updated to match an existing code.
- If an existing code already covers the same concept, use its label (possibly refined).
- If the new code is genuinely different, keep it as-is.
- If multiple similar codes should be merged together, list them.

Respond with a JSON object:
{{
  "decisions": [
    {{
      "new_code": "<the new code label>",
      "action": "keep" | "update" | "merge",
      "final_code": "<label to use in codebook>",
      "merge_with": ["<existing code1>", ...]   // only for action=merge
    }},
    ...
  ]
}}
Output only valid JSON. Nothing else."""

_THEME_CODER_SYSTEM = """You are a theme coder in thematic analysis of social media data.
You will receive a codebook in JSON containing codes and sample quotes.
Identify overarching themes that reflect deeper meanings of the data.
For each theme, provide:
  - A short title
  - One sentence describing what the theme captures
  - The {top_k} most relevant quotes (with their IDs)
{identity_block}
Respond with a JSON object:
{{
  "themes": [
    {{
      "theme": "<title>",
      "description": "<one sentence>",
      "quotes": ["<quote1>", ...],
      "quote_ids": ["<id1>", ...]
    }},
    ...
  ]
}}
Output only valid JSON. Nothing else."""

_THEME_AGGREGATOR_SYSTEM = """You are a theme aggregator in thematic analysis.
Merge themes with overlapping meanings from multiple coders into a final coherent set.
Retain distinct perspectives. Keep the top {top_k} quotes per theme.
Respond with a JSON object:
{{
  "themes": [
    {{
      "theme": "<title>",
      "description": "<one sentence>",
      "quotes": ["<quote1>", ...],
      "quote_ids": ["<id1>", ...]
    }},
    ...
  ]
}}
Output only valid JSON. Nothing else."""

_EVALUATOR_SYSTEM = """You are an evaluator in qualitative thematic analysis.
You will be given a theme name, its description, and a set of quotes associated with it.
For each quote, decide whether it is consistent with the theme (True) or not (False).
Respond with a JSON object:
{{
  "results": [
    {{"quote": "<quote text>", "consistent": true | false}},
    ...
  ]
}}
Output only valid JSON. Nothing else."""


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------

def coder_agent_batch(
    client: LLMClient,
    items: list[dict],
    identity: Optional[str] = None,
) -> list[dict]:
    """
    Code multiple pieces of text in a single API call.

    Args:
        items: List of {"id": str, "text": str} dicts.
        identity: Optional identity string for the coder.

    Returns:
        List of {"code": str, "quote": str, "quote_id": str}
    """
    if not items:
        return []

    identity_block = (
        f"\nYou approach this analysis as someone who identifies as: {identity}\n"
        "Let your background inform how you interpret and label the data."
        if identity
        else ""
    )
    system = _CODER_BATCH_SYSTEM.format(identity_block=identity_block)

    posts_text = "\n\n".join(
        f"Post ID: {item['id']}\n{item['text']}" for item in items
    )
    messages = [{"role": "user", "content": posts_text}]

    try:
        result = client.complete_json(messages, system=system)
        output = []
        for entry in result.get("items", []):
            item_id = str(entry.get("id", ""))
            for c in entry.get("codes", []):
                if "code" in c and "quote" in c:
                    output.append({"code": c["code"], "quote": c["quote"], "quote_id": item_id})
        return output
    except Exception as e:
        print(f"[coder_agent_batch] Error (batch size {len(items)}): {e} — falling back to individual calls")
        output = []
        for item in items:
            output.extend(coder_agent(client, item["text"], item["id"], identity))
        return output


def coder_agent(
    client: LLMClient,
    text: str,
    quote_id: str,
    identity: Optional[str] = None,
) -> list[dict]:
    """
    Code a single piece of text.

    Returns:
        List of {"code": str, "quote": str, "quote_id": str}
    """
    identity_block = (
        f"\nYou approach this analysis as someone who identifies as: {identity}\n"
        "Let your background inform how you interpret and label the data."
        if identity
        else ""
    )
    system = _CODER_SYSTEM.format(identity_block=identity_block)
    messages = [{"role": "user", "content": f"Post ID: {quote_id}\n\nText:\n{text}"}]

    try:
        result = client.complete_json(messages, system=system)
        codes = result.get("codes", [])
        return [
            {"code": c["code"], "quote": c["quote"], "quote_id": quote_id}
            for c in codes
            if "code" in c and "quote" in c
        ]
    except Exception as e:
        print(f"[coder_agent] Error on id={quote_id}: {e}")
        return []


def code_aggregator(
    client: LLMClient,
    all_coder_outputs: list[list[dict]],
    top_k: int = 20,
) -> list[dict]:
    """
    Merge codes from multiple coders.

    Args:
        all_coder_outputs: List of coder results (each is a list of code dicts).

    Returns:
        List of {"code": str, "quotes": [...], "quote_ids": [...]}
    """
    system = _AGGREGATOR_SYSTEM.format(top_k=top_k)
    payload = json.dumps(
        [{"coder": i + 1, "codes": outputs} for i, outputs in enumerate(all_coder_outputs)],
        indent=2,
    )
    messages = [{"role": "user", "content": payload}]

    try:
        result = client.complete_json(messages, system=system)
        return result.get("codes", [])
    except Exception as e:
        print(f"[code_aggregator] Error: {e}")
        # Fallback: flatten without merging
        flat = {}
        for outputs in all_coder_outputs:
            for item in outputs:
                code = item["code"]
                if code not in flat:
                    flat[code] = {"code": code, "quotes": [], "quote_ids": []}
                flat[code]["quotes"].append(item["quote"])
                flat[code]["quote_ids"].append(item["quote_id"])
        return list(flat.values())


def reviewer_agent(
    client: LLMClient,
    new_codes: list[dict],
    similar_codes: list[dict],
) -> list[dict]:
    """
    Compare new codes against similar existing codes and decide how to update.

    Args:
        new_codes: List of {"code", "quotes", "quote_ids"} from the aggregator.
        similar_codes: List of {"new_code", "similar": [...]} from codebook lookup.

    Returns:
        List of decisions: {"new_code", "action", "final_code", "merge_with"}
    """
    payload = json.dumps(
        {"new_codes": new_codes, "similar_codes": similar_codes}, indent=2
    )
    messages = [{"role": "user", "content": payload}]

    try:
        result = client.complete_json(messages, system=_REVIEWER_SYSTEM)
        return result.get("decisions", [])
    except Exception as e:
        print(f"[reviewer_agent] Error: {e}")
        # Fallback: keep all codes as-is
        return [
            {"new_code": c["code"], "action": "keep", "final_code": c["code"], "merge_with": []}
            for c in new_codes
        ]


def theme_coder_agent(
    client: LLMClient,
    codebook_json: str,
    identity: Optional[str] = None,
    top_k: int = 20,
) -> list[dict]:
    """
    Identify themes from the codebook.

    Returns:
        List of {"theme": str, "description": str, "quotes": [...], "quote_ids": [...]}
    """
    identity_block = (
        f"\nYou approach this analysis as someone who identifies as: {identity}\n"
        "Let your background inform how you interpret patterns across the codes."
        if identity
        else ""
    )
    system = _THEME_CODER_SYSTEM.format(top_k=top_k, identity_block=identity_block)
    messages = [{"role": "user", "content": f"Codebook:\n{codebook_json}"}]

    try:
        result = client.complete_json(messages, system=system)
        return result.get("themes", [])
    except Exception as e:
        print(f"[theme_coder_agent] Error: {e}")
        return []


def theme_aggregator(
    client: LLMClient,
    all_theme_outputs: list[list[dict]],
    top_k: int = 20,
) -> list[dict]:
    """
    Merge themes from multiple theme coders into a final set.

    Returns:
        List of {"theme": str, "description": str, "quotes": [...], "quote_ids": [...]}
    """
    system = _THEME_AGGREGATOR_SYSTEM.format(top_k=top_k)
    payload = json.dumps(
        [{"coder": i + 1, "themes": themes} for i, themes in enumerate(all_theme_outputs)],
        indent=2,
    )
    messages = [{"role": "user", "content": payload}]

    try:
        result = client.complete_json(messages, system=system)
        return result.get("themes", [])
    except Exception as e:
        print(f"[theme_aggregator] Error: {e}")
        # Fallback: concatenate all themes
        all_themes = []
        for themes in all_theme_outputs:
            all_themes.extend(themes)
        return all_themes


def evaluator_agent(
    client: LLMClient,
    theme: str,
    description: str,
    quotes: list[str],
) -> float:
    """
    Check consistency between a theme and its supporting quotes (credibility).

    Returns:
        Fraction of quotes consistent with the theme (0.0–1.0).
    """
    if not quotes:
        return 0.0

    payload = json.dumps(
        {"theme": theme, "description": description, "quotes": quotes}, indent=2
    )
    messages = [{"role": "user", "content": payload}]

    try:
        result = client.complete_json(messages, system=_EVALUATOR_SYSTEM)
        results = result.get("results", [])
        if not results:
            return 0.0
        consistent = sum(1 for r in results if r.get("consistent", False))
        return consistent / len(results)
    except Exception as e:
        print(f"[evaluator_agent] Error: {e}")
        return 0.0
