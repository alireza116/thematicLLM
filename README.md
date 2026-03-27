# Thematic-LM

Implementation of **Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis** (Qiao et al., WWW '25).

> Qiao, T., Walker, C., Cunningham, C., & Koh, Y. S. (2025). Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis. *Proceedings of the ACM Web Conference 2025 (WWW '25)*. https://doi.org/10.1145/3696410.3714595

---

## Overview

This tool runs inductive thematic analysis (Braun & Clarke, 2006) on free-text survey responses using a multi-agent LLM system. It is designed for online study data — open-ended survey questions, think-aloud responses, or any short text corpus — and scales to hundreds of responses without requiring manual coding.

The pipeline produces:
- A structured **codebook** of recurring concepts found in the data
- A set of **themes** — higher-order patterns that capture what the data is saying
- A **trustworthiness report** with credibility, dependability, and transferability scores
- A **markdown report** summarising themes, codes, and example quotes

---

## Project Structure

```
qual/
├── data/                        # Input data files (xlsx or csv)
│   └── qualitative_data_VIS_tidy.xlsx
│
├── study_context.txt            # (you create this) Study description for the LLM agents
│
├── results/                     # Output — created automatically on each run
│   └── <column>__<timestamp>/
│       ├── themes.json          # Final themes with quotes
│       ├── codebook.json        # Adaptive codebook built during coding
│       ├── trustworthiness.json # Credibility / dependability / transferability scores
│       └── report.md            # Human-readable markdown summary
│
├── thematic_lm/                 # Core library
│   ├── __init__.py
│   ├── llm.py                   # LLM provider abstraction (OpenAI / Anthropic / Gemini)
│   ├── codebook.py              # Adaptive codebook with embedding-based retrieval
│   ├── agents.py                # All agent functions and system prompts
│   ├── identities.py            # Coder identity profiles
│   ├── evaluation.py            # Trustworthiness metrics
│   ├── estimator.py             # Token and cost estimator
│   └── pipeline.py              # ThematicLMPipeline orchestration
│
├── run.py                       # Main entry point — run analysis on one column
├── run_all.py                   # Batch runner for multiple columns
├── requirements.txt
└── .env                         # API keys (never commit this file)
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root (it is git-ignored):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

You only need the key for the provider you intend to use. The default provider is **Gemini**.

### 3. Add your data

Place your data file in the `data/` folder. Supported formats: `.xlsx` and `.csv`.

The file must have at least one column of free-text responses. One column is analysed per run.

### 4. Write a study context file

Create `study_context.txt` in the project root and describe:
- What the study is about
- The specific survey question being analysed
- Any framing that helps the LLM understand what kinds of patterns are analytically relevant

**Example `study_context.txt`:**
```
This study investigates how people perceive and evaluate trust in data visualizations.
Participants were shown a series of charts and asked open-ended questions about their
reactions.

The column being analysed contains responses to the question:
"Why do you trust or distrust this visualization?"

Relevant themes may relate to: data source credibility, visual design choices,
prior knowledge of the topic, clarity and transparency of the chart, and emotional
or intuitive reactions.
```

The study context is injected into the **Coder** and **Theme Coder** agent prompts. It helps the agents focus their coding on aspects of the text that are analytically meaningful for your specific research questions, rather than coding everything equally.

---

## Running the Analysis

### Single column with study context

Create a plain text file (e.g. `study-context.txt`) that describes the study and the specific question being asked in that column:

```
This study investigates how people perceive and evaluate trust in data visualizations.
Participants were shown a series of charts displaying political and social trends.

The column being analysed contains responses to the question:
"Why do you trust or distrust this visualization?"
```

Then run:

```bash
python run.py --column pq_trustReason --study-context study-context.txt
```

`--study-context` accepts either a file path or a literal string — both work:

```bash
python run.py --column pq_trustReason \
  --study-context "Study on trust in data visualizations. Question: Why do you trust or distrust this visualization?"
```

### Quick test before a full run (random sample)

```bash
python run.py --column pq_trustReason --limit 20 --random --study-context study-context.txt
```

### All columns with per-column study context

`run_all.py` is configured with a shared base context and a per-column question for each column in the dataset. Edit the two variables at the top of `run_all.py` before running:

```python
# run_all.py — edit these before running

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
    "viz_summary_trend_pres_approval": (
        "The column contains participants' written summaries of a visualization "
        "showing presidential approval rating trends over time."
    ),
    # ... one entry per column
}
```

Each column receives `BASE_STUDY_CONTEXT` + its own entry from `COLUMN_CONTEXTS` combined. If a column has no entry in `COLUMN_CONTEXTS`, only the base context is used.

Run all columns:

```bash
python run_all.py
```

Test mode (20 random rows per column, useful to validate before the full run):

```bash
python run_all.py --test
```

Run a specific subset of columns:

```bash
python run_all.py --columns pq_trustReason pq_surpriseReason
```

### Full options

```
python run.py \
  --column        pq_trustReason        # Column to analyse (required)
  --data          data/myfile.xlsx      # Data file path (default: data/qualitative_data_VIS_tidy.xlsx)
  --study-context study_context.txt     # Path to study context text file
  --provider      gemini                # LLM provider: gemini | openai | anthropic
  --model         gemini-2.0-flash      # Model name (defaults per provider listed below)
  --n-coders      2                     # Number of independent coder agents
  --n-theme-coders 2                    # Number of independent theme coder agents
  --batch-size    10                    # Items per aggregation/reviewer batch
  --coder-batch-size 20                 # Items sent to coder in one API call
  --top-k-similar 10                    # Similar codes retrieved per new code (reviewer)
  --top-k-quotes  20                    # Max quotes stored per code/theme
  --limit         200                   # Cap number of rows (omit to use all)
  --random                              # Sample randomly (use with --limit)
  --seed          42                    # Random seed for --random
  --rpm           15                    # Requests per minute cap (0 = disable)
  --skip-eval                           # Skip trustworthiness evaluation
  --estimate                            # Print token/cost estimate and exit
  --results-dir   results               # Root folder for output
```

**Default models by provider:**

| Provider | Default model |
|---|---|
| `gemini` | `gemini-2.0-flash` |
| `openai` | `gpt-4o` |
| `anthropic` | `claude-sonnet-4-6` |

### Estimate tokens and cost before running

```bash
python run.py --column pq_trustReason --estimate
```

This prints a breakdown of estimated API calls, tokens, and cost without making any real API calls.

---

## How the Analysis Works

The pipeline runs in two sequential stages.

### Stage 1: Coding

The dataset is processed in batches (default: 10 items per batch). For each batch:

1. **Coder agents** (default: 2) independently read each text response and assign 1–3 short codes, each with a verbatim quote as evidence. When a study context is provided, coders focus on aspects of the text relevant to the research questions.

2. **Code Aggregator** merges the coders' outputs, combining codes with similar meanings into a unified set for the batch.

3. **Reviewer** queries the adaptive codebook for the top-*k* most similar existing codes (via sentence embedding cosine similarity), then decides whether each new code should be kept as-is, updated to match an existing label, or merged with existing entries.

The **adaptive codebook** accumulates across the full dataset. By the end of Stage 1 it contains a comprehensive, deduplicated set of codes grounded in evidence from all responses.

### Stage 2: Theme Development

1. **Theme Coder agents** (default: 2) each receive the full codebook and independently identify overarching themes — patterns that capture deeper meanings across many codes.

2. **Theme Aggregator** merges the theme coders' outputs into a final coherent set.

### Key parameters to tune

| Parameter | Effect | Recommendation |
|---|---|---|
| `--n-coders` | More coders → broader perspective, more API calls | 2–3 for most datasets |
| `--batch-size` | Larger batches → fewer reviewer calls, may reduce precision | 10–20 |
| `--coder-batch-size` | Larger → fewer coder API calls, faster | 15–20 for short responses |
| `--top-k-similar` | Higher → reviewer sees more context, uses more tokens | 5–10; reduce to cut costs |
| `--top-k-quotes` | Higher → more evidence per code, uses more tokens | 8–20 |

---

## Output Files

All output is written to `results/<column>__<timestamp>/`.

### `themes.json`

Array of theme objects:
```json
[
  {
    "theme": "Trust Through Source Credibility",
    "description": "Participants cited the perceived authority or reliability of the data source as the primary basis for trusting a visualization.",
    "quotes": ["The data comes from a government agency so I trust it.", "..."],
    "quote_ids": ["42", "107", "..."]
  }
]
```

### `codebook.json`

The adaptive codebook built during Stage 1. Each entry contains the code label, supporting quotes, source IDs, and the sentence embedding vector. This file can be passed back via `--codebook-path` to resume an interrupted run.

### `trustworthiness.json`

Trustworthiness scores following Lincoln & Guba (1985), adapted for LLM-based TA:

| Metric | What it measures |
|---|---|
| **Credibility** | Fraction of theme quotes that are genuinely consistent with the theme (LLM-as-judge) |
| **Confirmability** | Same as credibility; checks whether findings are grounded in the data |
| **Dependability** | ROUGE overlap between two independent runs — stability of themes across repetitions |
| **Transferability** | ROUGE overlap between themes derived from two halves of the dataset — generalisability |

### `report.md`

A human-readable markdown report with:
- All themes and their descriptions
- The codes that support each theme, with example quotes
- A full codebook table

---

## Coder Identities

By default, coders have no assigned identity. You can optionally assign each coder a different perspective, which increases the diversity of codes and surfaces themes that a single-perspective analysis might miss.

Pre-defined identity sets are in `thematic_lm/identities.py`:

```python
from thematic_lm import CLIMATE_IDENTITIES, STRESS_IDENTITIES, CODER_IDENTITIES
```

To use them:

```python
from thematic_lm import LLMClient, ThematicLMPipeline, CODER_IDENTITIES

pipeline = ThematicLMPipeline(
    client=client,
    n_coders=len(CODER_IDENTITIES),
    coder_identities=list(CODER_IDENTITIES.values()),
)
```

You can also define custom identities as plain strings:

```python
identities = [
    "a data visualization researcher focused on perceptual accuracy",
    "a general public user with limited statistical background",
    "a journalist evaluating whether a chart is misleading",
]
pipeline = ThematicLMPipeline(client=client, n_coders=3, coder_identities=identities)
```

---

## Architecture

```
                      ┌─────────────────────────────────────────────────┐
                      │              STAGE 1: CODING                    │
                      │                                                 │
Text Data (batched)   │   Coder 1 ─┐                                   │
──────────────────►   │   Coder 2 ─┼─► Code Aggregator ─► Reviewer ─► │─► Codebook
                      │   Coder N ─┘       (merge)        (retrieve     │
                      │                                    & update)    │
                      └─────────────────────────────────────────────────┘
                                                │
                                          Codebook (finalized)
                                                │
                      ┌─────────────────────────▼───────────────────────┐
                      │           STAGE 2: THEME DEVELOPMENT            │
                      │                                                 │
                      │  Theme Coder 1 ─┐                              │
                      │  Theme Coder 2 ─┼─► Theme Aggregator ─────────►│─► Final Themes
                      │  Theme Coder M ─┘       (merge)                │
                      │                                                 │
                      └─────────────────────────────────────────────────┘
```

### Agent roles

| Agent | Stage | Receives study context? | Role |
|---|---|---|---|
| **Coder** | 1 | Yes | Assigns 1–3 codes + verbatim quotes to each text response |
| **Code Aggregator** | 1 | No | Merges coder outputs, deduplicates similar codes |
| **Reviewer** | 1 | No | Maintains the adaptive codebook via keep/update/merge decisions |
| **Theme Coder** | 2 | Yes | Identifies overarching themes from the codebook |
| **Theme Aggregator** | 2 | No | Merges theme coders' outputs into the final theme set |
| **Evaluator** | Eval | No | LLM-as-judge: checks quote–theme consistency for credibility scoring |

---

## Citation

```bibtex
@inproceedings{qiao2025thematiclm,
  title     = {Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis},
  author    = {Qiao, Tingrui and Walker, Caroline and Cunningham, Chris and Koh, Yun Sing},
  booktitle = {Proceedings of the ACM Web Conference 2025 (WWW '25)},
  year      = {2025},
  doi       = {10.1145/3696410.3714595}
}
```
