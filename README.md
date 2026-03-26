# Thematic-LM

Implementation of **Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis** (Qiao et al., WWW '25).

> Qiao, T., Walker, C., Cunningham, C., & Koh, Y. S. (2025). Thematic-LM: A LLM-based Multi-agent System for Large-scale Thematic Analysis. *Proceedings of the ACM Web Conference 2025 (WWW '25)*. https://doi.org/10.1145/3696410.3714595

---

## Background

Thematic Analysis (TA) is a qualitative method for identifying patterns of meaning within unstructured text. The traditional process — familiarization, coding, theme development, and interpretation — is labour-intensive and does not scale to large datasets.

The paper identifies three core limitations of prior LLM-assisted TA approaches:

1. **Human dependency** — current methods require a human coder to oversee and provide feedback to the LLM, which is infeasible at scale.
2. **Mirroring bias** — iterative human feedback causes the LLM to converge on the human coder's viewpoint, narrowing the analysis.
3. **Sequential coding** — codes are generated per item without recalling previously coded data, preventing holistic analysis.

Thematic-LM addresses all three by distributing specialised roles across a team of LLM agents and maintaining an adaptive codebook that persists across the dataset.

---

## Architecture

The pipeline runs in two sequential stages, each composed of multiple specialised agents.

```
                        ┌─────────────────────────────────────────────────┐
                        │              STAGE 1: CODING                    │
                        │                                                 │
  Text Data (batched)   │   Coder 1 ─┐                                   │
  ──────────────────►   │   Coder 2 ─┼─► Code Aggregator ─► Reviewer ─► │─► Codebook
                        │   Coder N ─┘         (merge)      (retrieve     │
                        │                                     & update)   │
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

### Agent Roles

| Agent | Stage | Role |
|---|---|---|
| **Coder** | 1 | Reads each text item and assigns 1–3 short codes, each with a verbatim quote as evidence. Multiple coders run independently, optionally with different identity perspectives. |
| **Code Aggregator** | 1 | Receives all coders' outputs for a batch, merges codes with similar meanings, and organises them into a structured JSON format. |
| **Reviewer** | 1 | Maintains the adaptive codebook. For each new aggregated code, retrieves the top-*k* most similar existing codes (via cosine similarity on sentence embeddings), then decides whether to keep, update, or merge entries. |
| **Theme Coder** | 2 | Receives the complete codebook and identifies overarching themes that capture deeper patterns, each with a description and supporting quotes. |
| **Theme Aggregator** | 2 | Merges themes from all theme coders into a final coherent set, preserving distinct perspectives. |
| **Evaluator** | Eval | Acts as LLM-as-a-judge: checks whether each quote associated with a theme is genuinely consistent with it (used for credibility measurement). |

### Adaptive Codebook

The codebook is the mechanism that gives the system holistic awareness across the full dataset. Each entry stores:

- The **code label** (text)
- A list of **supporting quotes** and their source IDs
- A **sentence embedding** (via SentenceTransformer) used for similarity retrieval

After each batch, the reviewer queries the codebook for the top-*k* most similar codes, compares them to newly generated codes, and updates the codebook accordingly — simulating the iterative refinement that human teams perform in meetings.

---

## Coder Identities

TA is inherently subjective: different researchers bring different assumptions, and the same data can yield legitimately different themes depending on who analyses it (Braun & Clarke, 2006).

Thematic-LM simulates this by assigning each coder agent a distinct identity, instructing it to interpret data through that lens. The paper tests this on the Reddit climate change dataset (a polarising topic) using five perspectives:

| Identity | Perspective |
|---|---|
| **Human-Driven** | Anthropogenic causes are the primary driver; focus on industrial emissions and policy. |
| **Natural Cycles** | Climate change is partly natural; questions the extent of human influence. |
| **Progressive** | Environmental justice framing; systemic change, grassroots activism, protecting vulnerable communities. |
| **Conservative** | Market-driven, gradual solutions; economic stability over large regulatory interventions. |
| **Indigenous** | Traditional ecological knowledge; sacred stewardship, interconnectedness of all living beings. |

**Key finding from the paper:** agents with *different* identities produce codes with lower ROUGE overlap (~28–63%) compared to agents with *no* assigned identity (~58–77% overlap) or the *same* identity (~81–92% overlap). The identity-diverse system also surfaces themes that the no-identity baseline misses entirely — for example, *"Scepticism of Climate Science"* and *"Economic Impact of Climate Policies"*.

---

## Evaluation Framework

The paper argues that standard inter-rater reliability (e.g. Cohen's kappa) is not appropriate for TA because TA can legitimately reflect multiple valid perspectives. Instead, it adapts four **trustworthiness principles** from qualitative research (Lincoln & Guba, 1985):

### 1. Credibility & Confirmability

> Are the themes accurate representations of the data? Are they driven by the data rather than by model bias?

Measured by an LLM evaluator agent that checks, for each theme, what fraction of its associated quotes are genuinely consistent with the theme's description. Inconsistencies can signal hallucination or internal LLM bias.

### 2. Dependability

> Would repeating the process reveal similar findings?

Measured by running the full pipeline *N* times and computing average pairwise ROUGE scores across the resulting theme sets. Higher ROUGE = more stable themes across runs.

The paper uses symmetric ROUGE averaging (Equations 1–3):

```
ROUGE-1(A,B) = ½ · (ROUGE-1_{A→B} + ROUGE-1_{B→A})
ROUGE-2(A,B) = ½ · (ROUGE-2_{A→B} + ROUGE-2_{B→A})
ROUGE        = ½ · (ROUGE-1 + ROUGE-2)
```

### 3. Transferability

> Can the identified themes generalise to unseen data from the same domain?

Measured by splitting the dataset 50/50, running the pipeline independently on each split, and computing ROUGE overlap between the two resulting theme sets.

### Results from the paper

On the Dreaddit dataset (GPT-4o):

| Method | Credibility & Confirmability | Dependability | Transferability |
|---|---|---|---|
| Single LLM agent | 0.63 | 0.45 | 0.41 |
| Single + Codebook | 0.75 | 0.61 | 0.67 |
| System (1 Coder) | 0.92 | **0.81** | 0.86 |
| System (2 Coders) | **0.94** | 0.78 | **0.87** |

On the Reddit climate change dataset (GPT-4o):

| Method | Credibility & Confirmability | Dependability | Transferability |
|---|---|---|---|
| Single LLM agent | 0.66 | 0.56 | 0.73 |
| Single + Codebook | 0.74 | 0.69 | 0.78 |
| System (1 Coder) | 0.96 | 0.84 | **0.90** |
| System (2 Coders) | **0.98** | **0.86** | 0.89 |

---

## Code Structure

```
thematic_lm/
├── __init__.py          # Public API exports
├── llm.py               # Provider-agnostic LLM client (OpenAI / Anthropic / compatible)
├── codebook.py          # Adaptive codebook: storage, embedding, similarity retrieval
├── agents.py            # All six agent functions with prompts from Appendix B
├── identities.py        # Coder identity profiles (climate, stress, custom)
├── evaluation.py        # Credibility, dependability, transferability metrics
└── pipeline.py          # ThematicLMPipeline: orchestrates both stages
```

### `llm.py` — Provider Abstraction

A single `LLMClient` class wraps OpenAI and Anthropic SDKs behind a unified interface. The `provider="openai"` path also accepts any OpenAI-compatible endpoint via `base_url`, covering Groq, Together AI, Mistral, local Ollama instances, and others.

### `codebook.py` — Adaptive Codebook

Backed by an in-memory dictionary and serialisable to JSON. Codes are embedded with a SentenceTransformer model (default: `all-MiniLM-L6-v2`) and retrieved by cosine similarity. Supports add, update, merge, and trim operations that the reviewer agent calls after each batch.

### `agents.py` — Agent Functions

Stateless functions that each call the LLM with a specific system prompt (adapted from Appendix B of the paper) and return structured output. JSON mode is requested where supported by the provider.

### `pipeline.py` — Orchestration

`ThematicLMPipeline` accepts configuration (number of coders, identities, batch size, top-k parameters) and drives the two-stage process. It supports:

- Resuming from a saved codebook (`codebook_path`)
- Running with or without coder identities
- A `run_split()` helper for transferability evaluation

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `openai` | OpenAI API + compatible providers |
| `anthropic` | Anthropic API |
| `sentence-transformers` | Code embeddings for similarity retrieval |
| `scikit-learn` | Cosine similarity computation |
| `rouge-score` | ROUGE metrics for dependability/transferability |
| `tqdm` | Progress bars |

---

## Usage

```python
from thematic_lm import LLMClient, ThematicLMPipeline, trustworthiness_report

# 1. Configure the LLM provider
client = LLMClient(provider="openai", model="gpt-4o")
# client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")

# 2. Prepare data
data = [
    {"id": "1", "text": "I can't sleep from anxiety about losing my job."},
    {"id": "2", "text": "Rent went up again. Don't know how I'll manage."},
    # ...
]

# 3. Run the pipeline
pipeline = ThematicLMPipeline(client=client, n_coders=2, n_theme_coders=2)
themes = pipeline.run(data)

# 4. Evaluate trustworthiness
data_lookup = {item["id"]: item["text"] for item in data}
report = trustworthiness_report(client=client, themes=themes, data_lookup=data_lookup)
```

### Using identity-diverse coders

```python
from thematic_lm import CLIMATE_IDENTITIES

pipeline = ThematicLMPipeline(
    client=client,
    n_coders=5,
    n_theme_coders=5,
    coder_identities=list(CLIMATE_IDENTITIES.values()),
    theme_coder_identities=list(CLIMATE_IDENTITIES.values()),
)
themes = pipeline.run(climate_data)
```

See [example.py](example.py) for a full runnable script.

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
