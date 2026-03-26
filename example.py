"""
Example: run Thematic-LM on a small dataset.

Works with OpenAI, Anthropic, or any OpenAI-compatible provider.
Swap the LLMClient constructor to switch providers.
"""

import json
from thematic_lm import (
    LLMClient,
    ThematicLMPipeline,
    CLIMATE_IDENTITIES,
    trustworthiness_report,
)

# ---------------------------------------------------------------------------
# 1. Configure the LLM client
# ---------------------------------------------------------------------------

# --- OpenAI ---
client = LLMClient(provider="openai", model="gpt-4o")
# Reads OPENAI_API_KEY from environment by default.

# --- Anthropic ---
# client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
# Reads ANTHROPIC_API_KEY from environment by default.

# --- Any OpenAI-compatible endpoint (e.g. Groq) ---
# client = LLMClient(
#     provider="openai",
#     model="llama3-8b-8192",
#     api_key="gsk_...",
#     base_url="https://api.groq.com/openai/v1",
# )

# ---------------------------------------------------------------------------
# 2. Prepare your data
# ---------------------------------------------------------------------------

data = [
    {"id": "1", "text": "I can't sleep because I keep worrying about losing my job. The anxiety is overwhelming."},
    {"id": "2", "text": "My landlord raised the rent again. I don't know how I'll make ends meet this month."},
    {"id": "3", "text": "Had a panic attack at work today. My manager didn't even notice. Feeling invisible."},
    {"id": "4", "text": "Finally got approved for food assistance. Still embarrassed to use it but I have no choice."},
    {"id": "5", "text": "My parents keep pressuring me about grades but I'm already doing my best. I feel suffocated."},
    {"id": "6", "text": "Been arguing with my partner every day this week. We're both stressed and taking it out on each other."},
    {"id": "7", "text": "Posted about my struggles online and people actually responded with support. Didn't expect that."},
    {"id": "8", "text": "My online persona is so different from who I am in real life. It's exhausting to keep up."},
    {"id": "9", "text": "Therapy waitlist is 6 months. I don't know how I'm supposed to cope until then."},
    {"id": "10", "text": "Got laid off today. Sent out 50 applications last month and only heard back from 2."},
]

# ---------------------------------------------------------------------------
# 3. Basic run (no identities — paper's baseline)
# ---------------------------------------------------------------------------

pipeline = ThematicLMPipeline(
    client=client,
    n_coders=2,
    n_theme_coders=2,
    batch_size=5,
    codebook_path="codebook.json",   # saves progress; remove to disable
)

themes = pipeline.run(data)

print("\n=== Final Themes ===")
for t in themes:
    print(f"\nTheme: {t['theme']}")
    print(f"  {t['description']}")

# ---------------------------------------------------------------------------
# 4. Run with diverse coder identities (Section 3.2 / 4.3 of the paper)
# ---------------------------------------------------------------------------

# identity_pipeline = ThematicLMPipeline(
#     client=client,
#     n_coders=5,
#     n_theme_coders=5,
#     coder_identities=list(CLIMATE_IDENTITIES.values()),
#     theme_coder_identities=list(CLIMATE_IDENTITIES.values()),
#     batch_size=5,
# )
# themes_with_identities = identity_pipeline.run(climate_data)

# ---------------------------------------------------------------------------
# 5. Trustworthiness evaluation
# ---------------------------------------------------------------------------

# Build a lookup of id → original text for credibility evaluation
data_lookup = {item["id"]: item["text"] for item in data}

report = trustworthiness_report(
    client=client,
    themes=themes,
    data_lookup=data_lookup,
    # theme_runs=[themes, themes_run2, themes_run3],  # for dependability
    # train_themes=train_themes, val_themes=val_themes,  # for transferability
)

print("\n=== Trustworthiness Report ===")
print(json.dumps(report, indent=2))
