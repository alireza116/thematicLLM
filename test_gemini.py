"""
Quick test of the Gemini provider end-to-end through the Thematic-LM pipeline.

Runs the full two-stage analysis on a small sample dataset and prints the
resulting themes plus a credibility score.

Prerequisites:
    pip install -r requirements.txt
    # Set GEMINI_API_KEY in .env  (get one at https://aistudio.google.com/app/apikey)
"""

import json
from thematic_lm import LLMClient, ThematicLMPipeline, trustworthiness_report

# ---------------------------------------------------------------------------
# 1. Client
# ---------------------------------------------------------------------------

client = LLMClient(
    provider="openai",
    model="gpt-4o",   # fast + cheap; swap for "gemini-1.5-pro" for quality
    # api_key="AIza...",        # or set GEMINI_API_KEY in .env
)

# ---------------------------------------------------------------------------
# 2. Smoke test — plain completion
# ---------------------------------------------------------------------------

print("--- Smoke test: plain completion ---")
response = client.complete([{"role": "user", "content": "Say hello in one sentence."}])
print(response)

print("\n--- Smoke test: JSON completion ---")
result = client.complete_json(
    [{"role": "user", "content": "Return a JSON object with keys 'status' and 'model'."}]
)
print(json.dumps(result, indent=2))

# ---------------------------------------------------------------------------
# 3. Full pipeline test on a small stress dataset
# ---------------------------------------------------------------------------

data = [
    {"id": "1", "text": "I can't sleep because I keep worrying about losing my job. The anxiety is overwhelming."},
    {"id": "2", "text": "My landlord raised the rent again. I don't know how I'll make ends meet this month."},
    {"id": "3", "text": "Had a panic attack at work today. My manager didn't even notice. Feeling invisible."},
    {"id": "4", "text": "Finally got approved for food assistance. Still embarrassed to use it but I have no choice."},
    {"id": "5", "text": "My parents keep pressuring me about grades. I feel suffocated and burned out."},
    {"id": "6", "text": "Been arguing with my partner every day this week. We're both stressed and taking it out on each other."},
    {"id": "7", "text": "Posted about my struggles online and people actually responded with support. Didn't expect that."},
    {"id": "8", "text": "My online persona is so different from who I am in real life. It's exhausting to keep up."},
]

print("\n--- Running Thematic-LM pipeline with Gemini ---")

pipeline = ThematicLMPipeline(
    client=client,
    n_coders=2,
    n_theme_coders=2,
    batch_size=4,
)

themes = pipeline.run(data)

print("\n=== Themes ===")
for t in themes:
    print(f"\n  Theme: {t['theme']}")
    print(f"  {t['description']}")


# 4. Credibility evaluation


print("\n--- Evaluating credibility ---")
data_lookup = {item["id"]: item["text"] for item in data}

report = trustworthiness_report(
    client=client,
    themes=themes,
    data_lookup=data_lookup,
)

print("\n=== Trustworthiness Report ===")
print(json.dumps(report, indent=2))
