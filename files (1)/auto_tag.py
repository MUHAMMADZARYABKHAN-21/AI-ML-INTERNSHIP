"""
Task 5: Auto-Tagging Support Tickets Using LLM
DevelopersHub Corporation – AI/ML Engineering Internship

Compares zero-shot, few-shot, and chain-of-thought prompting with Claude
to tag support tickets with top-3 most probable categories.
"""

import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import anthropic

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
MODEL      = "claude-haiku-4-5-20251001"   # fast + cheap for classification
TAG_SCHEMA = [
    "Billing & Payments",
    "Account & Login",
    "Technical Bug",
    "Feature Request",
    "Shipping & Delivery",
    "Refund & Returns",
    "Product Quality",
    "Security & Privacy",
    "Performance",
    "General Inquiry",
]

# ─────────────────────────────────────────
#  1. Synthetic support ticket dataset
# ─────────────────────────────────────────
TICKET_DATA = [
    # (ticket_text, ground_truth_tag)
    ("My invoice shows double charge for last month subscription. Please fix this.",
     "Billing & Payments"),
    ("I forgot my password and the reset email never arrives.", "Account & Login"),
    ("App crashes every time I try to upload a photo on iOS 17.", "Technical Bug"),
    ("Would love a dark mode option in the mobile app!", "Feature Request"),
    ("Package marked as delivered but nothing at my door.", "Shipping & Delivery"),
    ("I returned the item 2 weeks ago and still no refund.", "Refund & Returns"),
    ("The headphones stopped working after just 3 days of use.", "Product Quality"),
    ("I think my account was hacked. Seeing unknown login from Russia.", "Security & Privacy"),
    ("Dashboard takes 30+ seconds to load. This is unusable.", "Performance"),
    ("What are your business hours for phone support?", "General Inquiry"),
    ("Charged twice in the same billing cycle.", "Billing & Payments"),
    ("Cannot log in — says account doesn't exist but I just signed up.", "Account & Login"),
    ("Export to CSV button does nothing when clicked.", "Technical Bug"),
    ("Please add an API so we can integrate with Zapier.", "Feature Request"),
    ("Order shipped to wrong address even though billing was correct.", "Shipping & Delivery"),
    ("Store said 5-7 days but it's been 3 weeks since purchase.", "Shipping & Delivery"),
    ("The jacket I received looks nothing like the photo online.", "Product Quality"),
    ("Where can I find information about your data retention policy?", "Security & Privacy"),
    ("Video calls lag badly with more than 3 participants.", "Performance"),
    ("Do you offer student discounts?", "General Inquiry"),
    ("My monthly fee went up without any notification.", "Billing & Payments"),
    ("Two-factor auth code not arriving via SMS.", "Account & Login"),
    ("Search results always return 0 items regardless of query.", "Technical Bug"),
    ("It would be great if you could add calendar sync.", "Feature Request"),
    ("Partial order arrived — missing 2 of 5 items ordered.", "Shipping & Delivery"),
    ("Want to return a defective product but return portal is broken.", "Refund & Returns"),
    ("Screen protector peeled off after one day.", "Product Quality"),
    ("Please delete all my personal data per GDPR request.", "Security & Privacy"),
    ("Why is the mobile app so slow compared to the website?", "Performance"),
    ("Can I upgrade my plan mid-month and get prorated billing?", "General Inquiry"),
]


def create_dataset() -> pd.DataFrame:
    df = pd.DataFrame(TICKET_DATA, columns=["ticket", "true_tag"])
    df["ticket_id"] = [f"TKT-{i+1:03d}" for i in range(len(df))]
    print(f"Dataset: {len(df)} tickets across {df['true_tag'].nunique()} categories")
    print(df["true_tag"].value_counts().to_string())
    return df


# ─────────────────────────────────────────
#  2. Prompting strategies
# ─────────────────────────────────────────
def zero_shot_prompt(ticket: str) -> str:
    tags_list = "\n".join(f"  - {t}" for t in TAG_SCHEMA)
    return f"""You are a customer support ticket classifier.

Classify the following support ticket into the TOP 3 most relevant categories from this list:
{tags_list}

Return ONLY a JSON object with this exact format:
{{"tags": [{{"tag": "<category>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}}]}}

Return exactly 3 tags ordered by confidence (highest first). No other text.

Support ticket:
\"\"\"{ticket}\"\"\""""


def few_shot_prompt(ticket: str) -> str:
    tags_list = "\n".join(f"  - {t}" for t in TAG_SCHEMA)
    return f"""You are a customer support ticket classifier.

Classify support tickets into the TOP 3 most relevant categories from this list:
{tags_list}

=== EXAMPLES ===

Ticket: "I was charged twice this month and need a refund for the duplicate charge."
Output: {{"tags": [
  {{"tag": "Billing & Payments", "confidence": 0.92, "reason": "Duplicate charge issue"}},
  {{"tag": "Refund & Returns",   "confidence": 0.78, "reason": "Refund requested"}},
  {{"tag": "Account & Login",    "confidence": 0.15, "reason": "May relate to account"}}
]}}

Ticket: "The app freezes when I try to open settings on my Android phone."
Output: {{"tags": [
  {{"tag": "Technical Bug",   "confidence": 0.95, "reason": "App freeze on specific action"}},
  {{"tag": "Performance",     "confidence": 0.60, "reason": "Freezing suggests performance issue"}},
  {{"tag": "General Inquiry", "confidence": 0.10, "reason": "Low relevance fallback"}}
]}}

Ticket: "My package shows delivered but I haven't received anything. Can I get a replacement or refund?"
Output: {{"tags": [
  {{"tag": "Shipping & Delivery", "confidence": 0.91, "reason": "Package delivery failure"}},
  {{"tag": "Refund & Returns",    "confidence": 0.75, "reason": "Refund/replacement requested"}},
  {{"tag": "Billing & Payments",  "confidence": 0.20, "reason": "Financial resolution implied"}}
]}}

=== NOW CLASSIFY THIS TICKET ===

Ticket: "{ticket}"
Output:"""


def chain_of_thought_prompt(ticket: str) -> str:
    tags_list = "\n".join(f"  - {t}" for t in TAG_SCHEMA)
    return f"""You are an expert customer support analyst.

Available categories:
{tags_list}

Step-by-step, analyse this support ticket and identify the top 3 tags:

1. Read the ticket carefully and identify the core problem
2. Consider which categories are PRIMARY matches
3. Consider which categories are SECONDARY/related matches
4. Assign confidence scores based on relevance

Return your analysis as a JSON object:
{{"reasoning": "<2-3 sentence analysis>",
  "tags": [
    {{"tag": "<category>", "confidence": <0.0-1.0>, "reason": "<specific reason>"}},
    {{"tag": "<category>", "confidence": <0.0-1.0>, "reason": "<specific reason>"}},
    {{"tag": "<category>", "confidence": <0.0-1.0>, "reason": "<specific reason>"}}
  ]
}}

Support Ticket: "{ticket}"

Return ONLY the JSON. No other text."""


# ─────────────────────────────────────────
#  3. API calls
# ─────────────────────────────────────────
def call_claude(client: anthropic.Anthropic, prompt: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()

            # Clean up markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return {"tags": [{"tag": "General Inquiry", "confidence": 0.5, "reason": "Parse error"}] * 3}
        except anthropic.RateLimitError:
            time.sleep(5 * (attempt + 1))


def run_strategy(client, df: pd.DataFrame, strategy: str, prompt_fn) -> pd.DataFrame:
    print(f"\n{'─'*50}")
    print(f"Running strategy: {strategy.upper()}")
    print(f"{'─'*50}")

    results = []
    for _, row in df.iterrows():
        prompt = prompt_fn(row["ticket"])
        result = call_claude(client, prompt)
        tags   = result.get("tags", [])[:3]

        top1_tag = tags[0]["tag"] if tags else "Unknown"
        top1_conf= tags[0]["confidence"] if tags else 0
        all_tags = [t["tag"] for t in tags]
        hit      = row["true_tag"] in all_tags

        results.append({
            "ticket_id":   row["ticket_id"],
            "true_tag":    row["true_tag"],
            "top1_tag":    top1_tag,
            "top1_conf":   top1_conf,
            "top3_tags":   all_tags,
            "hit_top3":    hit,
            "strategy":    strategy,
        })
        time.sleep(0.3)   # gentle rate limiting

    result_df = pd.DataFrame(results)
    acc_top1 = (result_df["top1_tag"] == result_df["true_tag"]).mean()
    acc_top3 = result_df["hit_top3"].mean()

    print(f"Top-1 Accuracy: {acc_top1:.2%}")
    print(f"Top-3 Accuracy: {acc_top3:.2%}")
    return result_df


# ─────────────────────────────────────────
#  4. Visualisations
# ─────────────────────────────────────────
def plot_strategy_comparison(all_results: dict):
    strategies = list(all_results.keys())
    top1_accs  = [(all_results[s]["top1_tag"] == all_results[s]["true_tag"]).mean()
                  for s in strategies]
    top3_accs  = [all_results[s]["hit_top3"].mean() for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Accuracy comparison
    x = np.arange(len(strategies))
    axes[0].bar(x - 0.2, top1_accs, 0.4, label="Top-1", color="#3498db")
    axes[0].bar(x + 0.2, top3_accs, 0.4, label="Top-3", color="#2ecc71")
    axes[0].set_xticks(x); axes[0].set_xticklabels(strategies, rotation=10)
    axes[0].set_ylim(0, 1.1); axes[0].set_title("Accuracy by Strategy")
    axes[0].legend(); axes[0].set_ylabel("Accuracy")
    for i, (t1, t3) in enumerate(zip(top1_accs, top3_accs)):
        axes[0].text(i-0.2, t1+0.02, f"{t1:.0%}", ha="center", fontsize=9)
        axes[0].text(i+0.2, t3+0.02, f"{t3:.0%}", ha="center", fontsize=9)

    # Confidence distribution
    best_strategy = strategies[np.argmax(top1_accs)]
    df_best = all_results[best_strategy]
    df_best.boxplot(column="top1_conf", by="true_tag", ax=axes[1],
                    rot=45, fontsize=7)
    axes[1].set_title(f"Confidence Distribution\n({best_strategy})")
    axes[1].set_xlabel("")

    # Confusion heatmap (top strategy)
    conf = pd.crosstab(df_best["true_tag"], df_best["top1_tag"])
    # Fill missing columns
    for col in TAG_SCHEMA:
        if col not in conf.columns: conf[col] = 0
    conf = conf.reindex(columns=[c for c in TAG_SCHEMA if c in conf.columns])
    sns.heatmap(conf, ax=axes[2], cmap="Blues", fmt="d", annot=True,
                linewidths=0.5, cbar=False, xticklabels=True)
    axes[2].set_title(f"Confusion Matrix\n({best_strategy})")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0, fontsize=7)

    plt.suptitle("Auto-Tagging Support Tickets – LLM Strategy Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("tagging_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: tagging_results.png")


def print_sample_predictions(df: pd.DataFrame, n: int = 5):
    print(f"\n{'─'*70}")
    print("SAMPLE PREDICTIONS")
    print(f"{'─'*70}")
    for _, row in df.head(n).iterrows():
        status = "✅" if row["hit_top3"] else "❌"
        print(f"{status} [{row['ticket_id']}]")
        print(f"   Ticket : {row['true_tag']}")
        print(f"   Top-3  : {' | '.join(row['top3_tags'])}")
        print()


# ─────────────────────────────────────────
#  5. Main
# ─────────────────────────────────────────
def main():
    client = anthropic.Anthropic()
    df     = create_dataset()

    strategies = {
        "Zero-Shot":          zero_shot_prompt,
        "Few-Shot":           few_shot_prompt,
        "Chain-of-Thought":   chain_of_thought_prompt,
    }

    all_results = {}
    for name, prompt_fn in strategies.items():
        result_df        = run_strategy(client, df, name, prompt_fn)
        all_results[name] = result_df

    # Summary table
    print("\n" + "═"*60)
    print("FINAL COMPARISON SUMMARY")
    print("═"*60)
    rows = []
    for name, rdf in all_results.items():
        top1 = (rdf["top1_tag"] == rdf["true_tag"]).mean()
        top3 = rdf["hit_top3"].mean()
        avg_conf = rdf["top1_conf"].mean()
        rows.append({"Strategy": name, "Top-1 Acc": f"{top1:.2%}",
                     "Top-3 Acc": f"{top3:.2%}", "Avg Confidence": f"{avg_conf:.3f}"})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    best = max(all_results.items(), key=lambda x: (x[1]["top1_tag"] == x[1]["true_tag"]).mean())
    print(f"\n🏆 Best strategy: {best[0]}")

    print_sample_predictions(best[1])
    plot_strategy_comparison(all_results)

    # Export results
    pd.concat(all_results.values()).to_csv("tagging_results.csv", index=False)
    print("Results saved to tagging_results.csv")


if __name__ == "__main__":
    main()
