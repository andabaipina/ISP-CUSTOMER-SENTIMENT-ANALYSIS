"""
ISP Complaint Topic Extraction
================================
Extracts the most common complaint themes from negative sentiment records
using TF-IDF keyword extraction and rule-based topic categorisation.

Run in Colab after the main pipeline has populated isp_sentiment.db

pip install scikit-learn wordcloud matplotlib
"""

import sqlite3
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DB_PATH = "isp_sentiment.db"

# ── Topic categories — keywords that map to complaint themes ──────────────────
TOPIC_KEYWORDS = {
    "Network outage":       ["down", "outage", "offline", "disconnected", "unavailable", "dead"],
    "Slow speed":           ["slow", "speed", "fast", "bandwidth", "throttle", "throttling", "lag", "latency"],
    "Data depletion":       ["data", "depleted", "finished", "exhausted", "used", "consumption", "bundle"],
    "Billing issues":       ["billing", "charge", "overcharge", "refund", "payment", "debit", "fraud", "money"],
    "Customer service":     ["service", "support", "agent", "hold", "response", "helpline", "complaint", "rude"],
    "Poor coverage":        ["coverage", "signal", "zone", "area", "location", "rural", "weak", "no network"],
    "Value for money":      ["expensive", "price", "cost", "cheap", "worth", "value", "overpriced"],
    "App / portal issues":  ["app", "portal", "website", "login", "account", "dashboard", "self-service"],
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_sentiment_data(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT isp, source, text_clean, compound, label, published_at
        FROM sentiment_scores
        ORDER BY published_at DESC
    """, conn)
    conn.close()
    print(f"Loaded {len(df)} records — {df[df.label=='negative'].shape[0]} negative")
    return df


# ── Rule-based topic tagger ───────────────────────────────────────────────────
def tag_topics(text: str) -> list[str]:
    """Return list of complaint topics found in a piece of text."""
    text_lower = text.lower()
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(topic)
    return found if found else ["Other"]


def extract_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every record with one or more complaint topics."""
    df = df.copy()
    df["topics"] = df["text_clean"].apply(tag_topics)
    # Explode so each row = one topic (for easy grouping)
    df_topics = df.explode("topics")
    return df_topics


# ── TF-IDF top keywords per ISP ───────────────────────────────────────────────
def top_keywords_per_isp(df: pd.DataFrame, n_keywords: int = 10) -> dict:
    """
    For each ISP, extract the top N keywords from negative reviews
    using TF-IDF (weights words that are important to that ISP specifically).
    """
    negative_df = df[df["label"] == "negative"].copy()
    results = {}

    for isp in negative_df["isp"].unique():
        isp_texts = negative_df[negative_df["isp"] == isp]["text_clean"].tolist()
        if len(isp_texts) < 3:
            continue
        try:
            vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),   # unigrams + bigrams
                min_df=2,
                stop_words="english",
            )
            tfidf_matrix = vectorizer.fit_transform(isp_texts)
            scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            vocab = vectorizer.get_feature_names_out()
            top_indices = scores.argsort()[::-1][:n_keywords]
            results[isp] = [(vocab[i], round(float(scores[i]), 4)) for i in top_indices]
        except Exception as e:
            print(f"TF-IDF failed for {isp}: {e}")

    return results


# ── Analysis functions ─────────────────────────────────────────────────────────
def complaint_topic_summary(df_topics: pd.DataFrame) -> pd.DataFrame:
    """Count complaints per topic per ISP."""
    negative = df_topics[df_topics["label"] == "negative"]
    summary = (
        negative.groupby(["isp", "topics"])
        .size()
        .reset_index(name="count")
        .sort_values(["isp", "count"], ascending=[True, False])
    )
    return summary


def top_complaints_per_isp(df_topics: pd.DataFrame) -> pd.DataFrame:
    """Return the #1 complaint topic per ISP."""
    summary = complaint_topic_summary(df_topics)
    top = summary.groupby("isp").first().reset_index()
    top.columns = ["isp", "top_complaint", "count"]
    return top


def monthly_topic_trend(df_topics: pd.DataFrame) -> pd.DataFrame:
    """Show how complaint topics change month by month."""
    df = df_topics.copy()
    df["month"] = pd.to_datetime(df["published_at"], errors="coerce").dt.to_period("M").astype(str)
    trend = (
        df[df["label"] == "negative"]
        .groupby(["month", "topics"])
        .size()
        .reset_index(name="count")
        .sort_values(["month", "count"], ascending=[True, False])
    )
    return trend


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_complaint_heatmap(df_topics: pd.DataFrame) -> None:
    """Heatmap: ISP (rows) × complaint topic (cols) → count."""
    summary = complaint_topic_summary(df_topics)
    pivot = summary.pivot(index="isp", columns="topics", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    plt.colorbar(im, ax=ax, label="Complaint count")
    ax.set_title("ISP Complaint Topics Heatmap", fontsize=14, fontweight="bold", pad=12)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = int(pivot.values[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=9,
                        color="white" if val > pivot.values.max() * 0.6 else "black")
    plt.tight_layout()
    plt.savefig("complaint_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: complaint_heatmap.png")


def plot_top_keywords(keyword_results: dict) -> None:
    """Horizontal bar chart of top TF-IDF keywords per ISP."""
    n_isps = len(keyword_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ["#E24B4A", "#D85A30", "#BA7517", "#1D9E75", "#185FA5", "#534AB7"]

    for idx, (isp, keywords) in enumerate(keyword_results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        words = [k[0] for k in keywords[:8]]
        scores = [k[1] for k in keywords[:8]]
        bars = ax.barh(words[::-1], scores[::-1], color=colors[idx % len(colors)], alpha=0.8)
        ax.set_title(isp, fontweight="bold", fontsize=12)
        ax.set_xlabel("TF-IDF score", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(keyword_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Top Complaint Keywords by ISP (TF-IDF)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("top_keywords_per_isp.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: top_keywords_per_isp.png")


def plot_topic_bar(df_topics: pd.DataFrame) -> None:
    """Stacked bar showing complaint topic mix per ISP."""
    summary = complaint_topic_summary(df_topics)
    pivot = summary.pivot(index="isp", columns="topics", values="count").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    topic_colors = {
        "Network outage":      "#E24B4A",
        "Slow speed":          "#D85A30",
        "Data depletion":      "#BA7517",
        "Billing issues":      "#EF9F27",
        "Customer service":    "#534AB7",
        "Poor coverage":       "#185FA5",
        "Value for money":     "#1D9E75",
        "App / portal issues": "#888780",
        "Other":               "#B4B2A9",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(pivot_pct))
    for topic in pivot_pct.columns:
        color = topic_colors.get(topic, "#888780")
        bars = ax.bar(pivot_pct.index, pivot_pct[topic], bottom=bottom,
                      label=topic, color=color, alpha=0.85)
        bottom += pivot_pct[topic].values

    ax.set_ylabel("% of complaints", fontsize=11)
    ax.set_title("Complaint Topic Mix by ISP", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("complaint_topic_mix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: complaint_topic_mix.png")


# ── Export for Tableau ─────────────────────────────────────────────────────────
def export_topic_csv(df_topics: pd.DataFrame, keyword_results: dict) -> None:
    """Export topic data to CSV for use in Tableau."""

    # Main topic data
    df_export = df_topics[["isp", "source", "label", "compound", "topics", "published_at"]].copy()
    df_export["month"] = pd.to_datetime(df_export["published_at"], errors="coerce").dt.to_period("M").astype(str)
    df_export.to_csv("topic_data.csv", index=False)
    print(f"Exported {len(df_export)} rows → topic_data.csv")

    # Keyword summary
    rows = []
    for isp, keywords in keyword_results.items():
        for rank, (word, score) in enumerate(keywords, 1):
            rows.append({"isp": isp, "keyword": word, "tfidf_score": score, "rank": rank})
    kw_df = pd.DataFrame(rows)
    kw_df.to_csv("keywords_per_isp.csv", index=False)
    print(f"Exported {len(kw_df)} rows → keywords_per_isp.csv")


# ── Main ───────────────────────────────────────────────────────────────────────
def run_topic_analysis():
    df = load_sentiment_data()
    df_topics = extract_topics(df)

    print("\n── Top complaint per ISP ──")
    print(top_complaints_per_isp(df_topics).to_string(index=False))

    print("\n── All complaint topics ──")
    print(complaint_topic_summary(df_topics).to_string(index=False))

    print("\n── TF-IDF keywords (negative reviews) ──")
    keyword_results = top_keywords_per_isp(df)
    for isp, kws in keyword_results.items():
        print(f"\n{isp}: {', '.join([k[0] for k in kws[:6]])}")

    # Charts
    plot_complaint_heatmap(df_topics)
    plot_top_keywords(keyword_results)
    plot_topic_bar(df_topics)

    # Export for Tableau
    export_topic_csv(df_topics, keyword_results)

    # Download from Colab
    try:
        from google.colab import files
        for f in ["topic_data.csv", "keywords_per_isp.csv",
                  "complaint_heatmap.png", "top_keywords_per_isp.png",
                  "complaint_topic_mix.png"]:
            files.download(f)
    except ImportError:
        print("Not in Colab — files saved locally.")


if __name__ == "__main__":
    run_topic_analysis()
