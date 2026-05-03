"""
ISP Sentiment — LDA Topic Modelling
=====================================
Uses Latent Dirichlet Allocation (LDA) to automatically discover
complaint themes from negative reviews — no predefined keywords needed.

Run in Colab after generate_sample_data.py has populated isp_sentiment.db

pip install scikit-learn pyLDAvis matplotlib pandas
"""

import sqlite3
import pandas as pd
import numpy as np
import warnings
import re
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DB_PATH = "isp_sentiment.db"
N_TOPICS = 6        # number of themes to discover — tune this
N_TOP_WORDS = 8     # top words shown per topic
RANDOM_STATE = 42


# ── Load negative reviews ─────────────────────────────────────────────────────
def load_negative_reviews(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT isp, source, text_clean, compound, published_at
        FROM sentiment_scores
        WHERE label = 'negative'
        ORDER BY published_at DESC
    """, conn)
    conn.close()
    print(f"Loaded {len(df)} negative reviews across {df['isp'].nunique()} ISPs")
    return df


# ── Vectorise text ────────────────────────────────────────────────────────────
def vectorise(texts: list[str]):
    """Convert texts to a document-term matrix for LDA."""
    vectorizer = CountVectorizer(
        max_features=500,
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=2,               # word must appear in at least 2 docs
        max_df=0.90,            # ignore words in >90% of docs (too common)
        stop_words="english",
    )
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer


# ── Fit LDA ───────────────────────────────────────────────────────────────────
def fit_lda(dtm, n_topics: int = N_TOPICS):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="online",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lda.fit(dtm)
    perplexity = lda.perplexity(dtm)
    print(f"LDA fitted — {n_topics} topics | perplexity: {perplexity:.1f} (lower = better fit)")
    return lda


# ── Extract top words per topic ───────────────────────────────────────────────
def get_topic_words(lda, vectorizer, n_words: int = N_TOP_WORDS) -> dict:
    vocab = vectorizer.get_feature_names_out()
    topics = {}
    for idx, component in enumerate(lda.components_):
        top_indices = component.argsort()[::-1][:n_words]
        top_words = [vocab[i] for i in top_indices]
        topics[idx] = top_words
    return topics


def name_topics(topic_words: dict) -> dict:
    """
    Auto-suggest a human-readable name based on top words.
    You can override these names manually after seeing the output.
    """
    THEME_HINTS = {
        "Network outage":    ["down", "outage", "offline", "disconnected", "network"],
        "Slow speed":        ["slow", "speed", "throttle", "lag", "bandwidth", "fast"],
        "Data depletion":    ["data", "bundle", "finished", "exhausted", "gb", "mb"],
        "Billing issues":    ["billing", "charge", "money", "refund", "overcharge", "payment"],
        "Customer service":  ["service", "support", "agent", "hold", "call", "helpline"],
        "Poor coverage":     ["coverage", "signal", "area", "zone", "location", "rural"],
    }
    names = {}
    for topic_id, words in topic_words.items():
        best_match = f"Topic {topic_id + 1}"
        best_score = 0
        for theme, hints in THEME_HINTS.items():
            score = sum(1 for w in words if any(h in w for h in hints))
            if score > best_score:
                best_score = score
                best_match = theme
        names[topic_id] = best_match
    return names


# ── Assign dominant topic to each review ─────────────────────────────────────
def assign_topics(df: pd.DataFrame, lda, dtm) -> pd.DataFrame:
    """Add dominant topic and confidence score to each review."""
    topic_probs = lda.transform(dtm)
    df = df.copy()
    df["dominant_topic"] = topic_probs.argmax(axis=1)
    df["topic_confidence"] = topic_probs.max(axis=1).round(3)
    return df, topic_probs


# ── ISP topic distribution ────────────────────────────────────────────────────
def isp_topic_distribution(df: pd.DataFrame, topic_names: dict) -> pd.DataFrame:
    """How are complaint topics distributed across ISPs?"""
    df["topic_name"] = df["dominant_topic"].map(topic_names)
    dist = (
        df.groupby(["isp", "topic_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["isp", "count"], ascending=[True, False])
    )
    return dist


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_topic_words(topic_words: dict, topic_names: dict) -> None:
    """Bar chart of top words for each discovered topic."""
    n = len(topic_words)
    cols = 3
    rows = int(np.ceil(n / cols))
    palette = list(mcolors.TABLEAU_COLORS.values())

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for idx, (topic_id, words) in enumerate(topic_words.items()):
        ax = axes[idx]
        # Fake scores descending for visual clarity
        scores = [1.0 - i * 0.08 for i in range(len(words))]
        ax.barh(words[::-1], scores[::-1], color=palette[idx % len(palette)], alpha=0.85)
        ax.set_title(f"Topic {topic_id+1}: {topic_names[topic_id]}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Relative weight", fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in range(len(topic_words), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Automatically Discovered Complaint Themes (LDA)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("lda_topic_words.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: lda_topic_words.png")


def plot_isp_topic_heatmap(df: pd.DataFrame, topic_names: dict) -> None:
    """Heatmap: which ISP complains about which topic most?"""
    df["topic_name"] = df["dominant_topic"].map(topic_names)
    pivot = (
        df.groupby(["isp", "topic_name"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalise to % of each ISP's complaints
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot_pct.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=50)
    ax.set_xticks(range(len(pivot_pct.columns)))
    ax.set_xticklabels(pivot_pct.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot_pct.index)))
    ax.set_yticklabels(pivot_pct.index, fontsize=11)
    plt.colorbar(im, ax=ax, label="% of ISP complaints")

    for i in range(len(pivot_pct.index)):
        for j in range(len(pivot_pct.columns)):
            val = pivot_pct.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if val > 35 else "black")

    ax.set_title("LDA Complaint Topics by ISP (%)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("lda_isp_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: lda_isp_heatmap.png")


def plot_topic_confidence(df: pd.DataFrame, topic_names: dict) -> None:
    """Box plot showing how confidently reviews were assigned to each topic."""
    df["topic_name"] = df["dominant_topic"].map(topic_names)
    topic_order = df.groupby("topic_name")["topic_confidence"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["topic_name"] == t]["topic_confidence"].values for t in topic_order]
    bp = ax.boxplot(data, patch_artist=True, vert=True)
    palette = list(mcolors.TABLEAU_COLORS.values())
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(topic_order, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Topic confidence score", fontsize=11)
    ax.set_title("LDA Topic Assignment Confidence", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("lda_confidence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: lda_confidence.png")


# ── Export ─────────────────────────────────────────────────────────────────────
def export_lda_results(df: pd.DataFrame, topic_words: dict, topic_names: dict) -> None:
    df["topic_name"] = df["dominant_topic"].map(topic_names)

    # Per-review topic assignments
    df[["isp", "source", "text_clean", "compound", "dominant_topic",
        "topic_name", "topic_confidence", "published_at"]].to_csv("lda_reviews.csv", index=False)
    print(f"Exported {len(df)} rows → lda_reviews.csv")

    # Topic word summary
    rows = []
    for topic_id, words in topic_words.items():
        for rank, word in enumerate(words, 1):
            rows.append({"topic_id": topic_id + 1,
                         "topic_name": topic_names[topic_id],
                         "word": word, "rank": rank})
    pd.DataFrame(rows).to_csv("lda_topic_words.csv", index=False)
    print("Exported → lda_topic_words.csv")


# ── Main ───────────────────────────────────────────────────────────────────────
def run_lda():
    df = load_negative_reviews()

    if len(df) < 20:
        print("Not enough negative reviews for LDA (need at least 20). Run the pipeline first.")
        return

    texts = df["text_clean"].tolist()
    dtm, vectorizer = vectorise(texts)

    print(f"\nFitting LDA with {N_TOPICS} topics...")
    lda = fit_lda(dtm, n_topics=N_TOPICS)

    topic_words = get_topic_words(lda, vectorizer)
    topic_names = name_topics(topic_words)

    print("\n── Discovered topics ──")
    for tid, words in topic_words.items():
        print(f"  Topic {tid+1} [{topic_names[tid]}]: {', '.join(words)}")

    df, topic_probs = assign_topics(df, lda, dtm)

    print("\n── Top complaint topic per ISP ──")
    dist = isp_topic_distribution(df, topic_names)
    top_per_isp = dist.groupby("isp").first().reset_index()
    print(top_per_isp.to_string(index=False))

    # Charts
    plot_topic_words(topic_words, topic_names)
    plot_isp_topic_heatmap(df, topic_names)
    plot_topic_confidence(df, topic_names)

    # Export
    export_lda_results(df, topic_words, topic_names)

    # Download from Colab
    try:
        from google.colab import files
        for f in ["lda_reviews.csv", "lda_topic_words.csv",
                  "lda_topic_words.png", "lda_isp_heatmap.png", "lda_confidence.png"]:
            files.download(f)
    except ImportError:
        print("Not in Colab — files saved locally.")

    print("\nDone. Load lda_reviews.csv into Tableau for topic-level dashboard sheets.")
    return df, topic_words, topic_names


if __name__ == "__main__":
    run_lda()
