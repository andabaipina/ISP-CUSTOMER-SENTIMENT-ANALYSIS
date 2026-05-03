"""
9mobile Deep Dive Analysis
============================
Extracts every complaint signal specific to 9mobile:
- Top complaint topics
- Worst performing months
- Complaint keywords (TF-IDF)
- Source breakdown (Twitter vs News)
- Sentiment trend over time
- Sample worst reviews

Run in Colab after generate_sample_data.py has been run.
pip install scikit-learn matplotlib wordcloud
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DB_PATH = "isp_sentiment.db"
ISP = "9mobile"

TOPIC_KEYWORDS = {
    "Network outage":    ["down", "outage", "offline", "disconnected", "dead"],
    "Slow speed":        ["slow", "speed", "throttle", "lag", "bandwidth"],
    "Data depletion":    ["data", "bundle", "finished", "exhausted", "depleted"],
    "Billing issues":    ["billing", "charge", "overcharge", "refund", "fraud", "money"],
    "Customer service":  ["service", "support", "agent", "hold", "helpline", "rude"],
    "Poor coverage":     ["coverage", "signal", "zone", "area", "weak", "rural"],
    "Value for money":   ["expensive", "price", "cost", "overpriced", "worth"],
    "App issues":        ["app", "portal", "login", "account", "website"],
}

COLORS = {
    "negative":  "#E24B4A",
    "neutral":   "#B4B2A9",
    "positive":  "#1D9E75",
    "primary":   "#E24B4A",
    "secondary": "#D85A30",
    "accent":    "#BA7517",
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    all_df = pd.read_sql_query("""
        SELECT isp, source, text_clean, compound, label, published_at
        FROM sentiment_scores
        WHERE published_at IS NOT NULL
    """, conn)
    conn.close()
    all_df["published_at"] = pd.to_datetime(all_df["published_at"], errors="coerce")
    all_df["month"] = all_df["published_at"].dt.to_period("M").astype(str)
    mobile_df = all_df[all_df["isp"] == ISP].copy()
    print(f"Loaded {len(mobile_df)} {ISP} records | {len(all_df)} total")
    return mobile_df, all_df


# ── Topic tagging ─────────────────────────────────────────────────────────────
def tag_topics(text):
    text = text.lower()
    found = [t for t, kws in TOPIC_KEYWORDS.items() if any(k in text for k in kws)]
    return found if found else ["Other"]


# ── TF-IDF keywords ───────────────────────────────────────────────────────────
def get_top_keywords(texts, n=12):
    if len(texts) < 3:
        return []
    vec = TfidfVectorizer(max_features=300, ngram_range=(1,2),
                          min_df=2, stop_words="english")
    try:
        matrix = vec.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).flatten()
        vocab  = vec.get_feature_names_out()
        top    = scores.argsort()[::-1][:n]
        return [(vocab[i], round(float(scores[i]), 4)) for i in top]
    except:
        return []


# ── Build the dashboard ───────────────────────────────────────────────────────
def build_dashboard(mobile_df, all_df):
    neg_df  = mobile_df[mobile_df["label"] == "negative"].copy()
    neg_df["topics"] = neg_df["text_clean"].apply(tag_topics)
    neg_exploded = neg_df.explode("topics")

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ── Title banner ──────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#E24B4A")
    ax_title.text(0.5, 0.65, "9mobile — Customer Complaint Deep Dive",
                  ha="center", va="center", fontsize=20, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.22,
                  f"Based on {len(mobile_df)} mentions  |  "
                  f"{len(neg_df)} complaints  |  "
                  f"{round(len(neg_df)/len(mobile_df)*100, 1)}% complaint rate",
                  ha="center", va="center", fontsize=12, color="#FDECEA",
                  transform=ax_title.transAxes)
    ax_title.axis("off")

    # ── Chart 1: Sentiment breakdown (donut) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    counts = mobile_df["label"].value_counts()
    labels = counts.index.tolist()
    colors = [COLORS.get(l, "#ccc") for l in labels]
    wedges, texts, autotexts = ax1.pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title("Sentiment Breakdown", fontweight="bold", fontsize=12, pad=10)

    # ── Chart 2: Complaint topics bar ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    topic_counts = neg_exploded["topics"].value_counts().head(7)
    bars = ax2.barh(topic_counts.index[::-1], topic_counts.values[::-1],
                    color=COLORS["primary"], alpha=0.85)
    ax2.bar_label(bars, padding=3, fontsize=9, color="#333")
    ax2.set_title("Top Complaint Topics", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Number of complaints", fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=9)

    # ── Chart 3: Twitter vs News ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    src_label = mobile_df.groupby(["source","label"]).size().unstack(fill_value=0)
    src_pct   = src_label.div(src_label.sum(axis=1), axis=0) * 100
    src_colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
    existing = [c for c in ["positive","neutral","negative"] if c in src_pct.columns]
    src_pct[existing].plot(kind="bar", ax=ax3, color=[COLORS[c] for c in existing],
                           alpha=0.85, edgecolor="white", width=0.5)
    ax3.set_title("Twitter vs News Breakdown", fontweight="bold", fontsize=12)
    ax3.set_ylabel("% of mentions", fontsize=9)
    ax3.set_xlabel("")
    ax3.tick_params(axis="x", rotation=0, labelsize=10)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.spines[["top","right"]].set_visible(False)

    # ── Chart 4: Monthly complaint trend ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    monthly = (
        mobile_df.groupby(["month","label"])
        .size().unstack(fill_value=0)
        .reset_index()
    )
    if "negative" in monthly.columns:
        ax4.fill_between(monthly["month"], monthly["negative"],
                         alpha=0.25, color=COLORS["negative"])
        ax4.plot(monthly["month"], monthly["negative"],
                 color=COLORS["negative"], linewidth=2.5, marker="o",
                 markersize=5, label="Complaints")
    if "positive" in monthly.columns:
        ax4.plot(monthly["month"], monthly["positive"],
                 color=COLORS["positive"], linewidth=1.5, linestyle="--",
                 marker="s", markersize=4, alpha=0.7, label="Positive")
    # Mark worst month
    if "negative" in monthly.columns and len(monthly) > 0:
        worst_idx = monthly["negative"].idxmax()
        worst_val = monthly.loc[worst_idx, "negative"]
        worst_mon = monthly.loc[worst_idx, "month"]
        ax4.annotate(f"Peak: {worst_mon}",
                     xy=(worst_mon, worst_val),
                     xytext=(worst_mon, worst_val + 2),
                     fontsize=9, color=COLORS["negative"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=COLORS["negative"], lw=1.2))
    ax4.set_title("Monthly Complaint Volume Over Time", fontweight="bold", fontsize=12)
    ax4.set_ylabel("Number of mentions", fontsize=9)
    ax4.tick_params(axis="x", rotation=35, labelsize=8)
    ax4.legend(fontsize=9)
    ax4.spines[["top","right"]].set_visible(False)
    ax4.set_facecolor("#FAFAFA")

    # ── Chart 5: TF-IDF keywords ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    kws = get_top_keywords(neg_df["text_clean"].tolist(), n=10)
    if kws:
        words  = [k[0] for k in kws]
        scores = [k[1] for k in kws]
        cmap   = plt.cm.Reds(np.linspace(0.4, 0.85, len(words)))
        ax5.barh(words[::-1], scores[::-1], color=cmap[::-1])
        ax5.set_title("Top Complaint Keywords\n(TF-IDF)", fontweight="bold", fontsize=12)
        ax5.set_xlabel("TF-IDF score", fontsize=9)
        ax5.tick_params(axis="y", labelsize=9)
        ax5.spines[["top","right"]].set_visible(False)
    else:
        ax5.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        ax5.axis("off")

    # ── Chart 6: 9mobile vs all ISPs avg compound ─────────────────────────────
    ax6 = fig.add_subplot(gs[3, :2])
    isp_avg = all_df.groupby("isp")["compound"].mean().sort_values()
    bar_colors = [COLORS["primary"] if i == ISP else "#B4B2A9" for i in isp_avg.index]
    bars6 = ax6.barh(isp_avg.index, isp_avg.values, color=bar_colors, alpha=0.85)
    ax6.axvline(0, color="#333", linewidth=0.8, linestyle="--")
    ax6.bar_label(bars6, fmt="%.3f", padding=3, fontsize=9)
    ax6.set_title(f"{ISP} vs All ISPs — Avg Sentiment Score", fontweight="bold", fontsize=12)
    ax6.set_xlabel("Average compound score (negative = worse)", fontsize=9)
    ax6.spines[["top","right"]].set_visible(False)
    ax6.set_facecolor("#FAFAFA")

    # ── Chart 7: Worst sample reviews table ──────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis("off")
    worst = neg_df.nsmallest(4, "compound")[["text_clean","compound"]].reset_index(drop=True)
    ax7.set_title("Sample Worst Reviews", fontweight="bold", fontsize=12, loc="left")
    for i, row in worst.iterrows():
        snippet = row["text_clean"][:55] + "..." if len(row["text_clean"]) > 55 else row["text_clean"]
        ax7.text(0.0, 0.82 - i*0.22, f"[{row['compound']:.2f}]",
                 transform=ax7.transAxes, fontsize=9,
                 color=COLORS["negative"], fontweight="bold")
        ax7.text(0.18, 0.82 - i*0.22, snippet,
                 transform=ax7.transAxes, fontsize=8.5,
                 color="#333", wrap=True)

    plt.suptitle("", y=1)
    plt.savefig("9mobile_dashboard.png", dpi=180, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.show()
    print("Saved: 9mobile_dashboard.png")


# ── Print text summary ────────────────────────────────────────────────────────
def print_summary(mobile_df, all_df):
    neg_df = mobile_df[mobile_df["label"] == "negative"].copy()
    neg_df["topics"] = neg_df["text_clean"].apply(tag_topics)
    neg_exploded = neg_df.explode("topics")

    print(f"\n{'='*55}")
    print(f"  9mobile Complaint Analysis Summary")
    print(f"{'='*55}")
    print(f"  Total mentions   : {len(mobile_df)}")
    print(f"  Total complaints : {len(neg_df)}")
    print(f"  Complaint rate   : {len(neg_df)/len(mobile_df)*100:.1f}%")
    print(f"  Avg sentiment    : {mobile_df['compound'].mean():.3f}")

    rank = all_df.groupby("isp")["compound"].mean().sort_values()
    pos  = list(rank.index).index(ISP) + 1
    print(f"  Sentiment rank   : #{pos} out of {len(rank)} ISPs (1 = worst)")

    print(f"\n  Top complaint topics:")
    for topic, cnt in neg_exploded["topics"].value_counts().head(5).items():
        print(f"    {topic:<22} {cnt} complaints")

    print(f"\n  Top complaint keywords:")
    kws = get_top_keywords(neg_df["text_clean"].tolist(), n=6)
    for word, score in kws:
        print(f"    {word:<22} {score:.4f}")

    monthly_neg = mobile_df[mobile_df["label"]=="negative"].groupby("month").size()
    if len(monthly_neg) > 0:
        worst_month = monthly_neg.idxmax()
        print(f"\n  Worst month      : {worst_month} ({monthly_neg.max()} complaints)")


# ── Export CSV for Tableau ────────────────────────────────────────────────────
def export_for_tableau(mobile_df):
    neg_df = mobile_df[mobile_df["label"] == "negative"].copy()
    neg_df["topics"] = neg_df["text_clean"].apply(tag_topics)
    neg_exploded = neg_df.explode("topics").reset_index(drop=True)
    neg_exploded.to_csv("9mobile_complaints.csv", index=False)
    print(f"\nExported {len(neg_exploded)} rows → 9mobile_complaints.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    mobile_df, all_df = load_data()
    print_summary(mobile_df, all_df)
    build_dashboard(mobile_df, all_df)
    export_for_tableau(mobile_df)

    try:
        from google.colab import files
        files.download("9mobile_dashboard.png")
        files.download("9mobile_complaints.csv")
    except ImportError:
        print("Files saved locally.")


if __name__ == "__main__":
    run()
