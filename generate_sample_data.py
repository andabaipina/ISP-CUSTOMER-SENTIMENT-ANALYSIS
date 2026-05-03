"""
ISP Sentiment — Sample Data Generator
======================================
Populates isp_sentiment.db with realistic synthetic data so you can
build and test your Tableau dashboard without waiting for a full scrape.

Run once:
    python generate_sample_data.py
"""

import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "isp_sentiment.db"

ISP_NAMES = ["MTN", "Airtel", "Glo", "9mobile", "Spectranet", "Smile"]

# Realistic sentiment profiles per ISP (avg_compound, std_dev)
ISP_PROFILES = {
    "MTN":        (0.05,  0.35),
    "Airtel":     (0.10,  0.30),
    "Glo":        (-0.12, 0.38),
    "9mobile":    (-0.08, 0.33),
    "Spectranet": (0.15,  0.28),
    "Smile":      (0.02,  0.32),
}

SAMPLE_TWEETS = {
    "positive": [
        "mtn internet speed has been great this week finally enjoying browsing",
        "airtel data plan worth every naira excellent network coverage",
        "spectranet fibre connection never disappoints fast reliable",
        "mtn customer service actually resolved my issue quickly impressed",
        "airtel 5g speeds are incredible streaming without buffering",
        "smile internet perfect for working from home no complaints",
        "just renewed my mtn data plan value for money as always",
        "airtel network coverage improved a lot in my area good job",
    ],
    "neutral": [
        "comparing mtn vs airtel data plans for monthly subscription",
        "anyone know the best isp in lagos for home internet",
        "mtn network maintenance scheduled for sunday morning",
        "airtel just launched new monthly data bundles check their website",
        "glo network coverage map updated for 2024",
        "9mobile and airtel merger talks what does it mean for customers",
        "spectranet vs smile which is better for small business",
        "mtn data prices unchanged this quarter",
    ],
    "negative": [
        "mtn internet completely down for 3 hours terrible service",
        "glo network always slow during peak hours so frustrating",
        "9mobile customer service kept me on hold for 45 minutes unacceptable",
        "airtel data finished before it should have suspected billing fraud",
        "glo internet speed is embarrassingly slow cannot work from home",
        "mtn network congestion every evening ruins my streaming experience",
        "9mobile coverage dead zone in my entire neighborhood useless",
        "worst customer service experience ever with glo never again",
        "airtel keeps throttling my connection after half my data is used",
        "glo internet disconnects every few minutes how is this acceptable",
    ],
}

SAMPLE_NEWS = {
    "positive": [
        "{isp} expands 4g lte coverage across 12 new states improves connectivity",
        "{isp} wins best telecom provider award customer satisfaction survey",
        "{isp} cuts data prices by 20 percent for all subscribers",
        "{isp} launches fibre broadband service fastest speeds in region",
        "{isp} customer satisfaction scores hit five year high",
    ],
    "neutral": [
        "{isp} announces network upgrade scheduled maintenance period",
        "{isp} quarterly earnings meet analyst expectations revenue stable",
        "{isp} signs partnership deal with international telecom provider",
        "{isp} introduces new business data plans for enterprise customers",
        "{isp} regulatory filing submitted to ncc for spectrum renewal",
    ],
    "negative": [
        "{isp} faces backlash over unexpected data depletion complaints surge",
        "{isp} outage affects thousands of customers in major cities",
        "{isp} slapped with fine by ncc over quality of service violations",
        "{isp} customer complaints up 35 percent in latest industry report",
        "{isp} network reliability drops amid infrastructure challenges",
    ],
}

COMPOUND_RANGES = {
    "positive": (0.15, 0.95),
    "negative": (-0.95, -0.15),
    "neutral":  (-0.05, 0.05),
}


def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"


def random_date(days_back: int = 90) -> datetime:
    return datetime.utcnow() - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def generate_records(n_per_isp: int = 120) -> list[dict]:
    records = []
    for isp in ISP_NAMES:
        avg_c, std_c = ISP_PROFILES[isp]
        for _ in range(n_per_isp):
            source = random.choice(["twitter", "twitter", "news"])  # 2:1 twitter bias
            compound = float(max(-1.0, min(1.0, random.gauss(avg_c, std_c))))
            label = label_from_compound(compound)

            if source == "twitter":
                text = random.choice(SAMPLE_TWEETS[label])
            else:
                template = random.choice(SAMPLE_NEWS[label])
                text = template.format(isp=isp.lower())

            pub_date = random_date(90)
            records.append({
                "source":       source,
                "isp":          isp,
                "text_clean":   text,
                "compound":     round(compound, 4),
                "positive":     round(max(0, compound) * random.uniform(0.6, 1.0), 4),
                "neutral":      round(random.uniform(0.1, 0.5), 4),
                "negative":     round(abs(min(0, compound)) * random.uniform(0.6, 1.0), 4),
                "label":        label,
                "published_at": pub_date.isoformat(),
                "scraped_at":   datetime.utcnow().isoformat(),
                "url":          f"https://example.com/{isp.lower()}/{random.randint(1000,9999)}",
            })
    return records


def seed_database(n_per_isp: int = 120) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT    NOT NULL,
            isp          TEXT    NOT NULL,
            text_clean   TEXT    NOT NULL,
            compound     REAL    NOT NULL,
            positive     REAL    NOT NULL,
            neutral      REAL    NOT NULL,
            negative     REAL    NOT NULL,
            label        TEXT    NOT NULL,
            published_at TEXT,
            scraped_at   TEXT    NOT NULL,
            url          TEXT
        )
    """)

    records = generate_records(n_per_isp)
    cursor.executemany("""
        INSERT INTO sentiment_scores
            (source, isp, text_clean, compound, positive, neutral, negative,
             label, published_at, scraped_at, url)
        VALUES
            (:source, :isp, :text_clean, :compound, :positive, :neutral, :negative,
             :label, :published_at, :scraped_at, :url)
    """, records)

    conn.commit()
    total = cursor.execute("SELECT COUNT(*) FROM sentiment_scores").fetchone()[0]
    conn.close()

    print(f"Inserted {len(records)} sample records.")
    print(f"Total records in DB: {total}")
    print(f"Database: {DB_PATH}")
    print("\nSentiment breakdown per ISP:")
    print(f"{'ISP':<12} {'Total':>6} {'Avg Score':>10} {'Pos%':>6} {'Neg%':>6}")
    print("-" * 44)
    for isp in ISP_NAMES:
        isp_recs = [r for r in records if r["isp"] == isp]
        avg_c = sum(r["compound"] for r in isp_recs) / len(isp_recs)
        pos_pct = 100 * sum(1 for r in isp_recs if r["label"] == "positive") / len(isp_recs)
        neg_pct = 100 * sum(1 for r in isp_recs if r["label"] == "negative") / len(isp_recs)
        print(f"{isp:<12} {len(isp_recs):>6} {avg_c:>10.3f} {pos_pct:>5.1f}% {neg_pct:>5.1f}%")


if __name__ == "__main__":
    seed_database(n_per_isp=120)  # 120 records × 6 ISPs = 720 total
