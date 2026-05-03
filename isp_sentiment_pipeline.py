"""
ISP Customer Sentiment Analysis Pipeline
=========================================
Scrapes Twitter/X mentions and news articles about ISPs,
runs NLP sentiment analysis, and stores results in SQLite.

Requirements:
    pip install requests beautifulsoup4 pandas nltk snscrape newsapi-python
    python -m nltk.downloader vader_lexicon punkt stopwords
"""

import sqlite3
import re
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Target ISPs — update these to match your region
ISP_NAMES = ["MTN", "Airtel", "Glo", "9mobile", "Spectranet", "Smile"]

# NewsAPI key — get a free key at https://newsapi.org
NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"

# SQLite database path
DB_PATH = "isp_sentiment.db"

# How many days back to look for news articles
LOOKBACK_DAYS = 30

# Twitter/X search terms per ISP (appended to ISP name)
TWITTER_KEYWORDS = [
    "internet down", "slow internet", "customer service",
    "data plan", "network issue", "speed", "billing", "outage",
]


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class RawRecord:
    source: str           # "twitter" | "news"
    isp: str              # e.g. "MTN"
    text: str             # raw scraped text
    url: str
    author: Optional[str]
    published_at: str     # ISO-8601 string
    scraped_at: str       # ISO-8601 string


@dataclass
class SentimentRecord:
    source: str
    isp: str
    text_clean: str
    compound: float       # VADER compound score: -1.0 to +1.0
    positive: float
    neutral: float
    negative: float
    label: str            # "positive" | "neutral" | "negative"
    published_at: str
    scraped_at: str
    url: str


# ── Database setup ────────────────────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist and return a connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT NOT NULL,
            isp          TEXT NOT NULL,
            text         TEXT NOT NULL,
            url          TEXT,
            author       TEXT,
            published_at TEXT,
            scraped_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source       TEXT NOT NULL,
            isp          TEXT NOT NULL,
            text_clean   TEXT NOT NULL,
            compound     REAL NOT NULL,
            positive     REAL NOT NULL,
            neutral      REAL NOT NULL,
            negative     REAL NOT NULL,
            label        TEXT NOT NULL,
            published_at TEXT,
            scraped_at   TEXT NOT NULL,
            url          TEXT,
            FOREIGN KEY (id) REFERENCES raw_data(id)
        );

        CREATE INDEX IF NOT EXISTS idx_isp         ON sentiment_scores(isp);
        CREATE INDEX IF NOT EXISTS idx_label       ON sentiment_scores(label);
        CREATE INDEX IF NOT EXISTS idx_published   ON sentiment_scores(published_at);
    """)

    conn.commit()
    log.info("Database initialised at: %s", db_path)
    return conn


# ── Text cleaning ─────────────────────────────────────────────────────────────
_STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/mentions/special chars, strip stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                        # Twitter @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                   # hashtags → plain word
    text = re.sub(r"[^\w\s]", " ", text)                    # punctuation
    text = re.sub(r"\s+", " ", text).strip()                # extra whitespace
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── Sentiment scoring ─────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> dict:
    """Return VADER scores and a human-readable label."""
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "compound": round(compound, 4),
        "positive": round(scores["pos"], 4),
        "neutral":  round(scores["neu"], 4),
        "negative": round(scores["neg"], 4),
        "label":    label,
    }


# ── Twitter / X scraping ──────────────────────────────────────────────────────
def scrape_twitter(isp: str, max_results: int = 100) -> list[RawRecord]:
    """
    Scrape tweets mentioning an ISP using snscrape (no API key needed).

    snscrape must be installed separately:
        pip install snscrape

    If you have a Twitter Developer API key, replace this with tweepy:
        import tweepy
        client = tweepy.Client(bearer_token="YOUR_BEARER_TOKEN")
        response = client.search_recent_tweets(query=query, max_results=100)
    """
    try:
        import snscrape.modules.twitter as sntwitter  # type: ignore
    except ImportError:
        log.warning("snscrape not installed. Run: pip install snscrape")
        return []

    records: list[RawRecord] = []
    scraped_at = datetime.utcnow().isoformat()

    for keyword in TWITTER_KEYWORDS[:3]:  # limit keywords per run
        query = f"{isp} {keyword} lang:en"
        log.info("Twitter search: %s", query)
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= max_results:
                    break
                records.append(RawRecord(
                    source="twitter",
                    isp=isp,
                    text=tweet.rawContent,
                    url=f"https://twitter.com/i/web/status/{tweet.id}",
                    author=tweet.user.username,
                    published_at=tweet.date.isoformat(),
                    scraped_at=scraped_at,
                ))
            time.sleep(1)  # be polite
        except Exception as exc:
            log.error("Twitter scrape failed for '%s': %s", query, exc)

    log.info("Collected %d tweets for %s", len(records), isp)
    return records


# ── News scraping via NewsAPI ─────────────────────────────────────────────────
def scrape_news_api(isp: str, max_results: int = 50) -> list[RawRecord]:
    """
    Fetch news articles mentioning an ISP using the NewsAPI.
    Sign up for a free key at https://newsapi.org (500 req/day free tier).
    """
    if NEWS_API_KEY == "YOUR_NEWSAPI_KEY_HERE":
        log.warning("NewsAPI key not set. Falling back to direct scrape.")
        return scrape_news_direct(isp)

    from_date = (datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{isp} internet",
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": min(max_results, 100),
        "apiKey": NEWS_API_KEY,
    }

    records: list[RawRecord] = []
    scraped_at = datetime.utcnow().isoformat()

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".strip()
            if not text:
                continue
            records.append(RawRecord(
                source="news",
                isp=isp,
                text=text,
                url=article.get("url", ""),
                author=article.get("author"),
                published_at=article.get("publishedAt", ""),
                scraped_at=scraped_at,
            ))
    except Exception as exc:
        log.error("NewsAPI failed for '%s': %s", isp, exc)

    log.info("Collected %d news articles for %s", len(records), isp)
    return records


def scrape_news_direct(isp: str) -> list[RawRecord]:
    """
    Fallback: scrape Google News RSS for ISP headlines (no API key needed).
    Good for development / when NewsAPI quota is exhausted.
    """
    records: list[RawRecord] = []
    scraped_at = datetime.utcnow().isoformat()
    query = f"{isp}+internet+customer+service"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:50]
        for item in items:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            desc = item.find("description")
            desc_text = BeautifulSoup(desc.get_text(), "html.parser").get_text() if desc else ""
            text = f"{title} {desc_text}".strip()
            pub_date = item.find("pubDate")
            records.append(RawRecord(
                source="news",
                isp=isp,
                text=text,
                url=item.find("link").get_text() if item.find("link") else "",
                author=None,
                published_at=pub_date.get_text() if pub_date else "",
                scraped_at=scraped_at,
            ))
        time.sleep(1)
    except Exception as exc:
        log.error("Direct news scrape failed for '%s': %s", isp, exc)

    log.info("Collected %d news items for %s (direct)", len(records), isp)
    return records


# ── Pipeline orchestration ────────────────────────────────────────────────────
def process_records(raw_records: list[RawRecord]) -> list[SentimentRecord]:
    """Clean text and score sentiment for a batch of raw records."""
    processed = []
    for rec in raw_records:
        clean = clean_text(rec.text)
        if len(clean) < 10:
            continue  # skip near-empty records after cleaning
        scores = score_sentiment(clean)
        processed.append(SentimentRecord(
            source=rec.source,
            isp=rec.isp,
            text_clean=clean,
            compound=scores["compound"],
            positive=scores["positive"],
            neutral=scores["neutral"],
            negative=scores["negative"],
            label=scores["label"],
            published_at=rec.published_at,
            scraped_at=rec.scraped_at,
            url=rec.url,
        ))
    return processed


def save_to_db(conn: sqlite3.Connection,
               raw: list[RawRecord],
               scored: list[SentimentRecord]) -> None:
    """Insert raw and scored records into SQLite, skipping duplicates."""
    cursor = conn.cursor()

    raw_inserted = 0
    for rec in raw:
        try:
            cursor.execute(
                """INSERT INTO raw_data (source, isp, text, url, author, published_at, scraped_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rec.source, rec.isp, rec.text, rec.url,
                 rec.author, rec.published_at, rec.scraped_at),
            )
            raw_inserted += 1
        except sqlite3.IntegrityError:
            pass

    scored_inserted = 0
    for rec in scored:
        try:
            cursor.execute(
                """INSERT INTO sentiment_scores
                   (source, isp, text_clean, compound, positive, neutral, negative,
                    label, published_at, scraped_at, url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (rec.source, rec.isp, rec.text_clean, rec.compound, rec.positive,
                 rec.neutral, rec.negative, rec.label, rec.published_at,
                 rec.scraped_at, rec.url),
            )
            scored_inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    log.info("Saved %d raw | %d scored records to DB", raw_inserted, scored_inserted)


def run_pipeline(isps: list[str] = ISP_NAMES) -> pd.DataFrame:
    """
    Full end-to-end pipeline:
      1. Scrape Twitter + News for each ISP
      2. Clean and score sentiment
      3. Store in SQLite
      4. Return a summary DataFrame
    """
    # Download NLTK data if needed
    for resource in ["vader_lexicon", "punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt"
                           else f"corpora/{resource}" if resource == "stopwords"
                           else f"sentiment/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    conn = init_db()
    all_scored: list[SentimentRecord] = []

    for isp in isps:
        log.info("── Processing ISP: %s ──", isp)
        raw: list[RawRecord] = []

        # Collect from both sources
        raw += scrape_twitter(isp, max_results=50)
        raw += scrape_news_api(isp, max_results=30)

        if not raw:
            log.warning("No data collected for %s", isp)
            continue

        scored = process_records(raw)
        save_to_db(conn, raw, scored)
        all_scored.extend(scored)
        time.sleep(2)  # polite delay between ISPs

    conn.close()

    if not all_scored:
        log.warning("Pipeline completed with no data.")
        return pd.DataFrame()

    # Build summary report
    df = pd.DataFrame([asdict(r) for r in all_scored])
    summary = (
        df.groupby(["isp", "label"])
          .size()
          .unstack(fill_value=0)
          .assign(total=lambda x: x.sum(axis=1))
          .assign(sentiment_score=lambda x: (
              (x.get("positive", 0) - x.get("negative", 0)) / x["total"] * 100
          ).round(1))
    )

    log.info("\n%s", summary.to_string())
    log.info("Pipeline complete. Database saved to: %s", DB_PATH)
    return df


# ── Quick analytics queries ───────────────────────────────────────────────────
def get_summary_stats(db_path: str = DB_PATH) -> None:
    """Print useful summary stats from the database."""
    conn = sqlite3.connect(db_path)

    queries = {
        "Overall sentiment by ISP": """
            SELECT isp,
                   COUNT(*) AS total,
                   ROUND(AVG(compound), 3) AS avg_compound,
                   SUM(CASE WHEN label='positive' THEN 1 ELSE 0 END) AS positive,
                   SUM(CASE WHEN label='neutral'  THEN 1 ELSE 0 END) AS neutral,
                   SUM(CASE WHEN label='negative' THEN 1 ELSE 0 END) AS negative
            FROM sentiment_scores
            GROUP BY isp
            ORDER BY avg_compound DESC
        """,
        "Daily sentiment trend (last 14 days)": """
            SELECT DATE(published_at) AS day,
                   isp,
                   ROUND(AVG(compound), 3) AS avg_sentiment,
                   COUNT(*) AS mentions
            FROM sentiment_scores
            WHERE published_at >= DATE('now', '-14 days')
            GROUP BY day, isp
            ORDER BY day DESC, isp
            LIMIT 40
        """,
        "Source breakdown": """
            SELECT source, isp, label, COUNT(*) AS count
            FROM sentiment_scores
            GROUP BY source, isp, label
            ORDER BY isp, source, label
        """,
    }

    for title, sql in queries.items():
        print(f"\n{'='*60}")
        print(f"  {title}")
        print("="*60)
        df = pd.read_sql_query(sql, conn)
        print(df.to_string(index=False))

    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        # python isp_sentiment_pipeline.py stats
        get_summary_stats()
    else:
        # python isp_sentiment_pipeline.py
        run_pipeline()
