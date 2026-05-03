# ISP Customer Sentiment Analysis
## Portfolio Project | Python · SQL · Tableau

---

### Project structure

```
isp-sentiment/
├── isp_sentiment_pipeline.py   ← Main pipeline (scrape → clean → NLP → store)
├── isp_sentiment_queries.sql   ← SQL schema + 11 analysis queries
├── requirements.txt            ← Python dependencies
└── isp_sentiment.db            ← SQLite database (auto-created on first run)
```

---

### Setup

**1. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2. Download NLTK data** (auto-runs on first pipeline execution, or manually):
```bash
python -c "import nltk; nltk.download(['vader_lexicon', 'punkt', 'stopwords'])"
```

**3. (Optional) Add your NewsAPI key**
Edit `isp_sentiment_pipeline.py` and replace:
```python
NEWS_API_KEY = "YOUR_NEWSAPI_KEY_HERE"
```
Get a free key at https://newsapi.org — 500 requests/day free tier.
Without a key, the pipeline falls back to Google News RSS automatically.

**4. (Optional) Twitter/X scraping**
The pipeline uses `snscrape` which requires no API key:
```bash
pip install snscrape
```
Or use the official Twitter API v2 with tweepy — see comments in the script.

---

### Running the pipeline

```bash
# Full pipeline — scrapes, scores, saves to DB
python isp_sentiment_pipeline.py

# View summary stats from the DB
python isp_sentiment_pipeline.py stats
```

---

### Connecting Tableau to the database

1. Open Tableau Desktop / Tableau Public
2. Connect → SQLite → select `isp_sentiment.db`
3. Drag the `sentiment_scores` table to the canvas
   — OR use **Custom SQL** and paste any query from `isp_sentiment_queries.sql`
4. Use the `v_dashboard_summary` view for the main dashboard sheet

**Recommended Tableau charts:**

| Chart type      | Fields                              | Purpose                       |
|-----------------|-------------------------------------|-------------------------------|
| Line chart      | date → avg(compound) by ISP         | Sentiment trend over time     |
| Bar chart       | ISP → avg(compound)                 | ISP comparison leaderboard    |
| Stacked bar     | ISP → count, colored by label       | Positive/neutral/negative mix |
| Heatmap         | ISP (rows) × month (cols) → avg compound | Seasonal patterns        |
| Word cloud      | Export text_clean to a word cloud tool | Top complaint keywords    |
| KPI card        | avg(compound) for each ISP          | At-a-glance sentiment score   |

---

### Customising for your region

Edit `ISP_NAMES` in `isp_sentiment_pipeline.py`:
```python
ISP_NAMES = ["MTN", "Airtel", "Glo", "9mobile", "Spectranet", "Smile"]
```

Edit `TWITTER_KEYWORDS` to tune what topics are collected:
```python
TWITTER_KEYWORDS = [
    "internet down", "slow internet", "customer service",
    "data plan", "network issue", "speed", "billing", "outage",
]
```

---

### Tech stack

| Layer        | Tool                        |
|--------------|-----------------------------|
| Scraping     | `requests`, `BeautifulSoup`, `snscrape` |
| Data wrangling | `pandas`                  |
| NLP / Sentiment | `VADER` (nltk)           |
| Storage      | `SQLite` (upgrade to PostgreSQL for production) |
| Visualisation | Tableau Public / Desktop   |
