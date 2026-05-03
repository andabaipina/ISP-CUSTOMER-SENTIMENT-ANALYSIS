# ISP-CUSTOMER-SENTIMENT-ANALYSIS
# 📡 ISP Customer Sentiment Analysis
### Python · SQL · Tableau | Portfolio Project

---

## 📌 Project Overview

This project analyses customer sentiment towards major Nigerian Internet Service Providers (ISPs) — **MTN, Airtel, Glo, 9mobile, Spectranet and Smile** — using data scraped from Twitter/X and news sites.

The goal is to answer three key business questions:
- Which ISP has the worst customer sentiment?
- Which ISP receives the most complaints?
- **Why** are customers dissatisfied — what specific issues drive complaints?

---

## 🔍 Key Findings

| ISP | Avg Sentiment Score | Complaint Rate | Top Complaint |
|---|---|---|---|
| Spectranet | +0.15 | Lowest | — |
| Airtel | +0.10 | Low | Coverage |
| MTN | +0.05 | Moderate | Speed |
| Smile | +0.02 | Moderate | Value |
| 9mobile | -0.12 | **Highest** | Customer service |
| Glo | -0.08 | High | Slow speed |

> **9mobile deep dive:** Analysis revealed that 9mobile scored the worst sentiment of all ISPs. Complaints are driven primarily by poor customer service response times, network outages, and data depletion issues — with complaint spikes concentrated in specific months suggesting recurring infrastructure problems.

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Scraping | requests, BeautifulSoup, snscrape | Collect tweets and news articles |
| Processing | pandas, regex, nltk | Clean and prepare text data |
| NLP | VADER, TF-IDF, LDA | Sentiment scoring and topic extraction |
| Storage | SQLite | Store and query all records |
| Analysis | SQL | Aggregate and surface insights |
| Visualisation | matplotlib, Tableau Public | Charts and interactive dashboard |

---

## 📁 Project Structure

```
isp-sentiment-analysis/
│
├── isp_sentiment_pipeline.py     # Main scraping + NLP + storage pipeline
├── generate_sample_data.py       # Seed DB with realistic sample data
├── export_for_tableau.py         # Export CSVs for Tableau connection
├── topic_analysis.py             # Rule-based + TF-IDF complaint topic extraction
├── lda_topic_modelling.py        # LDA automated theme discovery
├── 9mobile_deep_dive.py          # 9mobile focused deep dive dashboard
│
├── isp_sentiment_queries.sql     # 11 SQL analysis queries + schema
├── requirements.txt              # Python dependencies
│
├── outputs/
│   ├── isp_sentiment.db          # SQLite database
│   ├── sentiment_data.csv        # Main Tableau data source
│   ├── isp_summary.csv           # KPI summary data
│   ├── monthly_trend.csv         # Trend line data
│   ├── topic_data.csv            # Complaint topics data
│   ├── keywords_per_isp.csv      # TF-IDF keywords per ISP
│   ├── lda_reviews.csv           # LDA topic assignments
│   └── 9mobile_complaints.csv    # 9mobile deep dive data
│
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon punkt stopwords
```

### 2. Run the scraping pipeline
```bash
python isp_sentiment_pipeline.py
```

### 3. Seed sample data
```bash
python generate_sample_data.py
```

### 4. Export to Tableau
```bash
python export_for_tableau.py
```

### 5. Run topic analysis
```bash
python topic_analysis.py
python lda_topic_modelling.py
```

### 6. Run 9mobile deep dive
```bash
python 9mobile_deep_dive.py
```

---

## 🗄️ Database Schema

```sql
CREATE TABLE raw_data (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT,
    isp          TEXT,
    text         TEXT,
    url          TEXT,
    author       TEXT,
    published_at TEXT,
    scraped_at   TEXT
);

CREATE TABLE sentiment_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT,
    isp          TEXT,
    text_clean   TEXT,
    compound     REAL,
    positive     REAL,
    neutral      REAL,
    negative     REAL,
    label        TEXT,
    published_at TEXT,
    scraped_at   TEXT,
    url          TEXT
);
```

---

## 📊 Tableau Dashboards

### Dashboard 1 — ISP Sentiment Overview
| Sheet | Chart | Insight |
|---|---|---|
| KPI Summary | Text table | At-a-glance scores |
| Leaderboard | Bar chart | ISP ranking by sentiment |
| Complaint Rate | Dual bar | Volume vs rate |
| Trend | Line chart | Sentiment over time |
| Stacked bar | Bar chart | Sentiment mix |
| Source compare | Side-by-side bar | Twitter vs News |
| Heatmap | Square marks | Monthly patterns |

### Dashboard 2 — 9mobile Deep Dive
| Sheet | Chart | Insight |
|---|---|---|
| KPI Banner | Text | Total complaints, rate, avg score |
| Cause For Dissatisfaction| Bar Chart| Sentiment breakdown |
| Topics | Bar | What customers complain about |
| Trend | Line | When complaints spiked |
| Keywords | Bar | TF-IDF complaint language |
| ISP Comparison | Bar | 9mobile vs competitors |
| Twitter vs News | Stacked bar | Source of complaints |

🔗View Live Dashboard on Tableau
https://public.tableau.com/authoring/ISPCUSTOMERSENTIMENTANALYSIS/Sheet5/Dashboard%201#2

---

## 🧠 NLP Methodology

**VADER Sentiment Scoring**
Tuned for social media text. Each record gets a compound score (-1.0 to +1.0) and a label — positive, neutral, or negative.

**Rule-based Topic Tagging**
Maps complaints to 8 categories: Network outage, Slow speed, Data depletion, Billing issues, Customer service, Poor coverage, Value for money, App issues.

**LDA Topic Modelling**
Automatically discovers hidden complaint themes without predefined categories.

**TF-IDF Keywords**
Surfaces words uniquely important to each ISP's complaints.

---

## 💡 Key Insights

1. Glo and 9mobile score below neutral with the highest complaint rates
2. Network speed and outages are the most common complaint themes industry-wide
3. Customer service quality is a key differentiator between ISPs
4. Twitter drives more complaints than news coverage
5. Complaint spikes are seasonal — suggesting recurring infrastructure issues

---

## 🚀 Future Improvements

- Real-time scraping with scheduling (Airflow or cron)
- Expand to more ISPs and regions
- Aspect-based sentiment analysis (ABSA)
- Live Streamlit web app dashboard
- PostgreSQL for production-scale storage
- Named entity recognition (NER)

---

## 👤 Author

**[Your Name]**
[LinkedIn](#) · [GitHub](#) · [Tableau Public](#)

---

*Data collected for educational and portfolio purposes only.*
