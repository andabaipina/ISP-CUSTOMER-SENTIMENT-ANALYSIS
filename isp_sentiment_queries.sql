-- ============================================================
-- ISP Sentiment Analysis — SQL Schema & Analysis Queries
-- Compatible with SQLite and PostgreSQL
-- ============================================================


-- ── SCHEMA ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw_data (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,          -- 'twitter' | 'news'
    isp          TEXT NOT NULL,          -- e.g. 'MTN'
    text         TEXT NOT NULL,
    url          TEXT,
    author       TEXT,
    published_at TEXT,
    scraped_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT    NOT NULL,
    isp          TEXT    NOT NULL,
    text_clean   TEXT    NOT NULL,
    compound     REAL    NOT NULL,       -- -1.0 (most negative) to +1.0 (most positive)
    positive     REAL    NOT NULL,       -- proportion of positive sentiment
    neutral      REAL    NOT NULL,
    negative     REAL    NOT NULL,
    label        TEXT    NOT NULL,       -- 'positive' | 'neutral' | 'negative'
    published_at TEXT,
    scraped_at   TEXT    NOT NULL,
    url          TEXT
);

-- Indexes for fast filtering in Tableau
CREATE INDEX IF NOT EXISTS idx_isp       ON sentiment_scores(isp);
CREATE INDEX IF NOT EXISTS idx_label     ON sentiment_scores(label);
CREATE INDEX IF NOT EXISTS idx_published ON sentiment_scores(published_at);
CREATE INDEX IF NOT EXISTS idx_source    ON sentiment_scores(source);


-- ============================================================
-- ANALYSIS QUERIES
-- ============================================================


-- ── Q1: Overall sentiment leaderboard ─────────────────────────
-- Which ISP has the best/worst customer sentiment overall?
SELECT
    isp,
    COUNT(*)                                                      AS total_mentions,
    ROUND(AVG(compound), 3)                                       AS avg_compound_score,
    SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END)          AS positive_count,
    SUM(CASE WHEN label = 'neutral'  THEN 1 ELSE 0 END)          AS neutral_count,
    SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END)          AS negative_count,
    ROUND(
        100.0 * SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                             AS positive_pct,
    ROUND(
        100.0 * SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                             AS negative_pct
FROM sentiment_scores
GROUP BY isp
ORDER BY avg_compound_score DESC;


-- ── Q2: Daily sentiment trend per ISP ─────────────────────────
-- Great for the line chart in Tableau. Connect to this as a custom SQL source.
SELECT
    DATE(published_at)      AS day,
    isp,
    ROUND(AVG(compound), 3) AS avg_sentiment,
    COUNT(*)                AS mentions
FROM sentiment_scores
WHERE published_at IS NOT NULL
GROUP BY DATE(published_at), isp
ORDER BY day, isp;


-- ── Q3: Weekly sentiment trend ────────────────────────────────
-- Smoothed view for trend lines (less noisy than daily)
SELECT
    strftime('%Y-W%W', published_at)  AS week,
    isp,
    ROUND(AVG(compound), 3)           AS avg_sentiment,
    COUNT(*)                          AS mentions
FROM sentiment_scores
WHERE published_at IS NOT NULL
GROUP BY week, isp
ORDER BY week, isp;


-- ── Q4: Sentiment by source ───────────────────────────────────
-- Do Twitter users rate ISPs differently from news coverage?
SELECT
    source,
    isp,
    ROUND(AVG(compound), 3)   AS avg_compound,
    COUNT(*)                   AS total
FROM sentiment_scores
GROUP BY source, isp
ORDER BY isp, source;


-- ── Q5: Sentiment distribution heatmap data ───────────────────
-- For a Tableau heatmap: ISP (rows) vs label (columns)
SELECT
    isp,
    label,
    COUNT(*) AS count
FROM sentiment_scores
GROUP BY isp, label
ORDER BY isp, label;


-- ── Q6: Top negative records ──────────────────────────────────
-- Review the most negative content to understand complaint themes
SELECT
    isp,
    source,
    text_clean,
    compound,
    published_at
FROM sentiment_scores
WHERE label = 'negative'
ORDER BY compound ASC
LIMIT 50;


-- ── Q7: Top positive records ──────────────────────────────────
SELECT
    isp,
    source,
    text_clean,
    compound,
    published_at
FROM sentiment_scores
WHERE label = 'positive'
ORDER BY compound DESC
LIMIT 50;


-- ── Q8: Month-over-month sentiment change ─────────────────────
-- Flag improving or declining ISPs
SELECT
    isp,
    strftime('%Y-%m', published_at)   AS month,
    ROUND(AVG(compound), 3)           AS avg_sentiment,
    COUNT(*)                          AS mentions
FROM sentiment_scores
WHERE published_at IS NOT NULL
GROUP BY isp, month
ORDER BY isp, month;


-- ── Q9: ISP mention volume by source ─────────────────────────
-- Useful for a stacked bar chart in Tableau
SELECT
    isp,
    source,
    COUNT(*) AS mention_count
FROM sentiment_scores
GROUP BY isp, source
ORDER BY isp, source;


-- ── Q10: Sentiment score distribution buckets ────────────────
-- For a histogram in Tableau
SELECT
    isp,
    CASE
        WHEN compound BETWEEN  0.5 AND  1.0 THEN 'Very positive (0.5–1.0)'
        WHEN compound BETWEEN  0.05 AND  0.5 THEN 'Positive (0.05–0.5)'
        WHEN compound BETWEEN -0.05 AND  0.05 THEN 'Neutral (-0.05–0.05)'
        WHEN compound BETWEEN -0.5 AND -0.05 THEN 'Negative (-0.5–-0.05)'
        ELSE                                       'Very negative (-1.0–-0.5)'
    END AS score_bucket,
    COUNT(*) AS count
FROM sentiment_scores
GROUP BY isp, score_bucket
ORDER BY isp, score_bucket;


-- ── Q11: Tableau dashboard summary view ──────────────────────
-- Use this as a saved view / named query in Tableau
CREATE VIEW IF NOT EXISTS v_dashboard_summary AS
SELECT
    s.isp,
    s.source,
    s.label,
    s.compound,
    s.positive,
    s.negative,
    s.neutral,
    DATE(s.published_at)                          AS date,
    strftime('%Y-%m', s.published_at)             AS month,
    strftime('%Y-W%W', s.published_at)            AS week,
    s.url
FROM sentiment_scores s
WHERE s.published_at IS NOT NULL;
