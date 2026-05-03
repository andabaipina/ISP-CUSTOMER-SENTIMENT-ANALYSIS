"""
Export isp_sentiment.db to CSV files for Tableau connection.
Run this after the pipeline has collected data (or after seeding sample data).

    python export_for_tableau.py
"""

import sqlite3
import pandas as pd

DB_PATH = "isp_sentiment.db"

EXPORTS = {
    "sentiment_data.csv": """
        SELECT
            id,
            source,
            isp,
            text_clean,
            compound,
            positive,
            neutral,
            negative,
            label,
            DATE(published_at)                         AS date,
            strftime('%Y-%m', published_at)            AS month,
            strftime('%Y-W%W', published_at)           AS week,
            CAST(strftime('%H', published_at) AS INT)  AS hour_of_day,
            url
        FROM sentiment_scores
        WHERE published_at IS NOT NULL
        ORDER BY published_at DESC
    """,
    "isp_summary.csv": """
        SELECT
            isp,
            COUNT(*)                                                       AS total_mentions,
            ROUND(AVG(compound), 3)                                        AS avg_compound,
            SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END)           AS positive_count,
            SUM(CASE WHEN label = 'neutral'  THEN 1 ELSE 0 END)           AS neutral_count,
            SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END)           AS negative_count,
            ROUND(100.0 * SUM(CASE WHEN label = 'positive' THEN 1 ELSE 0 END) / COUNT(*), 1) AS positive_pct,
            ROUND(100.0 * SUM(CASE WHEN label = 'negative' THEN 1 ELSE 0 END) / COUNT(*), 1) AS negative_pct
        FROM sentiment_scores
        GROUP BY isp
        ORDER BY avg_compound DESC
    """,
    "monthly_trend.csv": """
        SELECT
            strftime('%Y-%m', published_at) AS month,
            isp,
            source,
            ROUND(AVG(compound), 3)         AS avg_sentiment,
            COUNT(*)                         AS mentions
        FROM sentiment_scores
        WHERE published_at IS NOT NULL
        GROUP BY month, isp, source
        ORDER BY month, isp
    """,
}


def export_all():
    conn = sqlite3.connect(DB_PATH)
    total_rows = 0

    for filename, query in EXPORTS.items():
        df = pd.read_sql_query(query, conn)
        df.to_csv(filename, index=False)
        total_rows += len(df)
        print(f"Exported {len(df):>5} rows → {filename}")

    conn.close()
    print(f"\nDone. Connect Tableau to sentiment_data.csv for the main dashboard.")
    print("Use isp_summary.csv for KPI cards and monthly_trend.csv for trend lines.")


if __name__ == "__main__":
    export_all()
