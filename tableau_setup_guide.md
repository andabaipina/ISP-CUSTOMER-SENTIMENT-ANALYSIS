# Tableau Dashboard Setup Guide
## ISP Customer Sentiment Analysis

---

### Step 1 — Connect Tableau to SQLite

Tableau Desktop does not support SQLite natively, so use one of these two methods:

#### Option A: Export to CSV (easiest — works with Tableau Public)
Run this in your notebook or terminal:

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("isp_sentiment.db")
df = pd.read_sql("SELECT * FROM sentiment_scores", conn)
df.to_csv("sentiment_data.csv", index=False)
conn.close()
print(f"Exported {len(df)} rows to sentiment_data.csv")
```

Then in Tableau: **Connect → Text File → select sentiment_data.csv**

#### Option B: ODBC driver (Tableau Desktop only)
1. Download the SQLite ODBC driver: http://www.ch-werner.de/sqliteodbc/
2. In Tableau: **Connect → Other Databases (ODBC)**
3. Select the SQLite3 ODBC driver
4. Point it to your `isp_sentiment.db` file

---

### Step 2 — Prepare your data in Tableau

After connecting, verify these field types in the Data Source tab:

| Field         | Type     | Notes                          |
|---------------|----------|--------------------------------|
| compound      | Number   | Continuous measure             |
| positive      | Number   | Continuous measure             |
| negative      | Number   | Continuous measure             |
| neutral       | Number   | Continuous measure             |
| label         | String   | Dimension (positive/neutral/negative) |
| isp           | String   | Dimension                      |
| source        | String   | Dimension (twitter/news)       |
| published_at  | Date     | Change to Date type if needed  |

To fix the date field: right-click `published_at` → Change Data Type → Date

---

### Step 3 — Build the 6 dashboard sheets

#### Sheet 1: ISP Sentiment Leaderboard (Bar Chart)
- **Rows:** ISP
- **Columns:** AVG(compound)
- **Color:** AVG(compound) → Edit Colors → use Red-Blue diverging, center at 0
- **Sort:** Descending by AVG(compound)
- **Reference line:** Add → Line → Value: 0 (marks the neutral boundary)
- Title: "Average Sentiment Score by ISP"

#### Sheet 2: Sentiment Trend Over Time (Line Chart)
- **Columns:** published_at (set to MONTH or WEEK)
- **Rows:** AVG(compound)
- **Color:** ISP
- **Filters:** Add ISP as a quick filter
- Title: "Sentiment Trend Over Time"

#### Sheet 3: Sentiment Distribution (Stacked Bar)
- **Columns:** ISP
- **Rows:** COUNT(id)
- **Color:** label → assign: green = positive, gray = neutral, red = negative
- **Marks:** Bar → check "Stack Marks"
- Title: "Sentiment Breakdown by ISP"

#### Sheet 4: Source Comparison (Side-by-Side Bar)
- **Columns:** ISP, source
- **Rows:** AVG(compound)
- **Color:** source (Twitter = blue, News = orange)
- Title: "Twitter vs News Sentiment"

#### Sheet 5: Monthly Heatmap
- **Columns:** MONTH(published_at)
- **Rows:** ISP
- **Color:** AVG(compound) → Red-Blue diverging
- **Marks:** Square
- Title: "Sentiment Heatmap by Month"

#### Sheet 6: KPI Summary (Text Table)
- **Rows:** ISP
- **Columns:** Measure Names
- **Values:** AVG(compound), COUNT(id), % positive, % negative
- To calculate % positive:
  - Create Calculated Field: `SUM(IF [label] = "positive" THEN 1 ELSE 0 END) / COUNT([id]) * 100`
  - Name it "Positive %"
- Title: "ISP Performance Summary"

---

### Step 4 — Assemble the dashboard

1. New Dashboard → set size to **1200 × 800** (or Automatic)
2. Layout suggestion:

```
┌─────────────────────┬─────────────────────┐
│  Sheet 6: KPI Table │  Sheet 1: Bar Chart  │
│      (top left)     │     (top right)      │
├─────────────────────┴─────────────────────┤
│         Sheet 2: Trend Line               │
│              (full width)                 │
├──────────────┬────────────┬───────────────┤
│ Sheet 3:     │ Sheet 4:   │ Sheet 5:      │
│ Stacked Bar  │ Src Compare│ Heatmap       │
└──────────────┴────────────┴───────────────┘
```

3. Add a dashboard filter: drag ISP to the filter shelf → apply to all sheets
4. Add a date range filter: drag published_at → Range of Dates → apply to all sheets

---

### Step 5 — Calculated fields to create

In Tableau, go to **Analysis → Create Calculated Field** for each:

**Sentiment Category Color**
```
IF [compound] >= 0.05 THEN "Positive"
ELSEIF [compound] <= -0.05 THEN "Negative"
ELSE "Neutral"
END
```

**Positive %**
```
SUM(IF [label] = "positive" THEN 1 ELSE 0 END) / COUNT([id]) * 100
```

**Negative %**
```
SUM(IF [label] = "negative" THEN 1 ELSE 0 END) / COUNT([id]) * 100
```

**Net Sentiment Score (for KPI)**
```
(SUM(IF [label] = "positive" THEN 1 ELSE 0 END) -
 SUM(IF [label] = "negative" THEN 1 ELSE 0 END)) / COUNT([id]) * 100
```

---

### Step 6 — Publish to Tableau Public (free portfolio hosting)

1. Sign up at https://public.tableau.com (free)
2. File → Save to Tableau Public
3. Your dashboard gets a public URL you can add to your portfolio/CV
4. Note: Tableau Public workbooks are visible to everyone — don't include sensitive data

---

### Tips for a polished portfolio dashboard

- Use a consistent color palette: one color per ISP throughout all sheets
- Add tooltips with the actual text_clean field so reviewers can read sample posts
- Add a text box explaining the methodology (VADER NLP, data sources, date range)
- Include a "Last updated" dynamic text: TODAY() formatted as a date
- Name your workbook: "ISP Customer Sentiment Analysis — [Your Name]"
