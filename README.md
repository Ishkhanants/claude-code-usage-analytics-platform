# Claude Code Analytics Platform

End-to-end analytics platform for Claude Code developer telemetry — from raw JSONL event streams to an interactive Streamlit dashboard with ML-powered insights.

## Architecture

Data flows through four sequential layers:

1. **Ingestion** — `generate_fake_data.py` produces synthetic JSONL telemetry and a CSV employee directory
2. **ETL** — `src/etl.py` streams the JSONL line-by-line, batch-inserts events into SQLite, then aggregates sessions in a second SQL pass
3. **Analytics / ML** — `src/analytics.py`, `src/ml.py`, and `src/cohort.py` query the warehouse and return DataFrames ready for rendering
4. **Dashboard** — `src/dashboard.py` consumes those functions and renders a 7-page Streamlit interface with Plotly charts

```
claude-code-usage-analytics-platform/
├── data/                          # Generated telemetry (gitignored)
│   ├── telemetry_logs.jsonl       # Raw JSONL event batches
│   └── employees.csv              # Engineer directory
├── db/
│   └── analytics.db               # SQLite warehouse (auto-created)
├── src/
│   ├── db.py                      # Schema + connection management
│   ├── etl.py                     # Streaming ETL pipeline
│   ├── analytics.py               # SQL-backed analytics functions
│   ├── ml.py                      # ML: forecasting, anomaly detection, clustering
│   ├── cohort.py                  # Retention, funnels, inter-session analysis
│   ├── validation.py              # Data quality checks + scored report
│   └── dashboard.py               # Streamlit multi-page dashboard
├── generate_fake_data.py          # Synthetic data generator
└── requirements.txt
```

## Dependencies

Requires **Python 3.8+**

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.35.0 | Dashboard framework |
| `pandas` | ≥2.0.0 | DataFrame operations |
| `numpy` | ≥1.24.0 | Numerical computing |
| `plotly` | ≥5.18.0 | Interactive charts |
| `scikit-learn` | ≥1.3.0 | ML models (IsolationForest, KMeans, polynomial regression) |
| `scipy` | ≥1.11.0 | Statistical tests (ANOVA, Spearman) |

Install all at once:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data (adjust scale as needed)
python3 generate_fake_data.py --num-users 80 --num-sessions 3000 --days 60

# 3. Run ETL (auto-runs on first dashboard launch, or manually)
python3 src/etl.py

# 4. Launch dashboard
streamlit run src/dashboard.py
```

The ETL step is idempotent — safe to re-run. Use `run_etl(force=True)` in Python to reload from scratch.

## Data Model

| Table | Rows (sample) | Description |
|-------|--------------|-------------|
| `employees` | 80 | Engineer master table |
| `sessions` | 3,000 | Aggregated coding sessions |
| `api_requests` | 71,227 | Individual Claude API calls |
| `tool_events` | 181,082 | Tool decisions + results |
| `user_prompts` | 20,829 | User prompt events |
| `api_errors` | 873 | API error events |

All tables are indexed on `user_email`, `timestamp`, `session_id`, and `model` for fast filtering.

## Dashboard Pages

| Page | Key Insights |
|------|-------------|
| 🏠 Overview | KPIs, daily trend bar+line, activity heatmap, practice donut |
| 📈 Usage Trends | Weekly trends, model adoption share, session forecasting |
| 💰 Cost & Tokens | Model cost breakdown, cache efficiency, token percentiles |
| 🔧 Tool Analysis | Usage volumes, success rates, practice heatmap |
| 👥 Teams & Users | Practice/level/location drill-down, top engineers |
| 🤖 ML Insights | Cost forecasting, anomaly detection, KMeans segmentation |
| 📊 Statistical Deep Dive | ANOVA, Spearman correlation, engagement scoring |

## ML Components

### Forecasting
- **Polynomial Regression (degree=2)** on daily cost and session count
- 14-day horizon with 95% confidence intervals from residual std
- Anomaly overlay (Z-score based, 7-day rolling window)

### Anomaly Detection
- **Isolation Forest** (`contamination=0.1`) on 6 per-user features
- Daily cost anomaly detection via rolling Z-score (threshold = 2.5σ)

### User Clustering
- **KMeans (k=4)** on StandardScaler-normalised features
- Segments: Power Users / Active Users / Moderate Users / Light Users
- Composite engagement score (frequency × depth × diversity)

### Statistical Tests
- One-way **ANOVA** across engineering practices
- **Spearman correlation** between seniority and cost per session
- Full descriptive statistics (skewness, kurtosis, percentiles)

## Data Quality

`src/validation.py` runs 10 automated checks on every ETL load and produces a weighted quality score (0–100):

- Schema completeness, referential integrity, null rates
- Value range plausibility, duplicate detection, business rules
- Temporal consistency, outlier distribution, volume sanity, session completeness
