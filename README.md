# claude-code-usage-analytics-platform
End-to-end analytics platform, that processes telemetry data from Claude Code sessions, transforming raw event streams into actionable insights regarding developer patterns and user behavior through an interactive dashboard.

# Claude Code Analytics Platform

End-to-end analytics platform for Claude Code developer telemetry — from raw JSONL event streams to an interactive Streamlit dashboard with ML-powered insights.

## Architecture

```
claude_analytics/
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
│   └── dashboard.py               # Streamlit multi-page dashboard
├── generate_fake_data.py          # Synthetic data generator
└── requirements.txt
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

## Data Model

| Table | Rows (sample) | Description |
|-------|--------------|-------------|
| `employees` | 80 | Engineer master table |
| `sessions` | 3,000 | Aggregated coding sessions |
| `api_requests` | 71,227 | Individual Claude API calls |
| `tool_events` | 181,082 | Tool decisions + results |
| `user_prompts` | 20,829 | User prompt events |
| `api_errors` | 873 | API error events |

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
