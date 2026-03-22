"""
ML module — predictive analytics and anomaly detection.

Components:
  1. Cost / session trend forecasting (polynomial regression + confidence bands)
  2. Isolation Forest anomaly detection on per-user usage vectors
  3. Usage clustering (KMeans) to segment engineers by behaviour
  4. Statistical change-point detection on daily cost series
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analytics import daily_activity, user_engagement_scores, _q


# ---------------------------------------------------------------------------
# 1. Trend forecasting
# ---------------------------------------------------------------------------

def forecast_daily_cost(forecast_days: int = 14) -> pd.DataFrame:
    """
    Fit a polynomial regression on daily cost, return historical +
    forecast with 95% confidence interval.
    """
    df = daily_activity()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["cost_usd"])
    df["day_num"] = (df["date"] - df["date"].min()).dt.days

    X = df["day_num"].values.reshape(-1, 1)
    y = df["cost_usd"].values

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg",  LinearRegression()),
    ])
    model.fit(X, y)

    # Historical fit
    df["predicted"] = model.predict(X)

    # Residual std for CI
    resid_std = np.std(y - df["predicted"].values)

    # Forecast horizon
    last_day = df["day_num"].max()
    future_days = np.arange(last_day + 1, last_day + 1 + forecast_days)
    future_dates = [df["date"].max() + pd.Timedelta(days=int(d - last_day)) for d in future_days]
    future_X = future_days.reshape(-1, 1)
    future_pred = model.predict(future_X)

    hist = df[["date", "cost_usd", "predicted", "day_num"]].copy()
    hist["is_forecast"] = False
    hist["lower_95"] = np.nan
    hist["upper_95"] = np.nan

    fut = pd.DataFrame({
        "date": future_dates,
        "cost_usd": np.nan,
        "predicted": future_pred,
        "day_num": future_days,
        "is_forecast": True,
        "lower_95": future_pred - 1.96 * resid_std,
        "upper_95": future_pred + 1.96 * resid_std,
    })

    combined = pd.concat([hist, fut], ignore_index=True)
    combined["lower_95"] = combined["lower_95"].clip(lower=0)
    return combined


def forecast_daily_sessions(forecast_days: int = 14) -> pd.DataFrame:
    """Same polynomial-regression forecast for daily session count."""
    df = daily_activity()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["sessions"])
    df["day_num"] = (df["date"] - df["date"].min()).dt.days

    X = df["day_num"].values.reshape(-1, 1)
    y = df["sessions"].values

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg",  LinearRegression()),
    ])
    model.fit(X, y)
    df["predicted"] = model.predict(X)
    resid_std = np.std(y - df["predicted"].values)

    last_day = df["day_num"].max()
    future_days = np.arange(last_day + 1, last_day + 1 + forecast_days)
    future_dates = [df["date"].max() + pd.Timedelta(days=int(d - last_day)) for d in future_days]
    future_pred = model.predict(future_days.reshape(-1, 1))

    hist = df[["date", "sessions", "predicted", "day_num"]].copy()
    hist["is_forecast"] = False
    hist["lower_95"] = np.nan
    hist["upper_95"] = np.nan

    fut = pd.DataFrame({
        "date": future_dates,
        "sessions": np.nan,
        "predicted": future_pred,
        "day_num": future_days,
        "is_forecast": True,
        "lower_95": (future_pred - 1.96 * resid_std).clip(0),
        "upper_95": future_pred + 1.96 * resid_std,
    })
    return pd.concat([hist, fut], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Anomaly detection (per-user)
# ---------------------------------------------------------------------------

def detect_user_anomalies() -> pd.DataFrame:
    """
    Flag users whose usage vector is anomalous via Isolation Forest.
    Features: sessions, api_calls, total_cost, avg_turns, error_rate
    """
    df = _q("""
        SELECT
            s.user_email,
            COUNT(DISTINCT s.session_id)        AS sessions,
            SUM(s.num_api_calls)                AS api_calls,
            ROUND(SUM(s.total_cost_usd),4)      AS total_cost,
            ROUND(AVG(s.num_turns),2)           AS avg_turns,
            ROUND(AVG(s.duration_min),2)        AS avg_duration,
            COALESCE(e_cnt.errors, 0)           AS errors
        FROM sessions s
        LEFT JOIN (
            SELECT user_email, COUNT(*) AS errors FROM api_errors GROUP BY user_email
        ) e_cnt ON e_cnt.user_email = s.user_email
        GROUP BY s.user_email
    """)

    from analytics import _q as q2
    from db import get_connection
    import sqlite3
    emp = _q("SELECT email, practice, level, full_name FROM employees")
    df = df.merge(emp, left_on="user_email", right_on="email", how="left")

    feature_cols = ["sessions", "api_calls", "total_cost", "avg_turns", "avg_duration", "errors"]
    X = df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    df["anomaly_label"] = iso.fit_predict(X_scaled)   # -1 = anomaly
    df["anomaly_score"] = iso.score_samples(X_scaled)  # lower = more anomalous
    df["is_anomaly"] = df["anomaly_label"] == -1
    df["anomaly_severity"] = pd.cut(
        -df["anomaly_score"],
        bins=3,
        labels=["Low", "Medium", "High"]
    )
    return df.drop(columns=["email"]).sort_values("anomaly_score")


def detect_daily_cost_anomalies() -> pd.DataFrame:
    """
    Flag days where cost deviates > 2.5 sigma from rolling 7-day mean.
    """
    df = daily_activity()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["rolling_mean"] = df["cost_usd"].rolling(7, min_periods=3).mean()
    df["rolling_std"]  = df["cost_usd"].rolling(7, min_periods=3).std()
    df["z_score"] = (df["cost_usd"] - df["rolling_mean"]) / (df["rolling_std"] + 1e-9)
    df["is_anomaly"] = df["z_score"].abs() > 2.5
    df["anomaly_direction"] = np.where(df["z_score"] > 0, "Spike", "Dip")
    return df


# ---------------------------------------------------------------------------
# 3. User clustering
# ---------------------------------------------------------------------------

def cluster_users(n_clusters: int = 4) -> pd.DataFrame:
    """
    KMeans clustering on per-user behavioural features.
    Returns df with cluster label and centroid distances.
    """
    df = user_engagement_scores()
    feature_cols = ["sessions", "avg_turns", "total_cost", "distinct_tools"]
    X = df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["cluster"] = km.fit_predict(X_scaled)

    # Label clusters by their centroid cost (Power / Active / Moderate / Light)
    cluster_costs = df.groupby("cluster")["total_cost"].mean().sort_values(ascending=False)
    labels = {c: l for c, l in zip(
        cluster_costs.index,
        ["Power Users", "Active Users", "Moderate Users", "Light Users"][:n_clusters]
    )}
    df["cluster_label"] = df["cluster"].map(labels)

    # Distance to centroid (anomaly proxy within cluster)
    centroids = scaler.inverse_transform(km.cluster_centers_)
    df["dist_to_centroid"] = np.linalg.norm(
        X - centroids[df["cluster"].values], axis=1
    ).round(3)

    return df


# ---------------------------------------------------------------------------
# 4. Statistical tests
# ---------------------------------------------------------------------------

def cost_anova_by_practice() -> dict:
    """One-way ANOVA: is mean cost per session significantly different across practices?"""
    df = _q("""
        SELECT e.practice, s.total_cost_usd
        FROM sessions s JOIN employees e ON e.email = s.user_email
        WHERE s.total_cost_usd > 0
    """)
    groups = [g["total_cost_usd"].values for _, g in df.groupby("practice")]
    f_stat, p_val = stats.f_oneway(*groups)
    return {"f_statistic": round(f_stat, 4), "p_value": round(p_val, 6),
            "significant": p_val < 0.05}


def level_cost_correlation() -> dict:
    """Spearman correlation between seniority level and cost per session."""
    df = _q("""
        SELECT e.level_num, s.total_cost_usd
        FROM sessions s JOIN employees e ON e.email = s.user_email
        WHERE s.total_cost_usd > 0
    """)
    rho, p = stats.spearmanr(df["level_num"], df["total_cost_usd"])
    return {"spearman_rho": round(rho, 4), "p_value": round(p, 6),
            "significant": p < 0.05, "direction": "positive" if rho > 0 else "negative"}


def session_duration_stats() -> pd.DataFrame:
    """Full descriptive statistics on session duration by practice."""
    df = _q("""
        SELECT e.practice, s.duration_min
        FROM sessions s JOIN employees e ON e.email = s.user_email
        WHERE s.duration_min > 0 AND s.duration_min < 480
    """)
    result = []
    for prac, grp in df.groupby("practice"):
        d = grp["duration_min"]
        result.append({
            "practice": prac,
            "n": len(d),
            "mean": round(d.mean(), 2),
            "median": round(d.median(), 2),
            "std": round(d.std(), 2),
            "p10": round(d.quantile(0.10), 2),
            "p90": round(d.quantile(0.90), 2),
            "skewness": round(stats.skew(d), 3),
            "kurtosis": round(stats.kurtosis(d), 3),
        })
    return pd.DataFrame(result).sort_values("mean", ascending=False)


def model_usage_trend() -> pd.DataFrame:
    """Weekly model share evolution (% of calls by model per week)."""
    df = _q("""
        SELECT
            strftime('%Y-W%W', timestamp) AS week,
            model,
            COUNT(*)                      AS calls
        FROM api_requests
        GROUP BY week, model
        ORDER BY week
    """)
    total_per_week = df.groupby("week")["calls"].transform("sum")
    df["share_pct"] = (df["calls"] / total_per_week * 100).round(2)
    return df


# ---------------------------------------------------------------------------
# 5. Time-series decomposition (manual STL-style)
# ---------------------------------------------------------------------------

def decompose_daily_cost() -> pd.DataFrame:
    """
    Manual trend + seasonality + residual decomposition of daily cost.

    Method:
      - Trend    : centred 7-day moving average
      - Seasonal : mean deviation from trend by day-of-week (7-period)
      - Residual : actual - trend - seasonal

    Returns a DataFrame with columns:
        date, actual, trend, seasonal, residual, residual_zscore
    """
    from analytics import daily_activity
    df = daily_activity()[["date", "cost_usd"]].copy()
    df["date"]   = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.rename(columns={"cost_usd": "actual"}, inplace=True)

    # Trend: centred 7-day MA
    df["trend"] = df["actual"].rolling(7, center=True, min_periods=4).mean()

    # De-trended series
    df["detrended"] = df["actual"] - df["trend"]

    # Seasonal: average de-trended value by day-of-week
    df["dow"] = df["date"].dt.dayofweek
    seasonal_means = df.groupby("dow")["detrended"].transform("mean")
    df["seasonal"] = seasonal_means

    # Residual
    df["residual"] = df["actual"] - df["trend"] - df["seasonal"]

    # Z-score of residuals
    r_std = df["residual"].std()
    df["residual_zscore"] = (df["residual"] / r_std).round(3)

    # Strength metrics (R² analogues)
    var_total    = df["actual"].var()
    var_trend    = df["trend"].var()
    var_seasonal = df["seasonal"].var()
    var_resid    = df["residual"].var()

    df.attrs["trend_strength"]    = round(1 - var_resid / max(var_trend + var_resid, 1e-9), 4)
    df.attrs["seasonal_strength"] = round(1 - var_resid / max(var_seasonal + var_resid, 1e-9), 4)

    return df.drop(columns=["detrended", "dow"])


# ---------------------------------------------------------------------------
# 6. Model cost efficiency index
# ---------------------------------------------------------------------------

def model_efficiency_index() -> pd.DataFrame:
    """
    Composite efficiency score per model, balancing:
      - Output yield        = output_tokens / total_tokens  (higher = better)
      - Cache leverage      = cache_read / (cache_read + input)  (higher = cheaper)
      - Cost per output tok = cost_usd / output_tokens  (lower = better)
      - Speed               = 1 / avg_duration_s  (higher = faster)

    Each dimension is normalised [0,1] then combined:
    efficiency = 0.30 × output_yield + 0.30 × cache_leverage
               + 0.25 × (1 - cost_norm) + 0.15 × speed_norm
    """
    df = _q("""
        SELECT
            model,
            COUNT(*)                                AS calls,
            SUM(output_tokens)                      AS output_tokens,
            SUM(total_tokens)                       AS total_tokens,
            SUM(cache_read_tokens)                  AS cache_read,
            SUM(input_tokens)                       AS input_tokens,
            ROUND(AVG(cost_usd), 6)                 AS avg_cost,
            ROUND(AVG(output_tokens), 2)            AS avg_output,
            ROUND(AVG(duration_ms) / 1000.0, 3)    AS avg_duration_s
        FROM api_requests
        GROUP BY model
    """)

    df["output_yield"]    = df["output_tokens"] / df["total_tokens"].replace(0, np.nan)
    df["cache_leverage"]  = df["cache_read"] / (df["cache_read"] + df["input_tokens"]).replace(0, np.nan)
    df["cost_per_output"] = df["avg_cost"] / df["avg_output"].replace(0, np.nan)

    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    df["output_yield_n"]    = _norm(df["output_yield"].fillna(0))
    df["cache_leverage_n"]  = _norm(df["cache_leverage"].fillna(0))
    df["cost_norm"]         = _norm(df["cost_per_output"].fillna(0))
    df["speed_n"]           = _norm(1 / df["avg_duration_s"].replace(0, np.nan).fillna(1))

    df["efficiency_score"] = (
        0.30 * df["output_yield_n"]
        + 0.30 * df["cache_leverage_n"]
        + 0.25 * (1 - df["cost_norm"])
        + 0.15 * df["speed_n"]
    ).round(4) * 100

    return df[[
        "model", "calls", "avg_cost", "avg_output", "avg_duration_s",
        "output_yield", "cache_leverage", "cost_per_output",
        "efficiency_score",
    ]].sort_values("efficiency_score", ascending=False)


# ---------------------------------------------------------------------------
# 7. Per-user session time-of-day profile
# ---------------------------------------------------------------------------

def user_activity_profiles() -> pd.DataFrame:
    """
    For each user, characterise their active hours:
      - peak_hour (most common session start hour)
      - night_owl_score  = fraction of sessions after 20:00
      - early_bird_score = fraction of sessions before 08:00
      - weekend_score    = fraction of sessions on Sat/Sun
    """
    df = _q("""
        SELECT
            user_email,
            CAST(strftime('%H', start_time) AS INTEGER) AS hour,
            CAST(strftime('%w', start_time) AS INTEGER) AS dow
        FROM sessions
    """)
    emp = _q("SELECT email, practice, level, full_name FROM employees")
    df = df.merge(emp, left_on="user_email", right_on="email", how="left")

    result = []
    for user, grp in df.groupby("user_email"):
        n = len(grp)
        result.append({
            "user_email": user,
            "full_name":  grp["full_name"].iloc[0],
            "practice":   grp["practice"].iloc[0],
            "level":      grp["level"].iloc[0],
            "sessions":   n,
            "peak_hour":  grp["hour"].mode()[0],
            "night_owl_score":   round((grp["hour"] >= 20).sum() / n * 100, 1),
            "early_bird_score":  round((grp["hour"] < 8).sum()  / n * 100, 1),
            "weekend_score":     round(grp["dow"].isin([0, 6]).sum() / n * 100, 1),
        })
    return pd.DataFrame(result).sort_values("night_owl_score", ascending=False)
