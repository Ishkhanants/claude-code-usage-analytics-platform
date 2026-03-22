"""
Cohort & retention analysis module.

Analyses:
  1.  Weekly engineer retention  — cohort heatmap (classic SaaS metric)
  2.  Session funnel             — prompt → api_call → tool_use → success
  3.  Time-to-first-use          — days from engineer join to first session
  4.  Session cadence            — distribution of inter-session gaps per user
  5.  Power user trajectory      — rolling 7-day session count per user segment
  6.  Tool adoption curves       — cumulative distinct-tool count over time
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analytics import _q


# ---------------------------------------------------------------------------
# 1. Weekly Engineer Retention Cohort
# ---------------------------------------------------------------------------

def weekly_retention_cohort() -> pd.DataFrame:
    """
    Classic cohort retention matrix.

    Cohort = the ISO week of a user's FIRST session.
    Retention[week N] = % of cohort users still active N weeks later.

    Returns a wide DataFrame (cohorts × weeks 0–8) with retention percentages.
    """
    df = _q("""
        SELECT
            user_email,
            strftime('%Y-W%W', start_time) AS week
        FROM sessions
        GROUP BY user_email, strftime('%Y-W%W', start_time)
    """)

    # First week per user = cohort
    first_week = df.groupby("user_email")["week"].min().rename("cohort_week")
    df = df.join(first_week, on="user_email")

    # All unique weeks sorted
    all_weeks = sorted(df["week"].unique())
    week_index = {w: i for i, w in enumerate(all_weeks)}

    df["week_num"]   = df["week"].map(week_index)
    df["cohort_num"] = df["cohort_week"].map(week_index)
    df["period"]     = df["week_num"] - df["cohort_num"]

    # Cohort sizes (users who started in that cohort week)
    cohort_sizes = df[df["period"] == 0].groupby("cohort_week")["user_email"].nunique()

    # Active users per cohort × period
    matrix = (
        df[df["period"] >= 0]
        .groupby(["cohort_week", "period"])["user_email"]
        .nunique()
        .unstack(fill_value=0)
    )

    # Convert to retention %
    for col in matrix.columns:
        matrix[col] = (matrix[col] / cohort_sizes * 100).round(1)

    # Keep only periods 0–8
    keep = [c for c in range(9) if c in matrix.columns]
    matrix = matrix[keep]
    matrix.columns = [f"Week +{c}" for c in keep]
    matrix.index.name = "Cohort"
    return matrix


# ---------------------------------------------------------------------------
# 2. Session Funnel
# ---------------------------------------------------------------------------

def session_funnel() -> pd.DataFrame:
    """
    Funnel: sessions that had at least one of each event type.

    Stage 1: Sessions with ≥1 user_prompt
    Stage 2: Sessions with ≥1 api_request
    Stage 3: Sessions with ≥1 tool (accepted)
    Stage 4: Sessions with ≥1 successful tool result
    Stage 5: Sessions with ≥3 turns (multi-turn engagement)
    """
    rows = []

    total = _q("SELECT COUNT(*) AS n FROM sessions").iloc[0]["n"]
    rows.append({"stage": "1 · Session started", "count": total, "pct": 100.0})

    s1 = _q("""
        SELECT COUNT(DISTINCT session_id) AS n FROM user_prompts
    """).iloc[0]["n"]
    rows.append({"stage": "2 · Has user prompt", "count": s1,
                 "pct": round(s1 / total * 100, 1)})

    s2 = _q("""
        SELECT COUNT(DISTINCT session_id) AS n FROM api_requests
    """).iloc[0]["n"]
    rows.append({"stage": "3 · Has API call", "count": s2,
                 "pct": round(s2 / total * 100, 1)})

    s3 = _q("""
        SELECT COUNT(DISTINCT session_id) AS n FROM tool_events
        WHERE event_type='decision' AND decision='accept'
    """).iloc[0]["n"]
    rows.append({"stage": "4 · Tool accepted", "count": s3,
                 "pct": round(s3 / total * 100, 1)})

    s4 = _q("""
        SELECT COUNT(DISTINCT session_id) AS n FROM tool_events
        WHERE event_type='result' AND success=1
    """).iloc[0]["n"]
    rows.append({"stage": "5 · Tool succeeded", "count": s4,
                 "pct": round(s4 / total * 100, 1)})

    s5 = _q("""
        SELECT COUNT(*) AS n FROM sessions WHERE num_turns >= 3
    """).iloc[0]["n"]
    rows.append({"stage": "6 · Multi-turn (≥3)", "count": s5,
                 "pct": round(s5 / total * 100, 1)})

    df = pd.DataFrame(rows)
    df["drop_off_pct"] = (
        df["pct"].shift(1).fillna(100) - df["pct"]
    ).round(1)
    return df


# ---------------------------------------------------------------------------
# 3. Inter-session gap distribution
# ---------------------------------------------------------------------------

def inter_session_gaps() -> pd.DataFrame:
    """
    For each user, compute the gap in hours between consecutive sessions.
    Returns one row per gap, with user metadata for segmentation.
    """
    df = _q("""
        SELECT s.user_email, s.start_time, e.practice, e.level
        FROM sessions s
        JOIN employees e ON e.email = s.user_email
        ORDER BY s.user_email, s.start_time
    """)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["prev_time"]  = df.groupby("user_email")["start_time"].shift(1)
    df = df.dropna(subset=["prev_time"])
    df["gap_hours"]  = (df["start_time"] - df["prev_time"]).dt.total_seconds() / 3600
    # Remove gaps > 30 days (likely separate use bursts, not regular cadence)
    df = df[df["gap_hours"] < 720]
    return df[["user_email", "practice", "level", "gap_hours"]]


# ---------------------------------------------------------------------------
# 4. Rolling 7-day session volume per cluster
# ---------------------------------------------------------------------------

def rolling_sessions_by_cluster() -> pd.DataFrame:
    """
    Weekly rolling session count grouped by KMeans cluster label.
    Requires ml.cluster_users() — imported lazily to avoid circular deps.
    """
    import ml as mlmod
    clusters = mlmod.cluster_users()[["user_email", "cluster_label"]]

    df = _q("""
        SELECT user_email, DATE(start_time) AS date, COUNT(*) AS sessions
        FROM sessions
        GROUP BY user_email, DATE(start_time)
    """)
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(clusters, on="user_email", how="left")
    df["cluster_label"] = df["cluster_label"].fillna("Unknown")

    agg = (
        df.groupby(["date", "cluster_label"])["sessions"]
        .sum()
        .reset_index()
    )
    agg = agg.sort_values("date")
    # 7-day rolling sum per cluster
    agg["rolling_7d"] = (
        agg.groupby("cluster_label")["sessions"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )
    return agg


# ---------------------------------------------------------------------------
# 5. Cumulative tool adoption
# ---------------------------------------------------------------------------

def tool_adoption_over_time() -> pd.DataFrame:
    """
    For each calendar day, the cumulative number of distinct tools
    ever used (globally and per practice).
    """
    df = _q("""
        SELECT
            DATE(t.timestamp)   AS date,
            t.tool_name,
            e.practice
        FROM tool_events t
        JOIN employees e ON e.email = t.user_email
        WHERE t.event_type = 'decision' AND t.decision = 'accept'
        GROUP BY date, t.tool_name, e.practice
        ORDER BY date
    """)
    df["date"] = pd.to_datetime(df["date"])

    # Global cumulative unique tools
    global_tools = (
        df.sort_values("date")
        .groupby("date")["tool_name"]
        .apply(set)
        .reset_index()
    )
    seen = set()
    counts = []
    for _, row in global_tools.iterrows():
        seen |= row["tool_name"]
        counts.append(len(seen))
    global_tools["cumulative_tools"] = counts
    global_tools["practice"] = "All"
    global_tools = global_tools[["date", "practice", "cumulative_tools"]]

    # Per-practice
    practice_rows = []
    for prac, grp in df.groupby("practice"):
        grp = grp.sort_values("date")
        day_sets = grp.groupby("date")["tool_name"].apply(set).reset_index()
        seen_p = set()
        for _, row in day_sets.iterrows():
            seen_p |= row["tool_name"]
            practice_rows.append({
                "date": row["date"],
                "practice": prac,
                "cumulative_tools": len(seen_p),
            })

    return pd.concat(
        [global_tools, pd.DataFrame(practice_rows)], ignore_index=True
    ).sort_values(["practice", "date"])


# ---------------------------------------------------------------------------
# 6. Daily new vs returning engineer activity
# ---------------------------------------------------------------------------

def new_vs_returning_daily() -> pd.DataFrame:
    """
    Each day: how many engineers are first-timers vs returning.
    """
    df = _q("""
        SELECT user_email, DATE(start_time) AS date
        FROM sessions
        GROUP BY user_email, DATE(start_time)
        ORDER BY user_email, date
    """)
    df["date"] = pd.to_datetime(df["date"])
    first_day = df.groupby("user_email")["date"].min().rename("first_date")
    df = df.join(first_day, on="user_email")
    df["type"] = np.where(df["date"] == df["first_date"], "New", "Returning")

    out = (
        df.groupby(["date", "type"])["user_email"]
        .nunique()
        .reset_index(name="engineers")
    )
    return out
