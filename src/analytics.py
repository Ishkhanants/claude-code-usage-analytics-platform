"""
Analytics module — all insights are computed via SQL + pandas.

Functions return DataFrames ready for Plotly / Streamlit rendering.
"""

import sqlite3
from functools import lru_cache
from typing import Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _q(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a query and return a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Overview / KPIs
# ---------------------------------------------------------------------------

def kpi_summary() -> dict:
    """Top-level KPIs for the overview page."""
    sql = """
    SELECT
        (SELECT COUNT(DISTINCT user_email) FROM sessions)            AS total_users,
        (SELECT COUNT(*)                   FROM sessions)            AS total_sessions,
        (SELECT COUNT(*)                   FROM api_requests)        AS total_api_calls,
        (SELECT ROUND(SUM(cost_usd),2)     FROM api_requests)        AS total_cost_usd,
        (SELECT SUM(total_tokens)          FROM api_requests)        AS total_tokens,
        (SELECT COUNT(*)                   FROM api_errors)          AS total_errors,
        (SELECT ROUND(AVG(num_api_calls),1)FROM sessions)            AS avg_api_per_session,
        (SELECT ROUND(AVG(duration_min),1) FROM sessions
         WHERE duration_min < 480)                                   AS avg_session_min
    """
    row = dict(_q(sql).iloc[0])
    # Error rate
    row["error_rate_pct"] = round(
        row["total_errors"] / max(row["total_api_calls"], 1) * 100, 2
    )
    return row


# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------

def daily_activity() -> pd.DataFrame:
    """Daily sessions, API calls, cost, tokens."""
    return _q("""
        SELECT
            DATE(start_time)            AS date,
            COUNT(*)                    AS sessions,
            SUM(num_api_calls)          AS api_calls,
            ROUND(SUM(total_cost_usd),4) AS cost_usd,
            SUM(total_tokens)           AS tokens,
            COUNT(DISTINCT user_email)  AS active_users
        FROM sessions
        GROUP BY 1
        ORDER BY 1
    """)


def hourly_heatmap() -> pd.DataFrame:
    """Sessions by day-of-week × hour for heatmap."""
    return _q("""
        SELECT
            CAST(strftime('%w', start_time) AS INTEGER) AS dow,
            CAST(strftime('%H', start_time) AS INTEGER) AS hour,
            COUNT(*)                                    AS sessions
        FROM sessions
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def weekly_trend() -> pd.DataFrame:
    """Weekly aggregated metrics for trend analysis."""
    return _q("""
        SELECT
            strftime('%Y-W%W', start_time)  AS week,
            COUNT(*)                         AS sessions,
            COUNT(DISTINCT user_email)       AS active_users,
            ROUND(SUM(total_cost_usd),2)     AS cost_usd,
            ROUND(AVG(total_cost_usd),4)     AS avg_cost_per_session
        FROM sessions
        GROUP BY 1
        ORDER BY 1
    """)


# ---------------------------------------------------------------------------
# Token & Cost Analytics
# ---------------------------------------------------------------------------

def cost_by_model() -> pd.DataFrame:
    return _q("""
        SELECT
            model,
            COUNT(*)                        AS calls,
            ROUND(SUM(cost_usd),2)          AS total_cost,
            ROUND(AVG(cost_usd),4)          AS avg_cost,
            SUM(input_tokens)               AS input_tokens,
            SUM(output_tokens)              AS output_tokens,
            SUM(cache_read_tokens)          AS cache_read,
            SUM(cache_creation_tokens)      AS cache_creation,
            ROUND(AVG(duration_ms)/1000.0,2) AS avg_duration_s
        FROM api_requests
        GROUP BY model
        ORDER BY total_cost DESC
    """)


def cost_by_practice() -> pd.DataFrame:
    return _q("""
        SELECT
            e.practice,
            COUNT(DISTINCT r.user_email)    AS engineers,
            COUNT(*)                        AS api_calls,
            ROUND(SUM(r.cost_usd),2)        AS total_cost,
            ROUND(AVG(r.cost_usd),4)        AS avg_cost_per_call,
            SUM(r.total_tokens)             AS total_tokens
        FROM api_requests r
        JOIN employees e ON e.email = r.user_email
        GROUP BY e.practice
        ORDER BY total_cost DESC
    """)


def cost_by_level() -> pd.DataFrame:
    return _q("""
        SELECT
            e.level,
            e.level_num,
            COUNT(DISTINCT r.user_email)    AS engineers,
            COUNT(*)                        AS api_calls,
            ROUND(SUM(r.cost_usd),2)        AS total_cost,
            ROUND(SUM(r.cost_usd)/COUNT(DISTINCT r.user_email),2) AS cost_per_engineer
        FROM api_requests r
        JOIN employees e ON e.email = r.user_email
        GROUP BY e.level, e.level_num
        ORDER BY e.level_num
    """)


def token_distribution() -> pd.DataFrame:
    """Percentile distribution of token counts per request."""
    df = _q("""
        SELECT input_tokens, output_tokens, cache_read_tokens,
               cache_creation_tokens, total_tokens, cost_usd
        FROM api_requests
        WHERE total_tokens > 0
    """)
    stats = {}
    for col in ["input_tokens", "output_tokens", "total_tokens", "cost_usd"]:
        stats[col] = {
            "p25": df[col].quantile(0.25),
            "p50": df[col].quantile(0.50),
            "p75": df[col].quantile(0.75),
            "p90": df[col].quantile(0.90),
            "p99": df[col].quantile(0.99),
            "mean": df[col].mean(),
            "std":  df[col].std(),
        }
    return pd.DataFrame(stats).T.round(2)


def cache_efficiency() -> pd.DataFrame:
    """Cache hit rate by model."""
    return _q("""
        SELECT
            model,
            COUNT(*) AS calls,
            SUM(cache_read_tokens)      AS cache_hits,
            SUM(cache_creation_tokens)  AS cache_writes,
            SUM(input_tokens)           AS fresh_tokens,
            ROUND(
              100.0 * SUM(cache_read_tokens) /
              NULLIF(SUM(cache_read_tokens) + SUM(input_tokens), 0)
            , 1) AS cache_hit_rate_pct
        FROM api_requests
        GROUP BY model
        ORDER BY cache_hit_rate_pct DESC
    """)


# ---------------------------------------------------------------------------
# Tool Analytics
# ---------------------------------------------------------------------------

def tool_usage_summary() -> pd.DataFrame:
    return _q("""
        SELECT
            tool_name,
            COUNT(*) FILTER (WHERE event_type='decision')       AS decisions,
            COUNT(*) FILTER (WHERE decision='accept' AND event_type='decision') AS accepted,
            COUNT(*) FILTER (WHERE decision='reject' AND event_type='decision') AS rejected,
            COUNT(*) FILTER (WHERE event_type='result')         AS executions,
            SUM(CASE WHEN event_type='result' AND success=1 THEN 1 ELSE 0 END) AS successes,
            ROUND(
              100.0 * SUM(CASE WHEN event_type='result' AND success=1 THEN 1 ELSE 0 END) /
              NULLIF(COUNT(*) FILTER (WHERE event_type='result'), 1)
            , 1) AS success_rate_pct,
            ROUND(AVG(CASE WHEN event_type='result' THEN duration_ms END)/1000.0, 2) AS avg_duration_s
        FROM tool_events
        GROUP BY tool_name
        ORDER BY decisions DESC
    """)


def tool_usage_by_practice() -> pd.DataFrame:
    return _q("""
        SELECT
            e.practice,
            t.tool_name,
            COUNT(*) AS uses
        FROM tool_events t
        JOIN employees e ON e.email = t.user_email
        WHERE t.event_type = 'decision' AND t.decision = 'accept'
        GROUP BY e.practice, t.tool_name
        ORDER BY e.practice, uses DESC
    """)


def tool_decision_sources() -> pd.DataFrame:
    return _q("""
        SELECT source, COUNT(*) AS count
        FROM tool_events
        WHERE event_type = 'decision'
        GROUP BY source
        ORDER BY count DESC
    """)


# ---------------------------------------------------------------------------
# User / Team Analytics
# ---------------------------------------------------------------------------

def top_users(limit: int = 20) -> pd.DataFrame:
    return _q(f"""
        SELECT
            r.user_email,
            e.full_name,
            e.practice,
            e.level,
            e.location,
            COUNT(DISTINCT s.session_id)        AS sessions,
            COUNT(r.id)                          AS api_calls,
            ROUND(SUM(r.cost_usd),2)             AS total_cost,
            ROUND(AVG(s.num_turns),1)            AS avg_turns,
            ROUND(AVG(s.duration_min),1)         AS avg_session_min
        FROM api_requests r
        JOIN employees e  ON e.email = r.user_email
        JOIN sessions s   ON s.user_email = r.user_email
        GROUP BY r.user_email
        ORDER BY total_cost DESC
        LIMIT {limit}
    """)


def practice_summary() -> pd.DataFrame:
    return _q("""
        SELECT
            e.practice,
            COUNT(DISTINCT e.email)              AS headcount,
            COUNT(DISTINCT s.session_id)         AS sessions,
            ROUND(SUM(s.total_cost_usd),2)       AS total_cost,
            ROUND(AVG(s.num_turns),1)            AS avg_turns_per_session,
            ROUND(AVG(s.duration_min),1)         AS avg_session_min,
            ROUND(SUM(s.total_cost_usd)/COUNT(DISTINCT e.email),2) AS cost_per_engineer
        FROM employees e
        LEFT JOIN sessions s ON s.user_email = e.email
        GROUP BY e.practice
        ORDER BY total_cost DESC
    """)


def location_summary() -> pd.DataFrame:
    return _q("""
        SELECT
            e.location,
            COUNT(DISTINCT e.email)             AS engineers,
            COUNT(DISTINCT s.session_id)        AS sessions,
            ROUND(SUM(s.total_cost_usd),2)      AS total_cost,
            ROUND(AVG(s.total_cost_usd),4)      AS avg_cost_per_session
        FROM employees e
        LEFT JOIN sessions s ON s.user_email = e.email
        GROUP BY e.location
        ORDER BY total_cost DESC
    """)


def session_depth_distribution() -> pd.DataFrame:
    """Distribution of session complexity (turns, API calls)."""
    return _q("""
        SELECT
            num_turns,
            num_api_calls,
            duration_min,
            total_cost_usd
        FROM sessions
        WHERE duration_min < 480 AND num_turns > 0
    """)


# ---------------------------------------------------------------------------
# Error Analytics
# ---------------------------------------------------------------------------

def error_summary() -> pd.DataFrame:
    return _q("""
        SELECT
            error_msg,
            status_code,
            COUNT(*) AS count,
            ROUND(100.0*COUNT()/SUM(COUNT(*)) OVER (), 1) AS pct
        FROM api_errors
        GROUP BY error_msg, status_code
        ORDER BY count DESC
    """)


def error_trend() -> pd.DataFrame:
    return _q("""
        SELECT
            DATE(timestamp)     AS date,
            COUNT(*)            AS errors,
            status_code
        FROM api_errors
        GROUP BY 1, 3
        ORDER BY 1
    """)


def error_by_model() -> pd.DataFrame:
    return _q("""
        SELECT
            model,
            COUNT(*) AS errors,
            GROUP_CONCAT(DISTINCT status_code) AS codes
        FROM api_errors
        WHERE model != ''
        GROUP BY model
        ORDER BY errors DESC
    """)


# ---------------------------------------------------------------------------
# Prompt Analytics
# ---------------------------------------------------------------------------

def prompt_length_distribution() -> pd.DataFrame:
    return _q("""
        SELECT
            prompt_length,
            user_email,
            timestamp
        FROM user_prompts
        WHERE prompt_length > 0
    """)


def prompt_stats_by_practice() -> pd.DataFrame:
    return _q("""
        SELECT
            e.practice,
            COUNT(p.id)                             AS prompts,
            ROUND(AVG(p.prompt_length),0)           AS avg_length,
            ROUND(MIN(p.prompt_length),0)           AS min_length,
            MAX(p.prompt_length)                    AS max_length,
            CAST(ROUND(
              (SELECT prompt_length FROM user_prompts WHERE user_email=p.user_email
               ORDER BY ABS(CAST(SUBSTR(CAST(ROW_NUMBER() OVER(ORDER BY id) AS TEXT),1) AS REAL) - 0.5)
               LIMIT 1)
            , 0) AS INTEGER)                        AS median_length_approx
        FROM user_prompts p
        JOIN employees e ON e.email = p.user_email
        GROUP BY e.practice
        ORDER BY avg_length DESC
    """)


# ---------------------------------------------------------------------------
# Advanced: user engagement scoring
# ---------------------------------------------------------------------------

def user_engagement_scores() -> pd.DataFrame:
    """
    Composite engagement score per user:
      - frequency (sessions / max_sessions)
      - depth (avg_turns / max_avg_turns)
      - diversity (distinct tools / max tools)
    Normalised to [0, 100].
    """
    df = _q("""
        SELECT
            s.user_email,
            e.practice,
            e.level,
            e.level_num,
            COUNT(DISTINCT s.session_id)            AS sessions,
            ROUND(AVG(s.num_turns), 2)              AS avg_turns,
            ROUND(SUM(s.total_cost_usd), 2)         AS total_cost,
            COUNT(DISTINCT t.tool_name)             AS distinct_tools
        FROM sessions s
        JOIN employees e ON e.email = s.user_email
        LEFT JOIN tool_events t ON t.session_id = s.session_id AND t.event_type = 'decision'
        GROUP BY s.user_email
    """)
    for col in ["sessions", "avg_turns", "distinct_tools"]:
        mx = df[col].max()
        df[f"{col}_n"] = df[col] / mx if mx > 0 else 0

    df["engagement_score"] = (
        0.40 * df["sessions_n"] +
        0.35 * df["avg_turns_n"] +
        0.25 * df["distinct_tools_n"]
    ) * 100
    df["engagement_score"] = df["engagement_score"].round(1)
    return df.sort_values("engagement_score", ascending=False)


# ---------------------------------------------------------------------------
# Per-user drill-down
# ---------------------------------------------------------------------------

def user_drilldown(email: str) -> dict:
    """
    Full profile for a single engineer: metadata, session stats, model mix,
    tool usage, daily activity, and prompt stats.
    """
    emp = _q("SELECT * FROM employees WHERE email = ?", (email,))
    if emp.empty:
        return {}

    sessions = _q("""
        SELECT session_id, start_time, end_time, duration_min,
               num_turns, num_api_calls, num_tool_calls, total_cost_usd, total_tokens
        FROM sessions WHERE user_email = ?
        ORDER BY start_time DESC
    """, (email,))

    model_mix = _q("""
        SELECT model, COUNT(*) AS calls,
               ROUND(SUM(cost_usd),4) AS cost,
               SUM(output_tokens) AS output_tokens
        FROM api_requests WHERE user_email = ?
        GROUP BY model ORDER BY calls DESC
    """, (email,))

    tools = _q("""
        SELECT tool_name, COUNT(*) AS uses,
               ROUND(100.0*SUM(success)/NULLIF(COUNT(*),0),1) AS success_rate
        FROM tool_events WHERE user_email = ? AND event_type='result'
        GROUP BY tool_name ORDER BY uses DESC
    """, (email,))

    daily = _q("""
        SELECT DATE(start_time) AS date, COUNT(*) AS sessions,
               ROUND(SUM(total_cost_usd),4) AS cost
        FROM sessions WHERE user_email = ?
        GROUP BY 1 ORDER BY 1
    """, (email,))

    errors = _q("""
        SELECT error_msg, status_code, COUNT(*) AS count
        FROM api_errors WHERE user_email = ?
        GROUP BY error_msg, status_code ORDER BY count DESC
    """, (email,))

    return {
        "employee": emp.iloc[0].to_dict(),
        "sessions": sessions,
        "model_mix": model_mix,
        "tools": tools,
        "daily": daily,
        "errors": errors,
    }


def all_user_emails() -> list[str]:
    return _q("SELECT email FROM employees ORDER BY email")["email"].tolist()
