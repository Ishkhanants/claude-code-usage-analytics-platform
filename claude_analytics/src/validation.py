"""
Data validation module — quality checks, integrity assertions, and a
scored quality report that surfaces problems before analytics consume bad data.

Checks performed
----------------
1.  Schema completeness  — required columns present in every table
2.  Null / empty rates   — flag columns above a configurable threshold
3.  Referential integrity— sessions whose user_email has no employee row
4.  Value-range checks   — cost, duration, token counts within plausible bounds
5.  Duplicate detection  — duplicate (session_id, timestamp) pairs
6.  Temporal consistency — events whose timestamp falls outside session window
7.  Business rules       — cost == 0 but tokens > 0 (likely a data defect)
8.  Distribution checks  — Z-score outlier % on key numeric columns
9.  Volume sanity        — day with suspiciously high/low event counts
10. Session completeness — sessions with no user_prompt event

Each check returns a ValidationResult dataclass.  run_all_checks() aggregates
them into a quality score between 0 and 100.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db import get_connection

# Configurable thresholds
NULL_RATE_WARN  = 0.01   # warn if >1 % of rows have nulls in a NOT-NULL column
NULL_RATE_FAIL  = 0.05   # fail  if >5 %
OUTLIER_Z_THRESH = 4.0   # flag rows with |Z| > 4 on key numeric columns
MAX_COST_PER_CALL = 10.0 # $10 per API call — almost certainly wrong
MAX_DURATION_MS = 300_000


@dataclass
class ValidationResult:
    check_name: str
    status: str          # "PASS" | "WARN" | "FAIL"
    message: str
    detail: Any = None   # extra data (DataFrame snippet, number, etc.)
    weight: float = 1.0  # contribution to quality score


def _q(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def _scalar(sql: str) -> Any:
    conn = get_connection()
    val = conn.execute(sql).fetchone()[0]
    conn.close()
    return val


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_schema_completeness() -> ValidationResult:
    """All expected tables exist and have rows."""
    expected = {
        "employees":    5,
        "sessions":     10,
        "api_requests": 11,
        "tool_events":  11,
        "user_prompts": 4,
        "api_errors":   8,
    }
    conn = get_connection()
    issues = []
    for table, min_cols in expected.items():
        try:
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            n    = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if len(info) < min_cols:
                issues.append(f"{table}: only {len(info)} columns (expected ≥{min_cols})")
            if n == 0:
                issues.append(f"{table}: 0 rows")
        except Exception as e:
            issues.append(f"{table}: {e}")
    conn.close()

    if issues:
        return ValidationResult("schema_completeness", "FAIL",
                                f"{len(issues)} schema issues", issues, weight=2.0)
    return ValidationResult("schema_completeness", "PASS",
                            "All tables present and non-empty", weight=2.0)


def check_referential_integrity() -> ValidationResult:
    """Sessions / requests whose user_email has no employee record."""
    orphan_sessions  = _scalar("""
        SELECT COUNT(*) FROM sessions s
        WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.email=s.user_email)
    """)
    orphan_requests  = _scalar("""
        SELECT COUNT(*) FROM api_requests r
        WHERE NOT EXISTS (SELECT 1 FROM employees e WHERE e.email=r.user_email)
    """)
    total_sessions   = _scalar("SELECT COUNT(*) FROM sessions")
    total_requests   = _scalar("SELECT COUNT(*) FROM api_requests")

    issues = []
    if orphan_sessions:
        pct = orphan_sessions / max(total_sessions, 1) * 100
        issues.append(f"{orphan_sessions} orphan sessions ({pct:.1f}%)")
    if orphan_requests:
        pct = orphan_requests / max(total_requests, 1) * 100
        issues.append(f"{orphan_requests} orphan api_requests ({pct:.1f}%)")

    if issues:
        status = "FAIL" if (orphan_sessions + orphan_requests) > 10 else "WARN"
        return ValidationResult("referential_integrity", status,
                                "; ".join(issues), weight=1.5)
    return ValidationResult("referential_integrity", "PASS",
                            "All FK relationships satisfied", weight=1.5)


def check_null_rates() -> ValidationResult:
    """Check null/empty rates on critical columns."""
    checks = [
        ("api_requests", "model",      "model IS NULL OR model=''"),
        ("api_requests", "cost_usd",   "cost_usd IS NULL"),
        ("api_requests", "timestamp",  "timestamp IS NULL OR timestamp=''"),
        ("sessions",     "user_email", "user_email IS NULL OR user_email=''"),
        ("sessions",     "start_time", "start_time IS NULL OR start_time=''"),
        ("tool_events",  "tool_name",  "tool_name IS NULL OR tool_name=''"),
    ]
    conn = get_connection()
    issues_warn, issues_fail = [], []
    for table, col, condition in checks:
        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        nulls = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}").fetchone()[0]
        rate  = nulls / max(total, 1)
        if rate > NULL_RATE_FAIL:
            issues_fail.append(f"{table}.{col}: {rate*100:.1f}% null")
        elif rate > NULL_RATE_WARN:
            issues_warn.append(f"{table}.{col}: {rate*100:.1f}% null")
    conn.close()

    if issues_fail:
        return ValidationResult("null_rates", "FAIL",
                                f"{len(issues_fail)} columns above fail threshold",
                                {"fail": issues_fail, "warn": issues_warn})
    if issues_warn:
        return ValidationResult("null_rates", "WARN",
                                f"{len(issues_warn)} columns above warn threshold",
                                {"warn": issues_warn})
    return ValidationResult("null_rates", "PASS", "All null rates within tolerance")


def check_value_ranges() -> ValidationResult:
    """Plausibility checks on numeric columns."""
    checks = {
        "negative_cost":       "SELECT COUNT(*) FROM api_requests WHERE cost_usd < 0",
        "excessive_cost":     f"SELECT COUNT(*) FROM api_requests WHERE cost_usd > {MAX_COST_PER_CALL}",
        "negative_tokens":     "SELECT COUNT(*) FROM api_requests WHERE total_tokens < 0",
        "excessive_duration": f"SELECT COUNT(*) FROM api_requests WHERE duration_ms > {MAX_DURATION_MS}",
        "negative_duration":   "SELECT COUNT(*) FROM tool_events WHERE duration_ms < 0",
        "future_timestamps":   "SELECT COUNT(*) FROM api_requests WHERE timestamp > '2026-12-31'",
        "very_old_timestamps": "SELECT COUNT(*) FROM api_requests WHERE timestamp < '2020-01-01'",
    }
    conn   = get_connection()
    issues = {}
    for name, sql in checks.items():
        n = conn.execute(sql).fetchone()[0]
        if n > 0:
            issues[name] = n
    conn.close()

    if issues:
        status = "FAIL" if any(v > 100 for v in issues.values()) else "WARN"
        return ValidationResult("value_ranges", status,
                                f"{len(issues)} range violations", issues)
    return ValidationResult("value_ranges", "PASS",
                            "All numeric values within expected ranges")


def check_duplicates() -> ValidationResult:
    """Detect duplicate (user_email, timestamp, model) tuples in api_requests."""
    dup_count = _scalar("""
        SELECT COUNT(*) FROM (
            SELECT user_email, timestamp, model, COUNT(*) AS cnt
            FROM api_requests
            GROUP BY user_email, timestamp, model
            HAVING cnt > 1
        )
    """)
    if dup_count > 0:
        status = "FAIL" if dup_count > 50 else "WARN"
        return ValidationResult("duplicates", status,
                                f"{dup_count} duplicate (email,timestamp,model) groups")
    return ValidationResult("duplicates", "PASS", "No duplicate API request rows detected")


def check_business_rules() -> ValidationResult:
    """Domain-specific rules: e.g. cost=0 but tokens > 1000."""
    zero_cost_heavy = _scalar("""
        SELECT COUNT(*) FROM api_requests
        WHERE cost_usd = 0 AND total_tokens > 1000
    """)
    no_output = _scalar("""
        SELECT COUNT(*) FROM api_requests
        WHERE output_tokens = 0
    """)
    impossible_cache = _scalar("""
        SELECT COUNT(*) FROM api_requests
        WHERE cache_read_tokens > 0 AND input_tokens = 0 AND cache_creation_tokens = 0
    """)

    issues = {}
    if zero_cost_heavy > 0:
        issues["zero_cost_with_tokens"] = zero_cost_heavy
    if no_output > 0:
        issues["zero_output_tokens"] = no_output
    if impossible_cache > 0:
        issues["suspicious_cache_pattern"] = impossible_cache

    if issues:
        status = "WARN" if max(issues.values()) < 500 else "FAIL"
        return ValidationResult("business_rules", status,
                                f"{len(issues)} business rule violations", issues)
    return ValidationResult("business_rules", "PASS",
                            "All business rules satisfied")


def check_temporal_consistency() -> ValidationResult:
    """Events whose timestamp is before session start or after session end."""
    out_of_window = _scalar("""
        SELECT COUNT(*) FROM api_requests r
        JOIN sessions s ON s.session_id = r.session_id
        WHERE r.timestamp < s.start_time OR r.timestamp > s.end_time
    """)
    total = _scalar("SELECT COUNT(*) FROM api_requests")
    pct   = out_of_window / max(total, 1) * 100

    if pct > 5:
        return ValidationResult("temporal_consistency", "FAIL",
                                f"{out_of_window} events ({pct:.1f}%) outside session window")
    if pct > 1:
        return ValidationResult("temporal_consistency", "WARN",
                                f"{out_of_window} events ({pct:.1f}%) outside session window")
    return ValidationResult("temporal_consistency", "PASS",
                            f"Only {pct:.2f}% events outside session window")


def check_outlier_distribution() -> ValidationResult:
    """Z-score outlier % on cost_usd and total_tokens."""
    df = _q("SELECT cost_usd, total_tokens FROM api_requests WHERE total_tokens > 0")

    results = {}
    for col in ["cost_usd", "total_tokens"]:
        arr  = df[col].values.astype(float)
        mean = arr.mean()
        std  = arr.std()
        if std > 0:
            z    = np.abs((arr - mean) / std)
            pct  = (z > OUTLIER_Z_THRESH).mean() * 100
            results[col] = round(pct, 2)

    extreme = {k: v for k, v in results.items() if v > 5}
    if extreme:
        return ValidationResult("outlier_distribution", "WARN",
                                f"Extreme outlier rate >5% on: {list(extreme.keys())}",
                                results)
    return ValidationResult("outlier_distribution", "PASS",
                            f"Outlier rates: {results}")


def check_daily_volume_sanity() -> ValidationResult:
    """Flag days with suspiciously low or high event counts (>3 IQR from median)."""
    df = _q("""
        SELECT DATE(timestamp) AS day, COUNT(*) AS events
        FROM api_requests
        GROUP BY day
    """)
    q1, q3 = df["events"].quantile(0.25), df["events"].quantile(0.75)
    iqr     = q3 - q1
    lo, hi  = q1 - 3 * iqr, q3 + 3 * iqr

    bad = df[(df["events"] < lo) | (df["events"] > hi)]
    if len(bad) > 0:
        return ValidationResult("daily_volume_sanity", "WARN",
                                f"{len(bad)} days with abnormal event counts",
                                bad.to_dict("records"))
    return ValidationResult("daily_volume_sanity", "PASS",
                            "Daily event volumes within expected range")


def check_session_completeness() -> ValidationResult:
    """Sessions that have no user_prompt event (headless / broken sessions)."""
    no_prompt = _scalar("""
        SELECT COUNT(*) FROM sessions s
        WHERE NOT EXISTS (
            SELECT 1 FROM user_prompts p WHERE p.session_id = s.session_id
        )
    """)
    total   = _scalar("SELECT COUNT(*) FROM sessions")
    pct     = no_prompt / max(total, 1) * 100

    if pct > 10:
        return ValidationResult("session_completeness", "WARN",
                                f"{no_prompt} sessions ({pct:.1f}%) have no user_prompt event")
    return ValidationResult("session_completeness", "PASS",
                            f"{pct:.1f}% sessions missing user_prompt (within tolerance)")


# ---------------------------------------------------------------------------
# Aggregate runner + quality score
# ---------------------------------------------------------------------------

def run_all_checks() -> dict:
    """
    Run every check, compute a weighted quality score [0..100], return report.
    """
    checks = [
        check_schema_completeness,
        check_referential_integrity,
        check_null_rates,
        check_value_ranges,
        check_duplicates,
        check_business_rules,
        check_temporal_consistency,
        check_outlier_distribution,
        check_daily_volume_sanity,
        check_session_completeness,
    ]

    results = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as e:
            results.append(ValidationResult(fn.__name__, "FAIL",
                                            f"Check raised exception: {e}"))

    # Score: PASS=1, WARN=0.5, FAIL=0
    total_weight = sum(r.weight for r in results)
    earned = sum(
        r.weight * (1.0 if r.status == "PASS" else 0.5 if r.status == "WARN" else 0.0)
        for r in results
    )
    quality_score = round(earned / total_weight * 100, 1)

    summary = {
        "quality_score": quality_score,
        "total_checks": len(results),
        "passed":  sum(1 for r in results if r.status == "PASS"),
        "warnings": sum(1 for r in results if r.status == "WARN"),
        "failed":  sum(1 for r in results if r.status == "FAIL"),
        "results": results,
    }
    return summary


def quality_summary_df(report: dict) -> pd.DataFrame:
    """Convert ValidationResult list to a display-ready DataFrame."""
    rows = []
    for r in report["results"]:
        rows.append({
            "Check": r.check_name.replace("_", " ").title(),
            "Status": r.status,
            "Message": r.message,
            "Weight": r.weight,
        })
    return pd.DataFrame(rows)
