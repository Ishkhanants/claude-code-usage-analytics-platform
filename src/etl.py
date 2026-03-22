"""
ETL pipeline: ingest telemetry_logs.jsonl + employees.csv → SQLite.

Design:
  - Stream-parse JSONL line by line to keep memory bounded
  - Batch-insert using executemany for throughput
  - Two-pass approach: (1) raw events → tables, (2) aggregate sessions
  - Idempotent: safe to re-run (drops + recreates tables)
"""

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from db import DB_PATH, create_schema, get_connection

DATA_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 5_000

LEVEL_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5,
               "L6": 6, "L7": 7, "L8": 8, "L9": 9, "L10": 10}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_bool(val) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, str):
        return 1 if val.lower() == "true" else 0
    return 0


# ---------------------------------------------------------------------------
# JSONL streaming parser
# ---------------------------------------------------------------------------

def iter_events(jsonl_path: Path) -> Iterator[dict]:
    """Yield individual telemetry events from log batches."""
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError:
                continue
            for log_event in batch.get("logEvents", []):
                raw_msg = log_event.get("message", "{}")
                try:
                    event = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue
                yield event


# ---------------------------------------------------------------------------
# Employee ingestion
# ---------------------------------------------------------------------------

def ingest_employees(conn: sqlite3.Connection, csv_path: Path) -> int:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            email, full_name, practice, level, location = (
                parts[0], parts[1], parts[2], parts[3], ",".join(parts[4:])
            )
            rows.append((
                email, full_name, practice, level, location,
                LEVEL_ORDER.get(level, 0)
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO employees VALUES (?,?,?,?,?,?)", rows
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Event ingestion (streaming, batched)
# ---------------------------------------------------------------------------

def ingest_events(conn: sqlite3.Connection, jsonl_path: Path) -> dict:
    api_buf, tool_buf, prompt_buf, error_buf = [], [], [], []
    counts = {"api_request": 0, "tool_decision": 0, "tool_result": 0,
              "user_prompt": 0, "api_error": 0, "unknown": 0}

    def flush():
        if api_buf:
            conn.executemany("""
                INSERT INTO api_requests
                  (session_id, user_email, timestamp, model,
                   input_tokens, output_tokens, cache_read_tokens,
                   cache_creation_tokens, total_tokens, cost_usd, duration_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, api_buf)
            api_buf.clear()
        if tool_buf:
            conn.executemany("""
                INSERT INTO tool_events
                  (session_id, user_email, timestamp, event_type, tool_name,
                   decision, source, success, duration_ms, result_size_bytes)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, tool_buf)
            tool_buf.clear()
        if prompt_buf:
            conn.executemany("""
                INSERT INTO user_prompts
                  (session_id, user_email, timestamp, prompt_length)
                VALUES (?,?,?,?)
            """, prompt_buf)
            prompt_buf.clear()
        if error_buf:
            conn.executemany("""
                INSERT INTO api_errors
                  (session_id, user_email, timestamp, error_msg,
                   status_code, model, attempt, duration_ms)
                VALUES (?,?,?,?,?,?,?,?)
            """, error_buf)
            error_buf.clear()
        conn.commit()

    total = 0
    for event in iter_events(jsonl_path):
        body = event.get("body", "")
        attrs = event.get("attributes", {})

        session_id = attrs.get("session.id", "")
        user_email = attrs.get("user.email", "")
        ts = attrs.get("event.timestamp", "")

        if body == "claude_code.api_request":
            inp = _safe_int(attrs.get("input_tokens", 0))
            out = _safe_int(attrs.get("output_tokens", 0))
            cr = _safe_int(attrs.get("cache_read_tokens", 0))
            cc = _safe_int(attrs.get("cache_creation_tokens", 0))
            api_buf.append((
                session_id, user_email, ts,
                attrs.get("model", "unknown"),
                inp, out, cr, cc, inp + out + cr + cc,
                _safe_float(attrs.get("cost_usd", 0)),
                _safe_int(attrs.get("duration_ms", 0)),
            ))
            counts["api_request"] += 1

        elif body == "claude_code.tool_decision":
            tool_buf.append((
                session_id, user_email, ts, "decision",
                attrs.get("tool_name", ""),
                attrs.get("decision", ""),
                attrs.get("source", ""),
                None, 0, None,
            ))
            counts["tool_decision"] += 1

        elif body == "claude_code.tool_result":
            tool_buf.append((
                session_id, user_email, ts, "result",
                attrs.get("tool_name", ""),
                attrs.get("decision_type", ""),
                attrs.get("decision_source", ""),
                _safe_bool(attrs.get("success", "true")),
                _safe_int(attrs.get("duration_ms", 0)),
                _safe_int(attrs.get("tool_result_size_bytes")) if attrs.get("tool_result_size_bytes") else None,
            ))
            counts["tool_result"] += 1

        elif body == "claude_code.user_prompt":
            prompt_buf.append((
                session_id, user_email, ts,
                _safe_int(attrs.get("prompt_length", 0)),
            ))
            counts["user_prompt"] += 1

        elif body == "claude_code.api_error":
            error_buf.append((
                session_id, user_email, ts,
                attrs.get("error", ""),
                attrs.get("status_code", ""),
                attrs.get("model", ""),
                _safe_int(attrs.get("attempt", 1)),
                _safe_int(attrs.get("duration_ms", 0)),
            ))
            counts["api_error"] += 1
        else:
            counts["unknown"] += 1

        total += 1
        if total % BATCH_SIZE == 0:
            flush()
            print(f"  Processed {total:,} events…", flush=True)

    flush()
    counts["total"] = total
    return counts


# ---------------------------------------------------------------------------
# Session aggregation
# ---------------------------------------------------------------------------

def build_sessions(conn: sqlite3.Connection) -> int:
    """
    Aggregate per-session stats from already-ingested event rows.
    """
    print("  Aggregating sessions…", flush=True)

    # Derive sessions from api_requests + user_prompts + tool_events
    conn.executescript("""
        INSERT OR REPLACE INTO sessions
            (session_id, user_email, start_time, end_time, duration_min,
             num_turns, num_api_calls, num_tool_calls, total_cost_usd, total_tokens)
        SELECT
            s.session_id,
            s.user_email,
            MIN(s.ts)                               AS start_time,
            MAX(s.ts)                               AS end_time,
            ROUND(
              (JULIANDAY(MAX(s.ts)) - JULIANDAY(MIN(s.ts))) * 1440
            , 2)                                    AS duration_min,
            COALESCE(p.turns, 0)                    AS num_turns,
            COALESCE(a.api_calls, 0)                AS num_api_calls,
            COALESCE(t.tool_calls, 0)               AS num_tool_calls,
            COALESCE(a.cost, 0)                     AS total_cost_usd,
            COALESCE(a.tokens, 0)                   AS total_tokens
        FROM (
            SELECT session_id, user_email, timestamp AS ts FROM api_requests
            UNION ALL
            SELECT session_id, user_email, timestamp  FROM user_prompts
            UNION ALL
            SELECT session_id, user_email, timestamp  FROM tool_events
        ) s
        LEFT JOIN (
            SELECT session_id, COUNT(*) AS turns FROM user_prompts GROUP BY session_id
        ) p ON p.session_id = s.session_id
        LEFT JOIN (
            SELECT session_id,
                   COUNT(*)        AS api_calls,
                   SUM(cost_usd)   AS cost,
                   SUM(total_tokens) AS tokens
            FROM api_requests GROUP BY session_id
        ) a ON a.session_id = s.session_id
        LEFT JOIN (
            SELECT session_id, COUNT(*) AS tool_calls
            FROM tool_events WHERE event_type = 'decision'
            GROUP BY session_id
        ) t ON t.session_id = s.session_id
        GROUP BY s.session_id, s.user_email;
    """)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    return count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_etl(force: bool = False) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / "telemetry_logs.jsonl"
    csv_path   = DATA_DIR / "employees.csv"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {jsonl_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Employee file not found: {csv_path}")

    conn = get_connection()
    
    # Check if already loaded
    if not force:
        try:
            n = conn.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0]
            if n > 0:
                print(f"Database already loaded ({n:,} API requests). Use force=True to reload.")
                conn.close()
                return
        except Exception:
            pass

    # Drop and recreate for idempotency
    print("Recreating schema…")
    tables = ["api_errors", "user_prompts", "tool_events", "api_requests",
              "sessions", "employees"]
    for t in tables:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()
    create_schema(conn)

    # Disable FK checks during bulk load for performance
    conn.execute("PRAGMA foreign_keys=OFF")
    t0 = time.time()

    print(f"Ingesting employees from {csv_path}…")
    n_emp = ingest_employees(conn, csv_path)
    print(f"  → {n_emp} employees loaded")

    print(f"Ingesting events from {jsonl_path}…")
    counts = ingest_events(conn, jsonl_path)
    print(f"  → {counts}")

    print("Building sessions…")
    n_sess = build_sessions(conn)
    print(f"  → {n_sess} sessions")

    elapsed = time.time() - t0
    print(f"\nETL complete in {elapsed:.1f}s")
    conn.close()


if __name__ == "__main__":
    run_etl(force=True)
