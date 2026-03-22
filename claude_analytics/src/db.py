"""
Database schema definition and connection management.
Uses SQLite via sqlite3 for zero-dependency storage.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "analytics.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-32000")  # 32 MB cache
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes."""
    conn.executescript("""
        -- ----------------------------------------------------------------
        -- employees: enriched user/employee master table
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS employees (
            email           TEXT PRIMARY KEY,
            full_name       TEXT NOT NULL,
            practice        TEXT NOT NULL,
            level           TEXT NOT NULL,
            location        TEXT NOT NULL,
            level_num       INTEGER NOT NULL  -- numeric for sorting/stats
        );

        -- ----------------------------------------------------------------
        -- sessions: one row per coding session
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            user_email      TEXT NOT NULL,
            start_time      TEXT NOT NULL,
            end_time        TEXT NOT NULL,
            duration_min    REAL,
            num_turns       INTEGER DEFAULT 0,
            num_api_calls   INTEGER DEFAULT 0,
            num_tool_calls  INTEGER DEFAULT 0,
            total_cost_usd  REAL DEFAULT 0,
            total_tokens    INTEGER DEFAULT 0,
            FOREIGN KEY (user_email) REFERENCES employees(email)
        );

        -- ----------------------------------------------------------------
        -- api_requests: individual Claude API calls
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS api_requests (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id              TEXT NOT NULL,
            user_email              TEXT NOT NULL,
            timestamp               TEXT NOT NULL,
            model                   TEXT NOT NULL,
            input_tokens            INTEGER DEFAULT 0,
            output_tokens           INTEGER DEFAULT 0,
            cache_read_tokens       INTEGER DEFAULT 0,
            cache_creation_tokens   INTEGER DEFAULT 0,
            total_tokens            INTEGER DEFAULT 0,
            cost_usd                REAL DEFAULT 0,
            duration_ms             INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id),
            FOREIGN KEY (user_email) REFERENCES employees(email)
        );

        -- ----------------------------------------------------------------
        -- tool_events: tool decisions and results
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS tool_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            user_email      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            event_type      TEXT NOT NULL,  -- 'decision' or 'result'
            tool_name       TEXT NOT NULL,
            decision        TEXT,           -- 'accept' / 'reject'
            source          TEXT,           -- decision source
            success         INTEGER,        -- 1/0 for results
            duration_ms     INTEGER DEFAULT 0,
            result_size_bytes INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        -- ----------------------------------------------------------------
        -- user_prompts: user prompt events
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS user_prompts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            user_email      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            prompt_length   INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        -- ----------------------------------------------------------------
        -- api_errors: API error events
        -- ----------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS api_errors (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            user_email      TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            error_msg       TEXT,
            status_code     TEXT,
            model           TEXT,
            attempt         INTEGER DEFAULT 1,
            duration_ms     INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        -- ----------------------------------------------------------------
        -- Indexes for common query patterns
        -- ----------------------------------------------------------------
        CREATE INDEX IF NOT EXISTS idx_api_req_email     ON api_requests(user_email);
        CREATE INDEX IF NOT EXISTS idx_api_req_ts        ON api_requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_api_req_model     ON api_requests(model);
        CREATE INDEX IF NOT EXISTS idx_api_req_session   ON api_requests(session_id);
        CREATE INDEX IF NOT EXISTS idx_tool_email        ON tool_events(user_email);
        CREATE INDEX IF NOT EXISTS idx_tool_ts           ON tool_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tool_name         ON tool_events(tool_name);
        CREATE INDEX IF NOT EXISTS idx_sessions_email    ON sessions(user_email);
        CREATE INDEX IF NOT EXISTS idx_sessions_start    ON sessions(start_time);
        CREATE INDEX IF NOT EXISTS idx_prompts_email     ON user_prompts(user_email);
        CREATE INDEX IF NOT EXISTS idx_errors_ts         ON api_errors(timestamp);
    """)
    conn.commit()
