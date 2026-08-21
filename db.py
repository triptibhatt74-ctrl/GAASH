"""
Gaash intelligence-layer database access.

The existing auth API uses raw sqlite3 against a single file (DATABASE =
"gaash.db"), not an ORM. We keep that convention rather than introducing
SQLAlchemy for no reason. sqlite3 is blocking, so every call from the new
async endpoints goes through asyncio.to_thread() (see `run_db`) to avoid
stalling the event loop — this preserves compatibility with the existing
code instead of rewriting it onto an async driver.
"""
import asyncio
import sqlite3
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, TypeVar

from config import DATABASE

T = TypeVar("T")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


async def run_db(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking sqlite3 function in a worker thread so it doesn't
    block the FastAPI event loop."""
    return await asyncio.to_thread(partial(fn, *args, **kwargs))


def init_gaash_tables() -> None:
    """Create the new Gaash tables if they don't already exist. Does not
    touch the existing `users` table."""
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user_ts "
            "ON conversation_messages (user_id, timestamp)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assessment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                assessment_type TEXT NOT NULL CHECK (assessment_type IN ('PHQ-9','GAD-7','PSS-10')),
                item_id INTEGER NOT NULL,
                score INTEGER,
                evidence TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_assess_user_type_ts "
            "ON assessment_records (user_id, assessment_type, timestamp)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS functional_impairments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                area TEXT NOT NULL,
                severity TEXT NOT NULL,
                evidence TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sleep_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                hours REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                risk_category TEXT NOT NULL,
                phq9_total INTEGER,
                gad7_total INTEGER,
                pss10_total INTEGER,
                trajectory TEXT,
                emergency_flag INTEGER NOT NULL DEFAULT 0,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS follow_ups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                scheduled_for TIMESTAMP NOT NULL,
                note TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','completed','cancelled')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS escalation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                trigger_message_id INTEGER,
                counselor_summary TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','reviewed','closed')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
