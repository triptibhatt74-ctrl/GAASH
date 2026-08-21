"""
Conversation memory. The LLM has no memory of its own — this module is
where persistence and context-bounding actually happen (system prompt
section 3: "Gaash does NOT independently store user history").
"""
import sqlite3
from typing import List, Optional

from config import MAX_RECENT_MESSAGES, MAX_WEEKLY_SUMMARIES
from db import get_conn, run_db


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_message_sync(user_id: int, role: str, content: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversation_messages (user_id, role, content) "
            "VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        conn.commit()
        return cur.lastrowid


async def save_message(user_id: int, role: str, content: str) -> int:
    return await run_db(_save_message_sync, user_id, role, content)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _recent_messages_sync(user_id: int, limit: int) -> List[sqlite3.Row]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversation_messages "
            "WHERE user_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return list(reversed(rows))  # chronological order


async def get_recent_messages(user_id: int, limit: int = MAX_RECENT_MESSAGES):
    return await run_db(_recent_messages_sync, user_id, limit)


def _recent_summaries_sync(user_id: int, limit: int) -> List[sqlite3.Row]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT week_start, week_end, summary_text FROM weekly_summaries "
            "WHERE user_id = ? ORDER BY week_start DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return list(reversed(rows))


async def get_recent_weekly_summaries(user_id: int, limit: int = MAX_WEEKLY_SUMMARIES):
    return await run_db(_recent_summaries_sync, user_id, limit)


def _latest_assessment_snapshot_sync(user_id: int) -> dict:
    """Most recent per-item score for each scale — a compact structured
    history, not the full raw item log."""
    out = {"PHQ-9": [], "GAD-7": [], "PSS-10": []}
    with get_conn() as conn:
        for scale in out:
            rows = conn.execute(
                """
                SELECT item_id, score, MAX(timestamp) as ts
                FROM assessment_records
                WHERE user_id = ? AND assessment_type = ? AND score IS NOT NULL
                GROUP BY item_id
                ORDER BY item_id
                """,
                (user_id, scale),
            ).fetchall()
            out[scale] = [{"item_id": r["item_id"], "score": r["score"]} for r in rows]
    return out


async def get_latest_assessment_snapshot(user_id: int) -> dict:
    return await run_db(_latest_assessment_snapshot_sync, user_id)


# ---------------------------------------------------------------------------
# Context builder — recent messages + weekly summaries + structured history
# ---------------------------------------------------------------------------

async def build_llm_context(
    user_id: int,
    current_message: str,
    preferred_language: Optional[str],
    sleep_hours: Optional[float],
    deepface_emotion: Optional[str],
) -> List[dict]:
    """Builds the bounded message list sent to the LLM. Never sends
    unlimited history — bounded by MAX_RECENT_MESSAGES / MAX_WEEKLY_SUMMARIES."""
    recent = await get_recent_messages(user_id)
    summaries = await get_recent_weekly_summaries(user_id)
    snapshot = await get_latest_assessment_snapshot(user_id)

    context_lines = ["[BACKEND CONTEXT — not user-authored]"]
    if preferred_language:
        context_lines.append(f"preferred_language: {preferred_language}")
    if summaries:
        context_lines.append("weekly_summaries (oldest to newest):")
        for s in summaries:
            context_lines.append(
                f"- {s['week_start']} to {s['week_end']}: {s['summary_text']}"
            )
    if any(snapshot[k] for k in snapshot):
        context_lines.append(f"latest_structured_assessment_snapshot: {snapshot}")
    if sleep_hours is not None:
        context_lines.append(f"sleep_hours_reported_this_turn: {sleep_hours}")
    if deepface_emotion:
        context_lines.append(
            f"optional_visual_emotion_signal (context only, NOT a symptom "
            f"or diagnosis): {deepface_emotion}"
        )

    messages: List[dict] = [{"role": "system", "content": "\n".join(context_lines)}]
    for row in recent:
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": current_message})
    return messages
