"""
Deterministic scoring. The LLM only extracts evidence + per-item scores
when frequency was explicit; everything about official totals, valid-item
requirements, and PSS-10 reverse scoring happens here, in plain code, so
it's auditable and doesn't depend on model behavior.

PHQ-9: items 1-9, 0-3 each, total 0-27.
GAD-7: items 1-7, 0-3 each, total 0-21.
PSS-10: items 1-10, 0-4 each, total 0-40, with items 4, 5, 7, 8 reverse
        scored (standard PSS-10 convention: reverse_score = 4 - raw_score).
"""
from typing import Dict, List, Optional

from db import get_conn, run_db
from schemas import FunctionalImpairment, SymptomItem, validate_symptom_items

PSS10_REVERSE_ITEMS = {4, 5, 7, 8}

_SCALE_ITEM_COUNT = {"PHQ-9": 9, "GAD-7": 7, "PSS-10": 10}


def _pss10_transform(item_id: int, score: int) -> int:
    if item_id in PSS10_REVERSE_ITEMS:
        return 4 - score
    return score


def compute_total(scale: str, item_scores: Dict[int, int]) -> Optional[int]:
    """Returns the official total, or None if not enough items have valid
    scores to consider the questionnaire complete for a formal total.
    Partial evidence is still stored per-item regardless of this."""
    required = set(range(1, _SCALE_ITEM_COUNT[scale] + 1))
    if not required.issubset(item_scores.keys()):
        return None
    total = 0
    for item_id, score in item_scores.items():
        total += _pss10_transform(item_id, score) if scale == "PSS-10" else score
    return total


# ---------------------------------------------------------------------------
# Persistence of extracted evidence (raw, per-item — not just totals)
# ---------------------------------------------------------------------------

def _store_items_sync(user_id: int, scale: str, items: List[SymptomItem]) -> None:
    if not items:
        return
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO assessment_records (user_id, assessment_type, item_id, score, evidence) "
            "VALUES (?, ?, ?, ?, ?)",
            [(user_id, scale, i.item_id, i.score, i.evidence) for i in items],
        )
        conn.commit()


def _store_impairments_sync(user_id: int, impairments: List[FunctionalImpairment]) -> None:
    if not impairments:
        return
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO functional_impairments (user_id, area, severity, evidence) "
            "VALUES (?, ?, ?, ?)",
            [(user_id, i.area, i.severity, i.evidence) for i in impairments],
        )
        conn.commit()


def _store_sleep_sync(user_id: int, hours: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sleep_reports (user_id, hours) VALUES (?, ?)",
            (user_id, hours),
        )
        conn.commit()


def _latest_scores_sync(user_id: int, scale: str) -> Dict[int, int]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT item_id, score FROM assessment_records
            WHERE user_id = ? AND assessment_type = ? AND score IS NOT NULL
            AND id IN (
                SELECT MAX(id) FROM assessment_records
                WHERE user_id = ? AND assessment_type = ? AND score IS NOT NULL
                GROUP BY item_id
            )
            """,
            (user_id, scale, user_id, scale),
        ).fetchall()
    return {r["item_id"]: r["score"] for r in rows}


async def persist_evidence_and_score(
    user_id: int,
    phq9_items: List[SymptomItem],
    gad7_items: List[SymptomItem],
    pss10_items: List[SymptomItem],
    impairments: List[FunctionalImpairment],
    sleep_hours: Optional[float],
) -> Dict[str, Optional[int]]:
    """Validates, stores raw evidence, and returns current official totals
    (None where the full item set isn't yet established) for each scale."""
    phq9_items = validate_symptom_items("PHQ-9", phq9_items)
    gad7_items = validate_symptom_items("GAD-7", gad7_items)
    pss10_items = validate_symptom_items("PSS-10", pss10_items)

    await run_db(_store_items_sync, user_id, "PHQ-9", phq9_items)
    await run_db(_store_items_sync, user_id, "GAD-7", gad7_items)
    await run_db(_store_items_sync, user_id, "PSS-10", pss10_items)
    await run_db(_store_impairments_sync, user_id, impairments)
    if sleep_hours is not None:
        await run_db(_store_sleep_sync, user_id, sleep_hours)

    totals: Dict[str, Optional[int]] = {}
    for scale in ("PHQ-9", "GAD-7", "PSS-10"):
        latest = await run_db(_latest_scores_sync, user_id, scale)
        totals[scale] = compute_total(scale, latest)
    return totals
