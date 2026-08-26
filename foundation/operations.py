"""Pure Phase 3 operational-policy helpers.

No helper here receives raw chat content or makes a clinical conclusion.  The
database layer decides *who* is authorized; these helpers make queue priority
and aggregate suppression deterministic and easy to test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


_PRIORITY_ORDER = {"critical": 4, "high": 3, "elevated": 2, "watch": 1, "stable": 0, "unknown": 0}


def normalize_priority(value: object) -> str:
    candidate = str(value or "unknown").lower()
    return candidate if candidate in _PRIORITY_ORDER else "unknown"


def concise_reason_summary(indicators: Iterable[Mapping[str, Any]], fallback: str = "Human review is pending for this case.") -> str:
    """Return existing evidence labels only, without model reasoning or PII."""

    labels: list[str] = []
    for indicator in indicators:
        label = str(indicator.get("label") or "").strip()
        if label and label not in labels:
            labels.append(label[:180])
        if len(labels) == 2:
            break
    return "; ".join(labels)[:500] or fallback


def sort_case_queue(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort queue entries by existing state, then most recent evaluation."""

    copied = [dict(row) for row in rows]
    return sorted(
        copied,
        key=lambda item: (
            _PRIORITY_ORDER[normalize_priority(item.get("priority") or item.get("state"))],
            str(item.get("last_evaluated_at") or ""),
        ),
        reverse=True,
    )


def suppress_small_cell(value: int, minimum_cell_count: int) -> dict[str, int | bool | None]:
    """Avoid returning small operational counts that could identify a person."""

    normalized = max(0, int(value))
    if normalized < minimum_cell_count:
        return {"value": None, "suppressed": True}
    return {"value": normalized, "suppressed": False}
