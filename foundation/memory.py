from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Mapping


def _terms(text: str) -> set[str]:
    return {item for item in re.findall(r"[\w'-]{3,}", text.casefold(), flags=re.UNICODE) if item}


def _as_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def select_bounded_owned_memory(
    *,
    user_id: int,
    query: str,
    memory_rows: Iterable[Mapping[str, object]],
    summary_rows: Iterable[Mapping[str, object]],
    max_facts: int,
    max_summaries: int,
    char_budget: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Rank only rows already constrained to the authenticated owner.

    The defensive user-id check makes accidental cross-user repository results
    ineligible.  Selection is relevance + recency bounded and never sends raw
    historic messages to a model or caller.
    """

    query_terms = _terms(query)
    facts: list[tuple[float, dict[str, object]]] = []
    for row in memory_rows:
        if int(row.get("user_id") or -1) != user_id or row.get("status", "active") != "active":
            continue
        item = dict(row)
        searchable = " ".join((_as_text(item.get("memory_key")), _as_text(item.get("memory_value")), _as_text(item.get("memory_type"))))
        overlap = len(query_terms & _terms(searchable))
        confidence = float(item.get("confidence") or 0)
        if overlap or not query_terms:
            facts.append((overlap * 3 + confidence, item))
    facts.sort(key=lambda value: (value[0], str(value[1].get("updated_at") or "")), reverse=True)

    summaries: list[tuple[float, dict[str, object]]] = []
    for row in summary_rows:
        if int(row.get("user_id") or -1) != user_id:
            continue
        item = dict(row)
        content = _as_text(item.get("summary_text"))
        overlap = len(query_terms & _terms(content))
        if content and (overlap or not query_terms):
            summaries.append((overlap * 3, item))
    summaries.sort(key=lambda value: (value[0], str(value[1].get("updated_at") or "")), reverse=True)

    used = 0
    selected_facts: list[dict[str, object]] = []
    for _, item in facts[:max_facts]:
        content_length = len(_as_text(item.get("memory_key"))) + len(_as_text(item.get("memory_value")))
        if used + content_length > char_budget:
            break
        selected_facts.append(item)
        used += content_length

    selected_summaries: list[dict[str, object]] = []
    for _, item in summaries[:max_summaries]:
        summary = _as_text(item.get("summary_text"))
        remaining = char_budget - used
        if remaining <= 0:
            break
        if len(summary) > remaining:
            item["summary_text"] = summary[:remaining].rstrip()
        selected_summaries.append(item)
        used += len(_as_text(item.get("summary_text")))
    return selected_facts, selected_summaries

