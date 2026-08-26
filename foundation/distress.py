"""Explainable, conservative longitudinal distress decision support.

This is intentionally not a diagnostic or crisis classifier.  It requires
corroboration for consequential states and returns UNKNOWN for thin evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any


_LEVEL_ORDER = {"unknown": 0, "stable": 1, "watch": 2, "elevated": 3, "high": 4, "critical": 5}


def _as_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _signal(source: str, label: str, direction: str = "unknown", confidence: float | None = None) -> dict[str, Any]:
    return {"source": source, "label": label, "direction": direction, "confidence": confidence}


def compute_dynamic_distress_state(
    *,
    events: Iterable[Mapping[str, Any]],
    assessment_trends: Mapping[str, object] | None = None,
    engagement: Mapping[str, object] | None = None,
    previous_states: Iterable[Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Use available observations without converting absence into evidence.

    HIGH requires at least two independent, recent corroborating signal classes
    (unless a real safety emergency is provided). Engagement contributes only
    as context and can never independently increase the state past WATCH.
    """

    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(days=21)
    observed = [item for item in events if (_as_time(item.get("timestamp")) or current_time) >= cutoff]
    indicators: list[dict[str, Any]] = []
    classes: set[str] = set()
    severe_classes: set[str] = set()
    emergency = False

    for event in observed:
        source = str(event.get("source_type") or "text")
        signals = event.get("structured_signals") if isinstance(event.get("structured_signals"), Mapping) else {}
        confidence = signals.get("emotion_confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        if signals.get("emergency") is True:
            emergency = True
            indicators.append(_signal("safety", "A safety escalation signal was supplied by the wellbeing service.", "worsening", confidence))
            classes.add("safety")
            continue
        risk = str(signals.get("risk_category") or "").upper()
        strength = str(signals.get("distress_strength") or "").lower()
        if risk == "HIGH_RISK":
            classes.add("risk")
            severe_classes.add("risk")
            indicators.append(_signal(source, "A high-risk wellbeing signal was supplied by the existing safety flow.", "worsening", confidence))
        elif risk == "MODERATE_RISK":
            classes.add("risk")
            indicators.append(_signal(source, "A moderate wellbeing risk signal was supplied by the existing safety flow.", "worsening", confidence))
        if strength == "high":
            classes.add(source)
            severe_classes.add(source)
            indicators.append(_signal(source, "A high-strength distress signal was observed.", "worsening", confidence))
        elif strength == "moderate":
            classes.add(source)
            indicators.append(_signal(source, "A moderate distress signal was observed.", "worsening", confidence))

    trends = assessment_trends or {}
    worsening_scales = [str(name) for name, value in trends.items() if isinstance(value, str) and "worsen" in value.lower()]
    improving_scales = [str(name) for name, value in trends.items() if isinstance(value, str) and "improv" in value.lower()]
    if worsening_scales:
        classes.add("assessment")
        indicators.append(_signal("assessment", f"Recent screening trend: {', '.join(worsening_scales[:3])} worsening.", "worsening"))
    elif improving_scales:
        indicators.append(_signal("assessment", f"Recent screening trend: {', '.join(improving_scales[:3])} improving.", "improving"))

    engagement_change = str((engagement or {}).get("change") or "unknown").lower()
    if engagement_change in {"decreased", "abrupt_change"}:
        indicators.append(_signal("engagement", "Recent engagement changed; this is contextual and not a safety conclusion.", "unknown"))

    historical_levels = [str(item.get("state") or "unknown").lower() for item in previous_states]
    persisted_elevated = sum(1 for level in historical_levels[:2] if _LEVEL_ORDER.get(level, 0) >= _LEVEL_ORDER["elevated"]) >= 2

    if emergency:
        state = "critical"
        trajectory = "worsening"
    elif len(severe_classes) >= 2 or ("risk" in severe_classes and "assessment" in classes) or (persisted_elevated and len(classes) >= 1):
        state = "high"
        trajectory = "worsening"
    elif len(classes) >= 2:
        state = "elevated"
        trajectory = "worsening"
    elif len(classes) == 1:
        state = "watch"
        trajectory = "worsening" if worsening_scales else "unknown"
    elif observed or improving_scales:
        state = "stable"
        trajectory = "improving" if improving_scales else "stable"
    else:
        state = "unknown"
        trajectory = "unknown"

    evidence_count = len({item["source"] for item in indicators if item["source"] != "engagement"})
    confidence = 0.0 if state == "unknown" else min(0.9, 0.3 + (evidence_count * 0.2) + (0.1 if persisted_elevated else 0))
    return {
        "state": state,
        "trajectory": trajectory,
        "confidence": round(confidence, 2),
        "contributing_indicators": indicators[:8],
        "requires_human_review": state in {"elevated", "high", "critical"},
        "computed_at": current_time,
    }
