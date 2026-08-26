"""Safe normalisation for the Phase 2 wellbeing event timeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ALLOWED_RISK_CATEGORIES = {"UNKNOWN", "LOW_RISK", "MODERATE_RISK", "HIGH_RISK"}
ALLOWED_STRENGTHS = {"low", "moderate", "high"}


def _text(value: object, maximum: int = 120) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:maximum] if value else None


def _confidence(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, number)) if number == number else None


def _strength(value: object) -> str | None:
    text = _text(value, 16)
    return text.lower() if text and text.lower() in ALLOWED_STRENGTHS else None


def normalise_event_signals(source_type: str, raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a compact, provenance-safe signal set.

    Raw messages, transcripts, model prompts, and hidden reasoning do not enter
    the event timeline.  Only already-produced structured observations do.
    """

    source = raw or {}
    result: dict[str, Any] = {"source_type": source_type}
    primary = _text(source.get("primary_emotion"), 40)
    severity = _strength(source.get("distress_strength") or source.get("emotion_severity"))
    risk_category = _text(source.get("risk_category"), 32)
    if primary:
        result["primary_emotion"] = primary
    confidence = _confidence(source.get("emotion_confidence") or source.get("confidence"))
    if confidence is not None:
        result["emotion_confidence"] = confidence
    if severity:
        result["distress_strength"] = severity
    if risk_category and risk_category.upper() in ALLOWED_RISK_CATEGORIES:
        result["risk_category"] = risk_category.upper()
    if isinstance(source.get("emergency"), bool):
        result["emergency"] = source["emergency"]
    trajectory = _text(source.get("assessment_trajectory"), 32)
    if trajectory in {"improving", "stable", "worsening", "unknown"}:
        result["assessment_trajectory"] = trajectory
    if isinstance(source.get("acoustic_available"), bool):
        result["acoustic_available"] = source["acoustic_available"]
    if isinstance(source.get("speech_detected"), bool):
        result["speech_detected"] = source["speech_detected"]
    return result
