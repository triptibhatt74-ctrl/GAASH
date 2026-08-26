"""Phase 2 contracts for evidence-backed longitudinal wellbeing monitoring.

These contracts deliberately describe observations and decision-support state,
not diagnoses.  They carry provenance without retaining raw audio or an LLM's
private reasoning.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from .common import ContractModel


class WellbeingEventSource(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    ASSESSMENT = "assessment"
    ENGAGEMENT = "engagement"
    CHECK_IN = "check_in"
    SAFETY = "safety"


class DynamicDistressLevel(StrEnum):
    UNKNOWN = "unknown"
    STABLE = "stable"
    WATCH = "watch"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class DistressTrajectory(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class WellbeingEventCreate(ContractModel):
    source_type: WellbeingEventSource
    structured_signals: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_reference: str | None = Field(default=None, max_length=128)
    conversation_id: str | None = Field(default=None, max_length=64)
    assessment_id: str | None = Field(default=None, max_length=64)


class WellbeingEventResponse(WellbeingEventCreate):
    id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    case_id: str | None = Field(default=None, max_length=64)
    timestamp: datetime


class DistressIndicator(ContractModel):
    source: WellbeingEventSource | str = Field(min_length=2, max_length=40)
    label: str = Field(min_length=2, max_length=160)
    direction: DistressTrajectory = DistressTrajectory.UNKNOWN
    confidence: float | None = Field(default=None, ge=0, le=1)


class DynamicDistressStateResponse(ContractModel):
    state: DynamicDistressLevel
    trajectory: DistressTrajectory = DistressTrajectory.UNKNOWN
    confidence: float = Field(ge=0, le=1)
    contributing_indicators: list[DistressIndicator] = Field(default_factory=list)
    requires_human_review: bool = False
    computed_at: datetime


class LongitudinalPredictionResponse(ContractModel):
    """Conservative worsening-risk decision support, never a diagnosis."""

    current_state: DynamicDistressLevel
    trajectory: DistressTrajectory = DistressTrajectory.UNKNOWN
    worsening_risk: str = Field(default="unknown", max_length=24)
    confidence: float = Field(ge=0, le=1)
    contributing_indicators: list[DistressIndicator] = Field(default_factory=list)
    requires_human_review: bool = False
    evaluated_at: datetime


class VoiceAnalysisResponse(ContractModel):
    transcript: str = Field(min_length=1, max_length=12000)
    transcript_status: str = Field(default="available", max_length=40)
    text_emotion: dict[str, Any] | None = None
    acoustic_emotion: dict[str, Any] | None = None
    acoustic_status: str = Field(default="unavailable", max_length=80)
    combined_emotion: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None
    duration_seconds: float | None = Field(default=None, ge=0, le=60 * 30)


class MonitoringAlertResponse(ContractModel):
    id: int = Field(gt=0)
    state: DynamicDistressLevel
    reason: str = Field(min_length=2, max_length=500)
    requires_human_review: bool
    status: str = Field(min_length=2, max_length=40)
    created_at: datetime


class EngagementSummary(ContractModel):
    period_start: datetime | None = None
    period_end: datetime | None = None
    conversation_count: int = Field(default=0, ge=0)
    assessment_completion_count: int = Field(default=0, ge=0)
    missed_check_in: bool = False
    change: str = Field(default="unknown", max_length=40)
    note: str = Field(default="", max_length=240)
