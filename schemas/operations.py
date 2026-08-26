"""Phase 3 contracts for authorized operational support.

These schemas intentionally separate private wellbeing data from the minimum
necessary operational information available to authorized support staff.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from .common import ContractModel, PageInfo
from .wellbeing import DistressIndicator, DistressTrajectory, DynamicDistressLevel


class CasePermission(StrEnum):
    CASE_SUMMARY = "case_summary"
    CASE_MANAGE = "case_manage"
    INTERVENTION_UPDATE = "intervention_update"


class InterventionType(StrEnum):
    COUNSELLING = "counselling"
    MENTAL_HEALTH_REFERRAL = "mental_health_referral"
    MEDICAL_REFERRAL = "medical_referral"
    LEGAL_AID = "legal_aid"
    WITNESS_PROTECTION_SUPPORT = "witness_protection_support"
    REHABILITATION = "rehabilitation"
    FINANCIAL_COMPENSATION_SUPPORT = "financial_compensation_support"
    RELOCATION_SUPPORT_SERVICES = "relocation_support_services"
    EMERGENCY_ASSISTANCE = "emergency_assistance"


class InterventionStatus(StrEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AlertReviewDecision(StrEnum):
    CONFIRMED = "confirmed"
    NOT_CONCERNING = "not_concerning"
    NEEDS_FOLLOW_UP = "needs_follow_up"
    FALSE_POSITIVE = "false_positive"


class AggregateLevel(StrEnum):
    DISTRICT = "district"
    STATE_UT = "state_ut"
    NATIONAL = "national"


class OperationalCaseSummary(ContractModel):
    """Pseudonymous case queue item; no names, chats, transcripts, or answers."""

    case_id: str = Field(min_length=1, max_length=64)
    public_case_reference: str = Field(min_length=1, max_length=64)
    priority: DynamicDistressLevel
    trajectory: DistressTrajectory = DistressTrajectory.UNKNOWN
    reason_summary: str = Field(min_length=1, max_length=500)
    last_evaluated_at: datetime | None = None
    last_reviewed_at: datetime | None = None
    assigned_professional: str | None = Field(default=None, max_length=64)
    pending_action: str | None = Field(default=None, max_length=80)
    alert_id: int | None = Field(default=None, gt=0)


class CaseQueueResponse(ContractModel):
    cases: list[OperationalCaseSummary] = Field(default_factory=list)
    page: PageInfo


class AssessmentSnapshot(ContractModel):
    assessment_type: str = Field(pattern=r"^(PHQ-9|GAD-7|PSS-10)$")
    score: int = Field(ge=0, le=40)
    completed_at: datetime | None = None


class CaseDetailResponse(OperationalCaseSummary):
    state_ut: str = Field(pattern=r"^IN-[A-Z]{2}$")
    district: str | None = Field(default=None, max_length=160)
    assessment_history: list[AssessmentSnapshot] = Field(default_factory=list)
    contributing_indicators: list[DistressIndicator] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    data_period_start: datetime | None = None
    data_period_end: datetime | None = None


class AlertReviewRequest(ContractModel):
    decision: AlertReviewDecision
    note: str | None = Field(default=None, max_length=2000)


class AlertReviewResponse(ContractModel):
    review_id: int = Field(gt=0)
    alert_id: int = Field(gt=0)
    decision: AlertReviewDecision
    reviewed_at: datetime


class CaseAssignmentRequest(ContractModel):
    assigned_to_user_id: int = Field(gt=0)


class CaseAssignmentResponse(ContractModel):
    case_id: str = Field(min_length=1, max_length=64)
    assigned_to_user_id: int = Field(gt=0)
    status: str = Field(pattern=r"^(active|reassigned|closed)$")
    updated_at: datetime


class InterventionCreate(ContractModel):
    intervention_type: InterventionType
    assigned_to_user_id: int | None = Field(default=None, gt=0)
    notes: str | None = Field(default=None, max_length=2000)


class InterventionUpdate(ContractModel):
    status: InterventionStatus | None = None
    assigned_to_user_id: int | None = Field(default=None, gt=0)
    notes: str | None = Field(default=None, max_length=2000)
    outcome: str | None = Field(default=None, max_length=2000)

    @field_validator("notes", "outcome", mode="before")
    @classmethod
    def empty_optional_text_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


class InterventionResponse(ContractModel):
    id: int = Field(gt=0)
    case_id: str = Field(min_length=1, max_length=64)
    intervention_type: InterventionType
    status: InterventionStatus
    created_by: int = Field(gt=0)
    assigned_to: int | None = Field(default=None, gt=0)
    created_at: datetime
    updated_at: datetime
    notes: str | None = Field(default=None, max_length=2000)
    outcome: str | None = Field(default=None, max_length=2000)


class CaseNoteCreate(ContractModel):
    note: str = Field(min_length=1, max_length=2000)


class CaseNoteResponse(ContractModel):
    id: int = Field(gt=0)
    created_at: datetime
    created_by: int = Field(gt=0)
    note: str = Field(min_length=1, max_length=2000)


class AggregateMetric(ContractModel):
    key: str = Field(min_length=1, max_length=80)
    value: int | None = Field(default=None, ge=0)
    suppressed: bool = False


class OperationalAggregateResponse(ContractModel):
    level: AggregateLevel
    state_ut: str | None = Field(default=None, pattern=r"^IN-[A-Z]{2}$")
    district: str | None = Field(default=None, max_length=160)
    minimum_cell_count: int = Field(ge=2, le=100)
    metrics: list[AggregateMetric] = Field(default_factory=list)
    generated_at: datetime


class IntegrationAdapterStatus(ContractModel):
    source: str = Field(min_length=2, max_length=80)
    status: str = Field(default="configuration_required", max_length=80)
    live_integration: bool = False
    last_synced_at: datetime | None = None
