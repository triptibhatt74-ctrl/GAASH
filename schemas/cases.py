from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from .common import ContractModel


class MonitoringStatus(StrEnum):
    NOT_ENROLLED = "not_enrolled"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class ConsentStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"


class CaseContextUpdate(ContractModel):
    """Beneficiary-editable context only; no role or external-system fields."""

    state_ut: str = Field(min_length=5, max_length=8, pattern=r"^IN-[A-Z]{2}$")
    district: str | None = Field(default=None, max_length=160)
    preferred_language: str | None = Field(default=None, min_length=2, max_length=20)

    @field_validator("district", "preferred_language", mode="before")
    @classmethod
    def empty_values_are_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


class CaseContextResponse(ContractModel):
    configured: bool
    id: str | None = Field(default=None, max_length=64)
    public_case_reference: str | None = Field(default=None, max_length=64)
    state_ut: str | None = Field(default=None, max_length=8)
    district: str | None = Field(default=None, max_length=160)
    monitoring_status: MonitoringStatus = MonitoringStatus.NOT_ENROLLED
    preferred_language: str | None = Field(default=None, max_length=20)
    consent_status: ConsentStatus = ConsentStatus.PENDING
    created_at: datetime | None = None
    updated_at: datetime | None = None

