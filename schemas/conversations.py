from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from .common import ContractModel


class MemoryObservationStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"


class MemoryFact(ContractModel):
    """A user-owned, provenance-bearing durable observation."""

    memory_key: str = Field(min_length=2, max_length=80)
    memory_value: str = Field(min_length=2, max_length=500)
    memory_type: str = Field(min_length=2, max_length=80)
    source_conversation_id: str | None = Field(default=None, max_length=64)
    source_message_id: int | None = Field(default=None, gt=0)
    source_kind: str = Field(default="conversation", max_length=40)
    confidence: float = Field(ge=0, le=1)
    status: MemoryObservationStatus = MemoryObservationStatus.ACTIVE
    observed_at: datetime


class ConversationSummaryView(ContractModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    summary_text: str = Field(min_length=1, max_length=5000)
    updated_at: datetime


class MemoryRetrievalRequest(ContractModel):
    query: str = Field(min_length=1, max_length=12000)
    conversation_id: str | None = Field(default=None, max_length=64)
    max_facts: int = Field(default=6, ge=1, le=12)
    max_summaries: int = Field(default=3, ge=0, le=6)
    char_budget: int = Field(default=4000, ge=300, le=8000)


class MemoryRetrievalResponse(ContractModel):
    facts: list[MemoryFact] = Field(default_factory=list)
    conversation_summaries: list[ConversationSummaryView] = Field(default_factory=list)
    char_budget: int = Field(ge=0)

