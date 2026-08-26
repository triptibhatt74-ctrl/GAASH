from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base contract: reject unexpected client fields by default."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StatusResponse(ContractModel):
    status: str = Field(min_length=1, max_length=80)


class PageInfo(ContractModel):
    limit: int = Field(ge=1, le=100)
    has_more: bool = False
    next_cursor: str | None = Field(default=None, max_length=256)


class Timestamped(ContractModel):
    created_at: datetime
    updated_at: datetime | None = None


T = TypeVar("T")


class Page(ContractModel, Generic[T]):
    items: list[T] = Field(default_factory=list)
    page: PageInfo

