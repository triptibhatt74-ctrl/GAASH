from __future__ import annotations

from datetime import date

from pydantic import Field, HttpUrl

from .common import ContractModel


class NationalResource(ContractModel):
    """Verified resource data contract. Optional fields stay absent when unknown."""

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=300)
    category: str = Field(min_length=1, max_length=120)
    state_ut: str | None = Field(default=None, max_length=32)
    district: str | None = Field(default=None, max_length=160)
    city: str | None = Field(default=None, max_length=160)
    description: str | None = Field(default=None, max_length=3000)
    phone: list[str] = Field(default_factory=list, max_length=8)
    website: HttpUrl | None = None
    address: str | None = Field(default=None, max_length=1000)
    access_mode: str | None = Field(default=None, max_length=120)
    availability: str | None = Field(default=None, max_length=400)
    eligibility: str | None = Field(default=None, max_length=1000)
    source_url: HttpUrl | None = None
    last_verified: date | None = None
    coordinates: tuple[float, float] | None = None

