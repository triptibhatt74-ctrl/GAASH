from __future__ import annotations

from pydantic import Field

from .common import ContractModel


class ApiError(ContractModel):
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=240)
    field_errors: dict[str, str] = Field(default_factory=dict)
    retry_after: int | None = Field(default=None, ge=1)
