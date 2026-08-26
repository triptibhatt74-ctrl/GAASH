from __future__ import annotations

from pydantic import Field

from .auth import PlatformRole
from .common import ContractModel


class UserProfileView(ContractModel):
    id: int = Field(gt=0)
    username: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=320)
    role: PlatformRole = PlatformRole.USER
    display_name: str | None = Field(default=None, max_length=160)
    preferred_language: str | None = Field(default=None, max_length=20)


class UserProfileUpdate(ContractModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=160)
    preferred_language: str | None = Field(default=None, min_length=2, max_length=20)

