from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .common import ContractModel


class PlatformRole(StrEnum):
    USER = "user"
    COUNSELLOR = "counsellor"
    AUTHORIZED_OFFICIAL = "authorized_official"
    ADMIN = "admin"


class AuthenticatedPrincipal(ContractModel):
    """Server-derived identity. Never populate this from a client request."""

    user_id: int = Field(gt=0)
    role: PlatformRole = PlatformRole.USER


class RoleView(ContractModel):
    role: PlatformRole

