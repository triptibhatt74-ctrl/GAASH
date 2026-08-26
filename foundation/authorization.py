from __future__ import annotations

from collections.abc import Iterable

from schemas.auth import PlatformRole


def is_role_allowed(role: PlatformRole, allowed_roles: Iterable[PlatformRole]) -> bool:
    """Small pure guard used by independently deployed backend services."""

    return role in set(allowed_roles)

