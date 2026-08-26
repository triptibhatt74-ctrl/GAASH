"""Integration-ready contracts for external case systems.

No connector is enabled here.  Deployments must provide a reviewed source
specification, credentials held only by the backend, and a data-protection
assessment before an adapter can make any outbound request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class ExternalCaseRecord:
    source: str
    external_case_id: str
    metadata: dict[str, Any]
    last_synced_at: datetime | None = None


class ExternalCaseAdapter(Protocol):
    """A future adapter must return metadata only, never private chat content."""

    source: str

    def validate_record(self, record: ExternalCaseRecord) -> ExternalCaseRecord: ...


def integration_status(source: str, enabled: bool = False) -> dict[str, object]:
    """A truthful configuration status; it never claims a live connection."""

    return {
        "source": source,
        "status": "disabled" if not enabled else "configuration_required",
        "live_integration": False,
        "last_synced_at": None,
    }

