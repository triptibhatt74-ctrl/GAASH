"""Shared privacy-notice persistence helpers for GAASH services.

The privacy notice is an acknowledgement record, not a blanket consent for all
processing.  Individual optional processing choices are recorded separately.

Privacy text requires review by qualified Indian legal/privacy counsel before
public production launch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date
from typing import Any


PRIVACY_NOTICE_TYPE = "privacy"
VOICE_TRANSCRIPTION_PURPOSE = "voice_transcription"


def current_policy_version() -> str:
    """Return the server-configured version for the published privacy notice."""
    version = os.getenv("GAASH_PRIVACY_POLICY_VERSION", "1.0").strip()
    if not version or len(version) > 64:
        raise RuntimeError("GAASH_PRIVACY_POLICY_VERSION must be between 1 and 64 characters.")
    return version


def current_policy_effective_date() -> str:
    """Return a validated ISO date configured by the server operator."""
    effective_date = os.getenv("GAASH_PRIVACY_POLICY_EFFECTIVE_DATE", "2026-08-22").strip()
    try:
        return date.fromisoformat(effective_date).isoformat()
    except ValueError as exc:
        raise RuntimeError("GAASH_PRIVACY_POLICY_EFFECTIVE_DATE must be an ISO date.") from exc


def create_privacy_tables(conn: Any) -> None:
    """Create the minimal account-bound acknowledgement and optional-consent tables."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS privacy_acknowledgements (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                policy_version TEXT NOT NULL,
                accepted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                locale TEXT,
                notice_type TEXT NOT NULL DEFAULT 'privacy',
                UNIQUE (user_id, policy_version)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_privacy_acknowledgements_current
            ON privacy_acknowledgements (user_id, policy_version, notice_type)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS privacy_processing_consents (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                purpose TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (decision IN ('granted', 'withdrawn')),
                policy_version TEXT NOT NULL,
                locale TEXT,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_privacy_processing_consents_latest
            ON privacy_processing_consents (user_id, purpose, recorded_at DESC, id DESC)
            """
        )


def _field(row: Any, name: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return row[index]


def get_privacy_acknowledgement(conn: Any, user_id: int, policy_version: str | None = None) -> dict[str, Any] | None:
    version = policy_version or current_policy_version()
    row = conn.execute(
        """
        SELECT policy_version, accepted_at, locale
        FROM privacy_acknowledgements
        WHERE user_id = %s AND policy_version = %s AND notice_type = %s
        ORDER BY accepted_at DESC, id DESC
        LIMIT 1
        """,
        (user_id, version, PRIVACY_NOTICE_TYPE),
    ).fetchone()
    if row is None:
        return None
    return {
        "policy_version": _field(row, "policy_version", 0),
        "accepted_at": _field(row, "accepted_at", 1),
        "locale": _field(row, "locale", 2),
    }


def record_privacy_acknowledgement(conn: Any, user_id: int, policy_version: str, locale: str | None = None) -> dict[str, Any]:
    current_version = current_policy_version()
    if policy_version != current_version:
        raise ValueError("The privacy notice version is no longer current. Refresh and review the latest notice.")
    conn.execute(
        """
        INSERT INTO privacy_acknowledgements (user_id, policy_version, locale, notice_type)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id, policy_version) DO NOTHING
        """,
        (user_id, current_version, locale, PRIVACY_NOTICE_TYPE),
    )
    acknowledgement = get_privacy_acknowledgement(conn, user_id, current_version)
    if acknowledgement is None:
        raise RuntimeError("Privacy acknowledgement could not be confirmed.")
    return acknowledgement


def get_voice_transcription_consent(conn: Any, user_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT decision, recorded_at, policy_version
        FROM privacy_processing_consents
        WHERE user_id = %s AND purpose = %s
        ORDER BY recorded_at DESC, id DESC
        LIMIT 1
        """,
        (user_id, VOICE_TRANSCRIPTION_PURPOSE),
    ).fetchone()
    if row is None:
        return {"granted": False, "recorded_at": None, "policy_version": None}
    return {
        "granted": _field(row, "decision", 0) == "granted",
        "recorded_at": _field(row, "recorded_at", 1),
        "policy_version": _field(row, "policy_version", 2),
    }


def record_voice_transcription_consent(conn: Any, user_id: int, granted: bool, locale: str | None = None) -> dict[str, Any]:
    decision = "granted" if granted else "withdrawn"
    conn.execute(
        """
        INSERT INTO privacy_processing_consents (user_id, purpose, decision, policy_version, locale)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (user_id, VOICE_TRANSCRIPTION_PURPOSE, decision, current_policy_version(), locale),
    )
    return get_voice_transcription_consent(conn, user_id)
