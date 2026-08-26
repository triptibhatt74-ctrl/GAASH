from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StateUt:
    code: str
    name: str
    kind: str


# ISO 3166-2:IN-compatible State / Union Territory identifiers.  Geography is
# deliberately data-driven so frontend components never need regional branches.
STATE_UTS: tuple[StateUt, ...] = (
    StateUt("IN-AP", "Andhra Pradesh", "state"),
    StateUt("IN-AR", "Arunachal Pradesh", "state"),
    StateUt("IN-AS", "Assam", "state"),
    StateUt("IN-BR", "Bihar", "state"),
    StateUt("IN-CT", "Chhattisgarh", "state"),
    StateUt("IN-GA", "Goa", "state"),
    StateUt("IN-GJ", "Gujarat", "state"),
    StateUt("IN-HR", "Haryana", "state"),
    StateUt("IN-HP", "Himachal Pradesh", "state"),
    StateUt("IN-JH", "Jharkhand", "state"),
    StateUt("IN-KA", "Karnataka", "state"),
    StateUt("IN-KL", "Kerala", "state"),
    StateUt("IN-MP", "Madhya Pradesh", "state"),
    StateUt("IN-MH", "Maharashtra", "state"),
    StateUt("IN-MN", "Manipur", "state"),
    StateUt("IN-ML", "Meghalaya", "state"),
    StateUt("IN-MZ", "Mizoram", "state"),
    StateUt("IN-NL", "Nagaland", "state"),
    StateUt("IN-OD", "Odisha", "state"),
    StateUt("IN-PB", "Punjab", "state"),
    StateUt("IN-RJ", "Rajasthan", "state"),
    StateUt("IN-SK", "Sikkim", "state"),
    StateUt("IN-TN", "Tamil Nadu", "state"),
    StateUt("IN-TS", "Telangana", "state"),
    StateUt("IN-TR", "Tripura", "state"),
    StateUt("IN-UP", "Uttar Pradesh", "state"),
    StateUt("IN-UT", "Uttarakhand", "state"),
    StateUt("IN-WB", "West Bengal", "state"),
    StateUt("IN-AN", "Andaman and Nicobar Islands", "union-territory"),
    StateUt("IN-CH", "Chandigarh", "union-territory"),
    StateUt("IN-DH", "Dadra and Nagar Haveli and Daman and Diu", "union-territory"),
    StateUt("IN-DL", "Delhi", "union-territory"),
    StateUt("IN-JK", "Jammu and Kashmir", "union-territory"),
    StateUt("IN-LA", "Ladakh", "union-territory"),
    StateUt("IN-LD", "Lakshadweep", "union-territory"),
    StateUt("IN-PY", "Puducherry", "union-territory"),
)

STATE_UT_BY_CODE = {item.code: item for item in STATE_UTS}


def canonical_state_ut(value: str | None) -> str | None:
    """Return a canonical State/UT code, or ``None`` for an unknown value."""

    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    return candidate if candidate in STATE_UT_BY_CODE else None


def require_state_ut(value: str) -> str:
    code = canonical_state_ut(value)
    if code is None:
        raise ValueError("A valid Indian State / UT code is required.")
    return code

