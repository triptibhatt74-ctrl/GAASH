"""
Wraps the EXISTING authentication mechanism (the `gaash_id` cookie set by
POST /login in the existing notebook) as a reusable FastAPI dependency.
No new auth system, no new Gaash-ID scheme, no token issuance here.

SECURITY NOTE (flagging, not fixing — auth is explicitly out of scope):
the existing cookie is a bare "GSH-<int>" string with no signature
(`secure=False`, no HMAC/JWT). Anyone who edits that cookie value can
impersonate another user. The ownership checks below enforce that a caller
can only touch their own gaash_id / role-appropriate data *given* the
identity the cookie claims — but they cannot detect a forged cookie value,
because the existing auth layer doesn't give them anything to verify it
against. This should be hardened (signed session cookie or JWT) before any
real user's screening data depends on it.
"""
import sqlite3

from fastapi import Depends, HTTPException, Request

from db import get_conn


def _parse_gaash_id(raw: str) -> int:
    try:
        return int(raw.replace("GSH-", ""))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid Gaash ID.")


def get_current_user_id(request: Request) -> int:
    """Mirrors the existing endpoints' `request.cookies.get("gaash_id")`
    check. Raises 401 exactly like the existing /language endpoint does."""
    raw = request.cookies.get("gaash_id")
    if not raw:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return _parse_gaash_id(raw)


def get_current_user(request: Request) -> sqlite3.Row:
    """Full current-user row (id, name, email_or_phone, age, language,
    role) for handlers that need role information."""
    user_id = get_current_user_id(request)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name, email_or_phone, age, language, role "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return row


def require_self_or_staff(path_gaash_id: str, current=Depends(get_current_user)):
    """Ownership check for /users/{gaash_id}/... routes: the caller must
    either be the user in question, or hold a staff role (counsellor/admin)
    per the existing `role` column. A user must not read another user's
    screening data."""
    target_id = _parse_gaash_id(path_gaash_id)
    if current["id"] != target_id and current["role"] not in ("counsellor", "admin"):
        raise HTTPException(status_code=403, detail="Forbidden")
    return target_id
