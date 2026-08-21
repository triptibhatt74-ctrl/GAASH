import sqlite3
import sys
import tempfile
from pathlib import Path

import authentication_jwt as auth
from fastapi.testclient import TestClient

# Use a throwaway database so we don't pollute gaash.db.
_tmp_db = Path(tempfile.gettempdir()) / f"gaash_auth_test_{auth.os.getpid()}.db"
if _tmp_db.exists():
    _tmp_db.unlink()
auth.DATABASE = str(_tmp_db)
# Re-initialize with the temp db path.
auth.init_db()

# Capture OTPs in memory instead of sending them.
_sent_otps = {}


def _mock_send_otp(identifier: str, identifier_type: str, otp: str, purpose: str):
    _sent_otps[(identifier, purpose)] = otp


auth.send_otp = _mock_send_otp

client = TestClient(auth.app)


def _expect(resp, code: int):
    if resp.status_code != code:
        print(f"FAIL {resp.url}: {resp.status_code} {resp.text}")
        sys.exit(1)


def test_register_requires_verification():
    resp = client.post(
        "/auth/register",
        json={"username": "alice", "email_or_phone": "alice@example.com", "password": "Password123"},
    )
    _expect(resp, 200)
    body = resp.json()
    assert body.get("status") == "verification_required", body
    assert "verification code has been sent" in body.get("message", "")
    assert ("alice@example.com", "REGISTRATION") in _sent_otps
    print("[PASS] /auth/register")


def test_verify_registration_then_login():
    otp = _sent_otps[("alice@example.com", "REGISTRATION")]
    resp = client.post(
        "/auth/verify-registration",
        json={"identifier": "alice@example.com", "otp": otp},
    )
    _expect(resp, 200)
    body = resp.json()
    assert "complete" in body.get("message", "").lower(), body

    # Login with email
    resp = client.post(
        "/auth/login",
        json={"identifier": "alice@example.com", "password": "Password123"},
    )
    _expect(resp, 200)
    body = resp.json()
    token = body["access_token"]
    assert body["user"]["gaash_id"].startswith("GSH-")
    print("[PASS] /auth/verify-registration + /auth/login")
    return token


def test_login_with_username_and_phone():
    # Register a phone user
    resp = client.post(
        "/auth/register",
        json={"username": "bob", "email_or_phone": "+919876543210", "password": "Password123"},
    )
    _expect(resp, 200)
    otp = _sent_otps[("+919876543210", "REGISTRATION")]
    resp = client.post(
        "/auth/verify-registration",
        json={"identifier": "+919876543210", "otp": otp},
    )
    _expect(resp, 200)

    resp = client.post(
        "/auth/login",
        json={"identifier": "bob", "password": "Password123"},
    )
    _expect(resp, 200)
    assert resp.json()["user"]["username"] == "bob"
    print("[PASS] login with username")


def test_unverified_cannot_login():
    resp = client.post(
        "/auth/register",
        json={"username": "carol", "email_or_phone": "carol@example.com", "password": "Password123"},
    )
    _expect(resp, 200)
    # Don't verify
    resp = client.post(
        "/auth/login",
        json={"identifier": "carol@example.com", "password": "Password123"},
    )
    _expect(resp, 401)
    print("[PASS] unverified user cannot login")


def test_password_reset_flow():
    resp = client.post(
        "/auth/forgot-password",
        json={"identifier": "alice@example.com"},
    )
    _expect(resp, 200)
    otp = _sent_otps[("alice@example.com", "PASSWORD_RESET")]

    resp = client.post(
        "/auth/verify-reset-otp",
        json={"identifier": "alice@example.com", "otp": otp},
    )
    _expect(resp, 200)
    reset_token = resp.json()["reset_token"]

    resp = client.post(
        "/auth/reset-password",
        json={"new_password": "NewPassword123"},
        headers={"Authorization": f"Bearer {reset_token}"},
    )
    _expect(resp, 200)

    # Old password fails
    resp = client.post(
        "/auth/login",
        json={"identifier": "alice@example.com", "password": "Password123"},
    )
    _expect(resp, 401)

    # New password works
    resp = client.post(
        "/auth/login",
        json={"identifier": "alice@example.com", "password": "NewPassword123"},
    )
    _expect(resp, 200)
    print("[PASS] password reset flow")


def test_bot_jwt_compatibility():
    # Get a fresh token from auth
    resp = client.post(
        "/auth/login",
        json={"identifier": "bob", "password": "Password123"},
    )
    _expect(resp, 200)
    token = resp.json()["access_token"]

    # Verify the token locally the same way bot.py does
    payload = auth.jwt.decode(token, auth.JWT_SECRET, algorithms=[auth.JWT_ALGORITHM])
    user_id = int(payload["sub"])
    assert isinstance(user_id, int)
    assert payload["gaash_id"].startswith("GSH-")
    print("[PASS] JWT is HS256 with numeric sub and gaash_id")


if __name__ == "__main__":
    test_register_requires_verification()
    test_verify_registration_then_login()
    test_login_with_username_and_phone()
    test_unverified_cannot_login()
    test_password_reset_flow()
    test_bot_jwt_compatibility()
    print("\nauthentication flow OK")
