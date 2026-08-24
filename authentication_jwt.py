import contextlib
import hashlib
import hmac
import html
import logging
import os
import asyncio
import secrets
import smtplib
import threading
import time
import psycopg
from psycopg import errors as pg_errors
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import jwt
from psycopg_pool import ConnectionPool
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from privacy import (
    current_policy_effective_date,
    current_policy_version,
    create_privacy_tables,
    get_privacy_acknowledgement,
    get_voice_transcription_consent,
    record_privacy_acknowledgement,
    record_voice_transcription_consent,
)

load_dotenv()
logger = logging.getLogger("gaash.auth")

# ============================================================
# CONFIGURATION
# ============================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    AUTH_DB_POOL.open()
    try:
        init_db()
        yield
    finally:
        AUTH_DB_POOL.close()


app = FastAPI(
    title="Gaash Authentication API",
    description="Authentication and account-management backend for Gaash",
    version="2.0.0",
    lifespan=lifespan,
)

# AUTH_HOST/AUTH_PORT let both services share one .env without colliding with bot.py's HOST/PORT.
HOST = os.getenv("AUTH_HOST") or os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("AUTH_PORT") or os.getenv("PORT", "8004"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

JWT_SECRET = os.getenv("GAASH_JWT_SECRET", "").strip()
if not JWT_SECRET:
    raise RuntimeError("GAASH_JWT_SECRET is not configured.")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
AUTH_SESSION_EXPIRE_DAYS = int(
    os.getenv("AUTH_SESSION_EXPIRE_DAYS", "30")
)

AUTH_SESSION_COOKIE_NAME = os.getenv(
    "AUTH_SESSION_COOKIE_NAME",
    "gaash_session",
)

AUTH_COOKIE_SECURE = (
    os.getenv("AUTH_COOKIE_SECURE", "true").lower()
    == "true"
)

AUTH_COOKIE_SAMESITE = os.getenv(
    "AUTH_COOKIE_SAMESITE",
    "none",
).lower()

if AUTH_COOKIE_SAMESITE not in {
    "lax",
    "strict",
    "none",
}:
    raise RuntimeError(
        "AUTH_COOKIE_SAMESITE must be lax, strict, or none."
    )

OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "2"))
OTP_RESEND_COOLDOWN_SECONDS = int(os.getenv("OTP_RESEND_COOLDOWN_SECONDS", "60"))
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "10"))
MAX_OTP_ATTEMPTS = 5

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

# Optional Resend email provider (takes precedence over SMTP when configured).
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
RESEND_FROM_EMAIL = os.getenv(
    "RESEND_FROM_EMAIL",
    "Gaash <onboarding@resend.dev>",
).strip()


_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:5174,http://127.0.0.1:5174"
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # Authentication uses Authorization headers, not cross-site cookies.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

_RATE_LIMIT_BUCKETS: dict[tuple[str, str], list[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def enforce_rate_limit(request: Request, scope: str, limit: int, window_seconds: int) -> None:
    """Apply a small process-local guard without recording account identifiers."""
    client = request.client.host if request.client else "unknown"
    key = (scope, client)
    now = time.monotonic()
    with _RATE_LIMIT_LOCK:
        attempts = [started for started in _RATE_LIMIT_BUCKETS.get(key, []) if now - started < window_seconds]
        if len(attempts) >= limit:
            retry_after = max(1, int(window_seconds - (now - attempts[0])) + 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Please wait before trying again.",
                headers={"Retry-After": str(retry_after)},
            )
        attempts.append(now)
        _RATE_LIMIT_BUCKETS[key] = attempts


class OTPServiceError(Exception):
    """Raised when an OTP cannot be delivered because the provider is not configured."""
    
AUTH_DB_POOL = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=5,
    timeout=5,
    open=False,
)

# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg.connect(DATABASE_URL)

@contextlib.contextmanager
def db_connection():
    with AUTH_DB_POOL.connection() as conn:
        yield conn

def init_db() -> None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS otp_verifications (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    email TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    otp_hash TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    used BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_otp_user_purpose
                ON otp_verifications (user_id, purpose, created_at)
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reset_tokens (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT UNIQUE NOT NULL,
                    used BOOLEAN NOT NULL DEFAULT FALSE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reset_token_hash
                ON reset_tokens (token_hash)
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_access_tokens (
                    token_hash TEXT PRIMARY KEY,
                    expires_at TIMESTAMPTZ NOT NULL,
                    revoked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id BIGSERIAL PRIMARY KEY,

                    user_id BIGINT NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE,

                    token_hash TEXT UNIQUE NOT NULL,

                    expires_at TIMESTAMPTZ NOT NULL,

                    revoked BOOLEAN NOT NULL
                        DEFAULT FALSE,

                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    last_used_at TIMESTAMPTZ
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                ON auth_sessions (
                    user_id,
                    expires_at
                )
                """
            )
        create_privacy_tables(conn)

        conn.commit()

def _health_db_sync() -> bool:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()

    return bool(row and row[0] == 1)

# ============================================================
# PASSWORD + OTP HELPERS
# ============================================================

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=64,
    )
    return f"{salt.hex()}:{key.hex()}"


def verify_password(password: str, stored_password: str) -> bool:
    try:
        salt_hex, key_hex = stored_password.split(":")
        new_key = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=16384,
            r=8,
            p=1,
            dklen=64,
        )
        return hmac.compare_digest(new_key, bytes.fromhex(key_hex))
    except (ValueError, TypeError):
        return False


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_access_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def hash_session_token(
    token: str,
) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def is_access_token_revoked(token: str) -> bool:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM revoked_access_tokens WHERE token_hash = %s AND expires_at > CURRENT_TIMESTAMP",
                (hash_access_token(token),),
            )
            return cur.fetchone() is not None


def revoke_access_token(token: str) -> None:
    """Record a validated access token as revoked until its normal expiry."""
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    with db_connection() as conn:
        with conn.cursor() as cur:
            # Opportunistically discard no-longer-relevant revocations.
            cur.execute("DELETE FROM revoked_access_tokens WHERE expires_at <= CURRENT_TIMESTAMP")
            cur.execute(
                """
                INSERT INTO revoked_access_tokens (token_hash, expires_at)
                VALUES (%s, %s)
                ON CONFLICT (token_hash) DO NOTHING
                """,
                (hash_access_token(token), expires_at),
            )
        conn.commit()


def as_utc_datetime(value: datetime | str) -> datetime:
    """Normalise Psycopg timestamps and string timestamps before comparison."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

def create_auth_session(
    conn,
    user_id: int,
) -> str:

    raw_token = secrets.token_urlsafe(48)

    token_hash = hash_session_token(
        raw_token
    )

    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(
            days=AUTH_SESSION_EXPIRE_DAYS
        )
    )

    with conn.cursor() as cur:

        # Cheap opportunistic cleanup.
        cur.execute(
            """
            DELETE FROM auth_sessions
            WHERE expires_at <= CURRENT_TIMESTAMP
               OR revoked = TRUE
            """
        )

        cur.execute(
            """
            INSERT INTO auth_sessions (
                user_id,
                token_hash,
                expires_at
            )
            VALUES (%s, %s, %s)
            """,
            (
                user_id,
                token_hash,
                expires_at,
            ),
        )

    return raw_token

def set_auth_session_cookie(
    response: Response,
    token: str,
) -> None:

    response.set_cookie(
        key=AUTH_SESSION_COOKIE_NAME,
        value=token,
        max_age=(
            AUTH_SESSION_EXPIRE_DAYS
            * 24
            * 60
            * 60
        ),
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        path="/",
    )


def clear_auth_session_cookie(
    response: Response,
) -> None:

    response.delete_cookie(
        key=AUTH_SESSION_COOKIE_NAME,
        path="/",
        secure=AUTH_COOKIE_SECURE,
        httponly=True,
        samesite=AUTH_COOKIE_SAMESITE,
    )
    
def enforce_trusted_origin(
    request: Request,
) -> None:

    origin = request.headers.get(
        "origin"
    )

    if not origin:
        return

    if origin not in CORS_ORIGINS:
        raise HTTPException(
            status_code=403,
            detail="Untrusted request origin.",
        )
        
def rotate_auth_session(
    raw_token: str,
) -> tuple[int, str, str, str]:

    token_hash = hash_session_token(
        raw_token
    )

    with contextlib.closing(
        get_connection()
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    s.id,
                    s.user_id,
                    s.expires_at,
                    s.revoked,
                    u.username,
                    u.email,
                    u.is_verified
                FROM auth_sessions s

                JOIN users u
                    ON u.id = s.user_id

                WHERE s.token_hash = %s

                FOR UPDATE
                """,
                (token_hash,),
            )

            row = cur.fetchone()

            if row is None:
                raise HTTPException(
                    status_code=401,
                    detail="Session expired or invalid.",
                )

            (
                session_id,
                user_id,
                expires_at,
                revoked,
                username,
                email,
                is_verified,
            ) = row

            expiry = as_utc_datetime(
                expires_at
            )

            if (
                revoked
                or not is_verified
                or datetime.now(timezone.utc)
                >= expiry
            ):
                cur.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked = TRUE
                    WHERE id = %s
                    """,
                    (session_id,),
                )

                conn.commit()

                raise HTTPException(
                    status_code=401,
                    detail="Session expired or invalid.",
                )

            # Old refresh token becomes unusable.
            cur.execute(
                """
                UPDATE auth_sessions
                SET
                    revoked = TRUE,
                    last_used_at =
                        CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (session_id,),
            )

            new_token = secrets.token_urlsafe(
                48
            )

            new_hash = hash_session_token(
                new_token
            )

            new_expiry = (
                datetime.now(timezone.utc)
                + timedelta(
                    days=AUTH_SESSION_EXPIRE_DAYS
                )
            )

            cur.execute(
                """
                INSERT INTO auth_sessions (
                    user_id,
                    token_hash,
                    expires_at,
                    last_used_at
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    CURRENT_TIMESTAMP
                )
                """,
                (
                    user_id,
                    new_hash,
                    new_expiry,
                ),
            )

        conn.commit()

    return (
        int(user_id),
        username,
        email,
        new_token,
    )
    
def revoke_auth_session(
    raw_token: str,
) -> None:

    token_hash = hash_session_token(
        raw_token
    )

    with contextlib.closing(
        get_connection()
    ) as conn:

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE auth_sessions
                SET revoked = TRUE
                WHERE token_hash = %s
                """,
                (token_hash,),
            )

        conn.commit()

def normalize_email(value: str) -> str:
    return value.strip().lower()


def is_email_address(value: str) -> bool:
    value = value.strip()
    if "@" not in value:
        return False
    local, _, domain = value.rpartition("@")
    return bool(local and domain and "." in domain)


def _lookup_user_by_email(
    conn,
    raw_email: str,
):
    email = normalize_email(raw_email)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, username, email, password, is_verified
            FROM users
            WHERE LOWER(email) = LOWER(%s)
            """,
            (email,),
        )
        return cur.fetchone()


def lock_user_for_otp(conn, user_id: int) -> None:
    """Serialize OTP replacement for one account within the current transaction."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (user_id,))

def create_access_token(user_id: int, gaash_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "gaash_id": gaash_id,
        "purpose": "access",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_password_reset_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "purpose": "password_reset",
        "iat": now,
        "exp": now + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def store_reset_token(conn, user_id: int, token: str) -> None:
    payload = jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
    )

    expires_at = datetime.fromtimestamp(
        payload["exp"],
        tz=timezone.utc,
    )

    token_hash = hash_reset_token(token)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO reset_tokens
            (user_id, token_hash, expires_at)
            VALUES (%s, %s, %s)
            """,
            (user_id, token_hash, expires_at),
        )

def numeric_gaash_id(gaash_id: str) -> int:
    if not gaash_id.startswith("GSH-"):
        raise HTTPException(status_code=400, detail="Invalid Gaash ID.")
    try:
        return int(gaash_id[4:])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Gaash ID.")


# ============================================================
# OTP DELIVERY
# ============================================================

def _send_email_via_smtp(recipient: str, otp: str, purpose: str) -> None:
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL]):
        raise OTPServiceError("Email service is not configured.")

    subject = (
        "Gaash verification code"
        if purpose == "REGISTRATION"
        else "Gaash password reset code"
    )
    body = _otp_email_text(otp, purpose)

    msg = EmailMessage()
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as exc:
        # No OTP or credential is logged here.
        raise OTPServiceError("Could not send email. Please check SMTP configuration.") from exc


def _send_email_via_resend(recipient: str, otp: str, purpose: str) -> None:
    if not RESEND_API_KEY:
        raise OTPServiceError("Resend is not configured.")

    try:
        import resend
    except ImportError as exc:
        raise OTPServiceError("RESEND_API_KEY is set but the resend package is not installed.") from exc

    resend.api_key = RESEND_API_KEY
    subject = (
        "Gaash verification code"
        if purpose == "REGISTRATION"
        else "Gaash password reset code"
    )
    safe_otp = html.escape(otp)
    body_text = _otp_email_text(otp, purpose)

    params = {
        "from": RESEND_FROM_EMAIL,
        "to": [recipient],
        "subject": subject,
        "text": body_text,
        "html": (
            f"<div style='font-family:Arial,sans-serif;max-width:560px;margin:auto;line-height:1.6'>"
            f"<p>Your Gaash code is:</p>"
            f"<p style='font-size:32px;font-weight:700;letter-spacing:6px;margin:24px 0'>{safe_otp}</p>"
            f"<p>This code expires in {OTP_EXPIRY_MINUTES} minutes. Do not share it.</p>"
            f"</div>"
        ),
    }

    try:
        response = resend.Emails.send(params)
    except Exception as exc:
        logger.warning("Resend delivery failed: %s", type(exc).__name__)
        raise OTPServiceError("Could not send email via Resend.") from exc

    email_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
    if not email_id:
        raise OTPServiceError("Resend did not return an email ID.")


def _otp_email_text(otp: str, purpose: str) -> str:
    return (
        f"Your Gaash {purpose.replace('_', ' ').lower()} code is: {otp}\n\n"
        f"This code expires in {OTP_EXPIRY_MINUTES} minutes. "
        "Do not share this code with anyone."
    )


def send_email_otp(recipient: str, otp: str, purpose: str) -> None:
    if RESEND_API_KEY:
        _send_email_via_resend(recipient, otp, purpose)
    else:
        _send_email_via_smtp(recipient, otp, purpose)

def send_otp(email: str, otp: str, purpose: str) -> None:
    send_email_otp(email, otp, purpose)


def store_otp(
    conn,
    user_id: int,
    email: str,
    purpose: str,
    otp: str,
) -> None:
    otp_hash = hash_otp(otp)

    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(minutes=OTP_EXPIRY_MINUTES)
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO otp_verifications
            (user_id, email, purpose, otp_hash, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                user_id,
                email,
                purpose,
                otp_hash,
                expires_at,
            ),
        )

def verify_and_consume_otp(
    conn,
    user_id: int,
    email: str,
    purpose: str,
    otp: str,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, otp_hash, expires_at, attempts, used
            FROM otp_verifications
            WHERE user_id = %s
              AND LOWER(email) = LOWER(%s)
              AND purpose = %s
            ORDER BY id DESC
            LIMIT 1
            FOR UPDATE
            """,
            (user_id, email, purpose),
        )

        row = cur.fetchone()

        if row is None:
            return False

        otp_id, stored_hash, expiry, attempts, used = row

        if used:
            return False

        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) >= expiry:
            cur.execute(
                """
                UPDATE otp_verifications
                SET used = TRUE
                WHERE id = %s
                """,
                (otp_id,),
            )
            return False

        if attempts >= MAX_OTP_ATTEMPTS:
            cur.execute(
                """
                UPDATE otp_verifications
                SET used = TRUE
                WHERE id = %s
                """,
                (otp_id,),
            )
            return False

        cur.execute(
            """
            UPDATE otp_verifications
            SET attempts = attempts + 1
            WHERE id = %s
            """,
            (otp_id,),
        )

        if not hmac.compare_digest(
            stored_hash,
            hash_otp(otp),
        ):
            return False

        cur.execute(
            """
            UPDATE otp_verifications
            SET used = TRUE
            WHERE id = %s
            """,
            (otp_id,),
        )

    return True


# ============================================================
# AUTHENTICATION DEPENDENCY
# ============================================================

def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer access token required.",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        if payload.get("purpose") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired access token.",
            )
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        )

    if is_access_token_revoked(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        )
    return user_id


@dataclass(frozen=True)
class ResetTokenContext:
    user_id: int
    token_hash: str


def get_reset_token_context(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> ResetTokenContext:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password reset token required.",
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        if payload.get("purpose") != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired reset token.",
        )

    token_hash = hash_reset_token(credentials.credentials)
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, used, expires_at FROM reset_tokens WHERE token_hash = %s FOR UPDATE",
            (token_hash,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )

        record_id, stored_user_id, used, expires_at = row
        if stored_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )
        if used:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )

        try:
            expiry = as_utc_datetime(expires_at)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )

        if datetime.now(timezone.utc) >= expiry:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )

    return ResetTokenContext(user_id=user_id, token_hash=token_hash)


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=8, max_length=100)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


class RequestOTPRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=100)


class VerifyRegistrationRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=100)
    otp: str = Field(..., min_length=6, max_length=6)


class VerifyResetOTPRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=100)
    otp: str = Field(..., min_length=6, max_length=6)


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=8, max_length=100)
    
class PrivacyAcceptRequest(BaseModel):
    policy_version: str = Field(..., min_length=1, max_length=64)
    locale: Optional[str] = Field(default=None, min_length=2, max_length=16)


class VoiceTranscriptionConsentRequest(BaseModel):
    granted: bool
    locale: Optional[str] = Field(default=None, min_length=2, max_length=16)


class PrivacyPolicyMetadataResponse(BaseModel):
    policy_version: str
    effective_date: str


class VoiceTranscriptionConsentResponse(BaseModel):
    granted: bool
    recorded_at: Optional[datetime] = None
    policy_version: Optional[str] = None


class PrivacyStatusResponse(PrivacyPolicyMetadataResponse):
    accepted: bool
    accepted_at: Optional[datetime] = None
    voice_transcription: VoiceTranscriptionConsentResponse


class AuthMessageResponse(BaseModel):
    message: str
    status: Optional[str] = None


class SafeUserResponse(BaseModel):
    user_id: int
    gaash_id: str
    username: str
    email: str


class AuthTokenResponse(BaseModel):
    message: str
    access_token: str
    token_type: str
    user: SafeUserResponse
    next_step: str


class ResetTokenResponse(BaseModel):
    message: str
    reset_token: str


class CurrentUserResponse(SafeUserResponse):
    created_at: datetime


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Gaash Authentication API is running",
        "status": "active",
        "port": PORT,
    }


@app.get("/health")
async def health():
    try:
        db_ok = await asyncio.wait_for(
            asyncio.to_thread(_health_db_sync),
            timeout=5.0,
        )
    except Exception as exc:
        logger.warning(
            "Auth health check database failure: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "service": "gaash-auth",
                "database": "unavailable",
            },
        ) from exc

    return {
        "status": "ok",
        "service": "gaash-auth",
        "database": "ok" if db_ok else "unavailable",
    }

@app.get("/privacy/policy", response_model=PrivacyPolicyMetadataResponse)
def get_privacy_policy_metadata():
    """Public metadata lets the static notice show the server's current version."""
    return {
        "policy_version": current_policy_version(),
        "effective_date": current_policy_effective_date(),
    }


@app.get("/privacy/status", response_model=PrivacyStatusResponse)
def get_privacy_status(user_id: int = Depends(get_current_user_id)):
    with db_connection() as conn:
        acknowledgement = get_privacy_acknowledgement(conn, user_id)
        voice_consent = get_voice_transcription_consent(conn, user_id)
    return {
        "policy_version": current_policy_version(),
        "effective_date": current_policy_effective_date(),
        "accepted": acknowledgement is not None,
        "accepted_at": acknowledgement["accepted_at"] if acknowledgement else None,
        "voice_transcription": voice_consent,
    }


@app.post("/privacy/accept", response_model=PrivacyStatusResponse)
def accept_privacy_notice(data: PrivacyAcceptRequest, user_id: int = Depends(get_current_user_id)):
    try:
        with db_connection() as conn:
            acknowledgement = record_privacy_acknowledgement(
                conn,
                user_id=user_id,
                policy_version=data.policy_version.strip(),
                locale=data.locale.strip() if data.locale else None,
            )
            voice_consent = get_voice_transcription_consent(conn, user_id)
            conn.commit()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {
        "policy_version": current_policy_version(),
        "effective_date": current_policy_effective_date(),
        "accepted": True,
        "accepted_at": acknowledgement["accepted_at"],
        "voice_transcription": voice_consent,
    }


@app.post("/privacy/voice-transcription-consent", response_model=VoiceTranscriptionConsentResponse)
def set_voice_transcription_consent(data: VoiceTranscriptionConsentRequest, user_id: int = Depends(get_current_user_id)):
    """Record a separately withdrawable choice for third-party audio transcription."""
    with db_connection() as conn:
        if get_privacy_acknowledgement(conn, user_id) is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Privacy notice acknowledgement required.",
            )
        consent = record_voice_transcription_consent(
            conn,
            user_id=user_id,
            granted=data.granted,
            locale=data.locale.strip() if data.locale else None,
        )
        conn.commit()
    return consent


@app.post("/auth/register", response_model=AuthMessageResponse)
def register(data: RegisterRequest, request: Request):
    enforce_rate_limit(request, "register", limit=5, window_seconds=600)
    username = data.username.strip()
    email = normalize_email(data.email)

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must contain at least 3 characters.")
    if not is_email_address(email):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid email address.",
        )

    with db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM users WHERE LOWER(username) = LOWER(%s)",
            (username,),
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail="Username is already taken.",
            )

        cursor.execute(
            "SELECT id FROM users WHERE LOWER(email) = LOWER(%s)",
            (email,),
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=409,
                detail="Email is already registered.",
            )

        try:
            cursor.execute(
                """
                INSERT INTO users (username, email, password, is_verified)
                VALUES (%s, %s, %s, FALSE)
                RETURNING id
                """,
                (
                    username,
                    email,
                    hash_password(data.password),
                ),
            )
        except pg_errors.UniqueViolation as exc:
            conn.rollback()
            raise HTTPException(status_code=409, detail="Username or email is already registered.") from exc

        user_id = cursor.fetchone()[0]

        cursor.execute(
            """
            UPDATE otp_verifications
            SET used = TRUE
            WHERE user_id = %s AND purpose = 'REGISTRATION' AND used = FALSE
            """,
            (user_id,),
        )

        otp = generate_otp()
        store_otp(
            conn,
            user_id,
            email,
            "REGISTRATION",
            otp,
        )

        try:
            send_otp(email, otp, "REGISTRATION")
        except OTPServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not send verification code.",
            ) from exc

        conn.commit()

    return {
        "status": "verification_required",
        "message": "A verification code has been sent to your email.",
    }



@app.post("/auth/verify-registration", response_model=AuthMessageResponse)
def verify_registration(data: VerifyRegistrationRequest, request: Request):
    enforce_rate_limit(request, "verify-registration", limit=10, window_seconds=600)
    email = normalize_email(data.email)
    if not is_email_address(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    with db_connection() as conn:
        user = _lookup_user_by_email(conn, data.email)

        if user is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired verification code.",
            )

        user_id = user[0]
        email = user[2]

        lock_user_for_otp(conn, user_id)

        if user[4]:
            raise HTTPException(
                status_code=400,
                detail="Account is already verified.",
            )

        verified = verify_and_consume_otp(
            conn,
            user_id,
            email,
            "REGISTRATION",
            data.otp,
        )
        if not verified:
            conn.commit()
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired verification code.",
            )

        conn.execute(
            "UPDATE users SET is_verified = TRUE WHERE id = %s",
            (user_id,),
        )
        conn.commit()

    return {
        "message": "Registration complete. You may now log in.",
    }


@app.post("/auth/resend-otp", response_model=AuthMessageResponse)
def resend_otp(data: RequestOTPRequest, request: Request):
    enforce_rate_limit(request, "resend-registration-otp", limit=3, window_seconds=600)
    email = normalize_email(data.email)
    generic_response = {
        "message": "If your account is eligible, a new verification code has been sent."
    }

    with db_connection() as conn:
        user = _lookup_user_by_email(conn, data.email)

        if user is None or user[4]:
            return generic_response

        user_id = user[0]
        email = user[2]

        lock_user_for_otp(conn, user_id)

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at
            FROM otp_verifications
            WHERE user_id = %s
              AND purpose = 'REGISTRATION'
              AND used = FALSE
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        )

        row = cursor.fetchone()

        if row:
            last_sent = as_utc_datetime(row[0])

            elapsed = (
                datetime.now(timezone.utc) - last_sent
            ).total_seconds()

            if elapsed < OTP_RESEND_COOLDOWN_SECONDS:
                raise HTTPException(
                    status_code=429,
                    detail="Please wait before requesting a new code.",
                )

        cursor.execute(
            """
            UPDATE otp_verifications
            SET used = TRUE
            WHERE user_id = %s
               AND purpose = 'REGISTRATION'
               AND used = FALSE
            """,
            (user_id,),
        )

        otp = generate_otp()
        store_otp(conn, user_id, email, "REGISTRATION", otp)

        try:
            send_otp(email, otp, "REGISTRATION")
        except OTPServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail="Could not send verification code.",
            ) from exc

        conn.commit()

    return generic_response


@app.post(
    "/auth/login",
    response_model=AuthTokenResponse,
)
def login(
    data: LoginRequest,
    request: Request,
    response: Response,
):
    enforce_rate_limit(
        request,
        "login",
        limit=10,
        window_seconds=600,
    )

    email = normalize_email(
        data.email
    )

    if not is_email_address(email):
        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide a valid "
                "email address."
            ),
        )

    with contextlib.closing(
        get_connection()
    ) as conn:

        user = _lookup_user_by_email(
            conn,
            email,
        )

        if (
            user is None
            or not verify_password(
                data.password,
                user[3],
            )
        ):
            raise HTTPException(
                status_code=
                    status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Invalid credentials or "
                    "account not verified."
                ),
            )

        (
            user_id,
            username,
            stored_email,
            _,
            is_verified,
        ) = user

        if not is_verified:
            raise HTTPException(
                status_code=
                    status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Invalid credentials or "
                    "account not verified."
                ),
            )

        refresh_token = create_auth_session(
            conn,
            user_id,
        )

        conn.commit()

    gaash_id = f"GSH-{user_id}"

    access_token = create_access_token(
        user_id,
        gaash_id,
    )

    set_auth_session_cookie(
        response,
        refresh_token,
    )

    return {
        "message": "Login successful.",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user_id,
            "gaash_id": gaash_id,
            "username": username,
            "email": stored_email,
        },
        "next_step": "dashboard",
    }

@app.post(
    "/auth/refresh",
    response_model=AuthTokenResponse,
)
def refresh_session(
    request: Request,
    response: Response,
):
    enforce_rate_limit(
        request,
        "refresh-session",
        limit=30,
        window_seconds=600,
    )

    enforce_trusted_origin(
        request
    )

    refresh_token = (
        request.cookies.get(
            AUTH_SESSION_COOKIE_NAME
        )
    )

    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="No active session.",
        )

    try:
        (
            user_id,
            username,
            email,
            new_refresh_token,
        ) = rotate_auth_session(
            refresh_token
        )

    except HTTPException:
        clear_auth_session_cookie(
            response
        )
        raise

    gaash_id = f"GSH-{user_id}"

    access_token = create_access_token(
        user_id,
        gaash_id,
    )

    set_auth_session_cookie(
        response,
        new_refresh_token,
    )

    return {
        "message": "Session refreshed.",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user_id,
            "gaash_id": gaash_id,
            "username": username,
            "email": email,
        },
        "next_step": "dashboard",
    }

@app.post("/auth/forgot-password", response_model=AuthMessageResponse)
def forgot_password(data: RequestOTPRequest, request: Request):
    enforce_rate_limit(request, "forgot-password", limit=3, window_seconds=600)
    generic_response = {
        "message": "If an account exists for this email, a verification code has been sent to that email."
    }

    with db_connection() as conn:
        user = _lookup_user_by_email(conn, data.email)

        if user is None or not user[4]:
            return generic_response

        user_id, registered_email = user[0], user[2]

        lock_user_for_otp(conn, user_id)

        conn.execute(
            """
            UPDATE otp_verifications
            SET used = TRUE
            WHERE user_id = %s
            AND purpose = 'PASSWORD_RESET'
            AND used = FALSE
            """,
            (user_id,),
        )

        otp = generate_otp()

        store_otp(
            conn,
            user_id,
            registered_email,
            "PASSWORD_RESET",
            otp,
        )

        try:
            send_otp(
                registered_email,
                otp,
                "PASSWORD_RESET",
            )
        except OTPServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not send verification code.",
            ) from exc

        conn.commit()

    return generic_response


@app.post("/auth/verify-reset-otp", response_model=ResetTokenResponse)
def verify_reset_otp(data: VerifyResetOTPRequest, request: Request):
    enforce_rate_limit(request, "verify-reset-otp", limit=10, window_seconds=600)
    email = normalize_email(data.email)
    if not is_email_address(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    with db_connection() as conn:
        user = _lookup_user_by_email(conn, data.email)
        if user is None or not user[4]:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired reset code.",
            )

        user_id, registered_contact = user[0], user[2]
        lock_user_for_otp(conn, user_id)
        verified = verify_and_consume_otp(conn, user_id, registered_contact, "PASSWORD_RESET", data.otp)
        if not verified:
            conn.commit()
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired reset code.",
            )

        # A new reset grant supersedes any previously issued reset credential.
        conn.execute(
            "UPDATE reset_tokens SET used = TRUE WHERE user_id = %s AND used = FALSE",
            (user_id,),
        )
        reset_token = create_password_reset_token(user_id)
        store_reset_token(conn, user_id, reset_token)
        conn.commit()

    return {
        "message": "Verification successful.",
        "reset_token": reset_token,
    }


@app.post("/auth/reset-password", response_model=AuthMessageResponse)
def reset_password(
    data: ResetPasswordRequest,
    request: Request,
    reset_context: ResetTokenContext = Depends(get_reset_token_context),
):
    enforce_rate_limit(request, "reset-password", limit=5, window_seconds=600)
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                FROM reset_tokens
                WHERE token_hash = %s
                  AND user_id = %s
                  AND used = FALSE
                  AND expires_at > CURRENT_TIMESTAMP
                FOR UPDATE
                """,
                (reset_context.token_hash, reset_context.user_id),
            )
            if cur.fetchone() is None:
                conn.rollback()
                raise HTTPException(status_code=401, detail="Invalid or expired reset token.")
            cur.execute(
                "UPDATE users SET password = %s WHERE id = %s",
                (hash_password(data.new_password), reset_context.user_id),
            )
            cur.execute(
                """
                UPDATE auth_sessions
                SET revoked = TRUE
                WHERE user_id = %s
                  AND revoked = FALSE
                """,
                (
                    reset_context.user_id,
                ),
            )
            cur.execute(
                """
                UPDATE reset_tokens
                SET used = TRUE
                WHERE token_hash = %s AND user_id = %s AND used = FALSE
                """,
                (reset_context.token_hash, reset_context.user_id),
            )
            if cur.rowcount != 1:
                conn.rollback()
                raise HTTPException(status_code=401, detail="Invalid or expired reset token.")
            # A successful reset makes every outstanding reset credential for
            # the account unusable, not only the one presented here.
            cur.execute(
                "UPDATE reset_tokens SET used = TRUE WHERE user_id = %s AND used = FALSE",
                (reset_context.user_id,),
            )
        conn.commit()

    return {
        "message": "Password reset successfully. You can now sign in.",
    }


@app.post(
    "/auth/logout",
    response_model=AuthMessageResponse,
)
def logout(
    request: Request,
    response: Response,
    credentials:
        HTTPAuthorizationCredentials
        = Depends(security),
    _: int = Depends(
        get_current_user_id
    ),
):
    enforce_rate_limit(
        request,
        "logout",
        limit=20,
        window_seconds=600,
    )

    enforce_trusted_origin(
        request
    )

    # Kill current access token.
    revoke_access_token(
        credentials.credentials
    )

    # Kill persistent browser session.
    refresh_token = (
        request.cookies.get(
            AUTH_SESSION_COOKIE_NAME
        )
    )

    if refresh_token:
        revoke_auth_session(
            refresh_token
        )

    clear_auth_session_cookie(
        response
    )

    return {
        "message": "Signed out."
    }


@app.get("/me", response_model=CurrentUserResponse)
def get_current_user(user_id: int = Depends(get_current_user_id)):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, email, created_at
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        )
        user = cursor.fetchone()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "user_id": user[0],
        "gaash_id": f"GSH-{user[0]}",
        "username": user[1],
        "email": user[2],
        "created_at": user[3],
    }


@app.get("/user/{gaash_id}", response_model=CurrentUserResponse)
def get_user(
    gaash_id: str,
    user_id: int = Depends(get_current_user_id),
):
    target_id = numeric_gaash_id(gaash_id)

    if target_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot access another user's account.",
        )

    return get_current_user(user_id)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
