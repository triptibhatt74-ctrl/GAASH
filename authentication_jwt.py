import contextlib
import hashlib
import hmac
import html
import os

import secrets
import smtplib
import psycopg
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import jwt

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(
    title="Gaash Authentication API",
    description="Authentication and account-management backend for Gaash",
    version="2.0.0",
)

# AUTH_HOST/AUTH_PORT let both services share one .env without colliding with bot.py's HOST/PORT.
HOST = os.getenv("AUTH_HOST") or os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("AUTH_PORT") or os.getenv("PORT", "8004"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gaashai_db_user:n5zReAPcVTqeNzzbbt7MLwcw0giuJEZk@dpg-da4k1sk9v7es738e8450-a.ohio-postgres.render.com/gaashai_db").strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

JWT_SECRET = os.getenv("GAASH_JWT_SECRET", "CHANGE_THIS_IN_PRODUCTION")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


class OTPServiceError(Exception):
    """Raised when an OTP cannot be delivered because the provider is not configured."""


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg.connect(DATABASE_URL)


def _ensure_columns(conn: psycopg.Connection, table: str, columns: dict) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_db() -> None:
    with contextlib.closing(get_connection()) as conn:
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

        conn.commit()


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

    conn.commit()


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
        print("RESEND ERROR:", repr(exc))
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
            conn.commit()
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
            conn.commit()
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
            conn.commit()
            return False

        cur.execute(
            """
            UPDATE otp_verifications
            SET used = TRUE
            WHERE id = %s
            """,
            (otp_id,),
        )

    conn.commit()
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

    return user_id


def get_reset_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
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
    with contextlib.closing(get_connection()) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, used, expires_at FROM reset_tokens WHERE token_hash = %s",
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
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )

        if datetime.now(timezone.utc) >= expiry:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired reset token.",
            )

        cur.execute(
            "UPDATE reset_tokens SET used = True WHERE id = %s",
            (record_id,),
        )
        conn.commit()

    return user_id


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
def health():
    return {"status": "ok", "service": "Gaash Authentication API"}


@app.post("/auth/register")
def register(data: RegisterRequest):
    username = data.username.strip()
    email = normalize_email(data.email)

    if not is_email_address(email):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid email address.",
        )

    with contextlib.closing(get_connection()) as conn:
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

        cursor.execute(
            """
            INSERT INTO users (username, email, password, is_verified)
            VALUES (%s, %s, %s, False)
            """,
            (
                username,
                email,
                hash_password(data.password),
            ),
        )

        user_id = cursor.fetchone()[0]

        cursor.execute(
            """
            UPDATE otp_verifications
            SET used = True
            WHERE user_id = %s AND purpose = 'REGISTRATION' AND used = False
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



@app.post("/auth/verify-registration")
def verify_registration(data: VerifyRegistrationRequest):
    email = normalize_email(data.email)
    if not is_email_address(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    with contextlib.closing(get_connection()) as conn:
        user = _lookup_user_by_email(conn, data.email)

        if user is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired verification code.",
            )

        user_id = user[0]
        email = user[2]

        if user[4]:
            raise HTTPException(
                status_code=400,
                detail="Account is already verified.",
            )

        if not verify_and_consume_otp(
            conn,
            user_id,
            email,
            "REGISTRATION",
            data.otp,
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired verification code.",
            )

        conn.execute(
            "UPDATE users SET is_verified = True WHERE id = %s",
            (user_id,),
        )
        conn.commit()

    return {
        "message": "Registration complete. You may now log in.",
    }


@app.post("/auth/resend-otp")
def resend_otp(data: RequestOTPRequest):
    email = normalize_email(data.email)
    generic_response = {
        "message": "If your account is eligible, a new verification code has been sent."
    }

    with contextlib.closing(get_connection()) as conn:
        user = _lookup_user_by_email(conn, data.email)

        if user is None or user[4]:
            return generic_response

        user_id = user[0]
        email = user[2]

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at
            FROM otp_verifications
            WHERE user_id = %s
              AND purpose = 'REGISTRATION'
              AND used = False
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        )

        row = cursor.fetchone()

        if row:
            last_sent = datetime.fromisoformat(row[0])
            if last_sent.tzinfo is None:
                last_sent = last_sent.replace(tzinfo=timezone.utc)

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
            SET used = 1
            WHERE user_id = %s
              AND purpose = 'REGISTRATION'
              AND used = 0
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


@app.post("/auth/login")
def login(data: LoginRequest):
    email = normalize_email(data.email)
    if not is_email_address(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    with contextlib.closing(get_connection()) as conn:
        user = _lookup_user_by_email(conn, email)

    if user is None or not verify_password(data.password, user[3]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or account not verified.",
        )

    user_id, username, stored_identifier, _, is_verified = user[0], user[1], user[2], user[3], user[4]
    if not is_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or account not verified.",
        )

    gaash_id = f"GSH-{user_id}"
    token = create_access_token(user_id, gaash_id)

    return {
        "message": "Login successful.",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user_id,
            "gaash_id": gaash_id,
            "username": username,
            "email": stored_identifier,
        },
        "next_step": "dashboard",
    }


@app.post("/auth/forgot-password")
def forgot_password(data: RequestOTPRequest):
    generic_response = {
        "message": "If an account exists for this email, a verification code has been sent to that email."
    }

    with contextlib.closing(get_connection()) as conn:
        user = _lookup_user_by_email(conn, data.email)

        if user is None or not user[4]:
            return generic_response

        user_id, registered_email = user[0], user[2]

        conn.execute(
            "UPDATE otp_verifications SET used = True"
            "WHERE user_id = %s AND purpose = 'PASSWORD_RESET' AND used = False",
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


@app.post("/auth/verify-reset-otp")
def verify_reset_otp(data: VerifyResetOTPRequest):
    email = normalize_email(data.email)
    if not is_email_address(email):
        raise HTTPException(status_code=400, detail="Please provide a valid email address.")

    with contextlib.closing(get_connection()) as conn:
        user = _lookup_user_by_email(conn, data.email)
        if user is None or not user[4]:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired reset code.",
            )

        user_id, registered_contact = user[0], user[2]
        if not verify_and_consume_otp(conn, user_id, registered_contact, "PASSWORD_RESET", data.otp):
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired reset code.",
            )

        reset_token = create_password_reset_token(user_id)
        store_reset_token(conn, user_id, reset_token)

    return {
        "message": "Verification successful.",
        "reset_token": reset_token,
    }


@app.post("/auth/reset-password")
def reset_password(
    data: ResetPasswordRequest,
    user_id: int = Depends(get_reset_user_id),
):
    with contextlib.closing(get_connection()) as conn:
        conn.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (hash_password(data.new_password), user_id),
        )
        conn.commit()

    return {
        "message": "Password reset successfully. You can now sign in.",
    }


@app.get("/me")
def get_current_user(user_id: int = Depends(get_current_user_id)):
    with contextlib.closing(get_connection()) as conn:
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


@app.get("/user/{gaash_id}")
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

init_db()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
