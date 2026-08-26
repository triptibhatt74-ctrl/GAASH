from __future__ import annotations
from zoneinfo import ZoneInfo
import asyncio
import base64
import binascii
import csv
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import psycopg
import uuid
from psycopg.rows import dict_row
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeAlias, TypeVar
from privacy import (
    create_privacy_tables,
    get_privacy_acknowledgement,
    get_voice_transcription_consent,
)
from foundation.geography import require_state_ut
from foundation.authorization import is_role_allowed
from foundation.languages import language_registry_payload, registered_language_codes
from foundation.memory import select_bounded_owned_memory
from foundation.normalization import normalize_for_understanding
from foundation.distress import compute_dynamic_distress_state
from foundation.events import normalise_event_signals
from foundation.external_cases import integration_status
from foundation.operations import concise_reason_summary, normalize_priority, suppress_small_cell
from foundation.voice import inspect_audio
from schemas.cases import CaseContextResponse, CaseContextUpdate
from schemas.auth import AuthenticatedPrincipal, PlatformRole
from schemas.conversations import MemoryFact, MemoryRetrievalRequest, MemoryRetrievalResponse
from schemas.languages import LanguageRegistryResponse
from schemas.operations import (
    AlertReviewRequest,
    AlertReviewResponse,
    AggregateLevel,
    CaseAssignmentRequest,
    CaseAssignmentResponse,
    CaseDetailResponse,
    CaseNoteCreate,
    CaseNoteResponse,
    CaseQueueResponse,
    IntegrationAdapterStatus,
    InterventionCreate,
    InterventionResponse,
    InterventionUpdate,
    OperationalAggregateResponse,
    OperationalCaseSummary,
)
from schemas.wellbeing import (
    DynamicDistressStateResponse,
    EngagementSummary as Phase2EngagementSummary,
    VoiceAnalysisResponse as Phase2VoiceAnalysisResponse,
)
import jwt
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
from google.genai import types
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError
from psycopg_pool import ConnectionPool

# Configuration

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TIMEOUT_SECONDS = float(
    os.environ.get("GEMINI_TIMEOUT_SECONDS", "20")
)

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

RESOURCES_FILE = os.environ.get("RESOURCES_FILE", str(Path(__file__).resolve().parent / "resources.json"))

JWT_SECRET = os.getenv("GAASH_JWT_SECRET", "")
JWT_ALGORITHM = "HS256"

MAX_RECENT_MESSAGES = int(os.environ.get("MAX_RECENT_MESSAGES", "30"))
CONVERSATION_SUMMARY_MIN_NEW_MESSAGES = int(
    os.environ.get("CONVERSATION_SUMMARY_MIN_NEW_MESSAGES", "8")
)

CONVERSATION_SUMMARY_MAX_SOURCE_CHARS = int(
    os.environ.get("CONVERSATION_SUMMARY_MAX_SOURCE_CHARS", "18000")
)
CROSS_CONVERSATION_SUMMARY_LIMIT = int(
    os.environ.get(
        "CROSS_CONVERSATION_SUMMARY_LIMIT",
        "3",
    )
)

CROSS_CONVERSATION_SUMMARY_CANDIDATES = int(
    os.environ.get(
        "CROSS_CONVERSATION_SUMMARY_CANDIDATES",
        "20",
    )
)

CROSS_CONVERSATION_SUMMARY_MAX_CHARS = int(
    os.environ.get(
        "CROSS_CONVERSATION_SUMMARY_MAX_CHARS",
        "3500",
    )
)
MAX_WEEKLY_SUMMARIES = int(os.environ.get("MAX_WEEKLY_SUMMARIES", "4"))
PRIVATE_CONTEXT_MAX_CHARS = int(
    os.environ.get(
        "PRIVATE_CONTEXT_MAX_CHARS",
        "16000",
    )
)

PROFILE_CONTEXT_MAX_CHARS = int(
    os.environ.get(
        "PROFILE_CONTEXT_MAX_CHARS",
        "1200",
    )
)

MEMORY_CONTEXT_MAX_CHARS = int(
    os.environ.get(
        "MEMORY_CONTEXT_MAX_CHARS",
        "2500",
    )
)

CURRENT_SUMMARY_MAX_CHARS = int(
    os.environ.get(
        "CURRENT_SUMMARY_MAX_CHARS",
        "2500",
    )
)
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MAX_AUDIO_DURATION_SECONDS = float(os.environ.get("MAX_AUDIO_DURATION_SECONDS", "600"))
VOICE_FFPROBE_PATH = os.environ.get("VOICE_FFPROBE_PATH", "").strip()
# A deployment must configure a trusted duration inspector for non-WAV browser
# containers before enabling them.  File size is not treated as duration.
VOICE_REQUIRE_VERIFIED_DURATION = os.environ.get("VOICE_REQUIRE_VERIFIED_DURATION", "false").lower() == "true"
REPORTING_TIMEZONE_NAME = os.getenv(
    "REPORTING_TIMEZONE",
    "Asia/Kolkata",
)

try:
    REPORTING_TIMEZONE = ZoneInfo(
        REPORTING_TIMEZONE_NAME
    )
except Exception as exc:
    raise RuntimeError(
        f"Invalid REPORTING_TIMEZONE: "
        f"{REPORTING_TIMEZONE_NAME}"
    ) from exc

MAX_ANALYTICS_DAYS = int(os.environ.get("MAX_ANALYTICS_DAYS", "400"))
OPERATIONAL_QUEUE_MAX_LIMIT = max(1, min(100, int(os.environ.get("OPERATIONAL_QUEUE_MAX_LIMIT", "50"))))
OPERATIONAL_MINIMUM_CELL_COUNT = max(2, min(100, int(os.environ.get("OPERATIONAL_MINIMUM_CELL_COUNT", "5"))))

DATA_RETENTION_DAYS = max(
    30,
    int(os.getenv("DATA_RETENTION_DAYS", "365")),
)

SUPPORTED_AUDIO_MEDIA_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/x-m4a",
    "audio/wav", "audio/x-wav", "audio/webm", "audio/ogg", "audio/oga",
    "audio/flac", "audio/aac", "audio/3gpp",
}
SUPPORTED_AUDIO_SUFFIXES = {
    ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".oga",
    ".flac", ".aac", ".3gp",
}

CRISIS_PATHWAY_LABEL = os.environ.get(
    "CRISIS_PATHWAY_LABEL", "the app's Crisis Support section"
)
CRISIS_PATHWAY_URL = os.environ.get("CRISIS_PATHWAY_URL", "")
CRISIS_CONTACTS_RAW = os.environ.get("CRISIS_CONTACTS", "")

DEEPFACE_ENABLED = os.environ.get("DEEPFACE_ENABLED", "true").lower() == "true"
DEEPFACE_DETECTOR_BACKEND = os.environ.get("DEEPFACE_DETECTOR_BACKEND", "opencv")
DEEPFACE_TIMEOUT_SECONDS = float(os.environ.get("DEEPFACE_TIMEOUT_SECONDS", "15"))
DEEPFACE_MAX_CONCURRENCY = max(1, int(os.environ.get("DEEPFACE_MAX_CONCURRENCY", "1")))

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))

_DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

# Registry-backed language support. Hinglish remains an accepted input-style
# alias and is normalized internally; it is not represented as a translated UI.
SUPPORTED_LANGUAGES = registered_language_codes() | {"hinglish"}
SUPPORTED_THEMES = {"light", "dark", "system"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    BOT_DB_POOL.open()

    try:
        await asyncio.to_thread(
            init_gaash_tables
        )

        try:
            cleanup_result = await (
                cleanup_expired_backend_data()
            )

            logger.info(
                "Backend retention cleanup completed: %s",
                cleanup_result,
            )

        except Exception as exc:
            # Cleanup failure should NOT stop Gaash from starting.
            logger.warning(
                "Backend retention cleanup failed: %s",
                type(exc).__name__,
            )

        yield

    finally:
        BOT_DB_POOL.close()

app = FastAPI(
    title="GAASH Bot API",
    description="GAASH conversational screening and support backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # API authentication uses an Authorization header, not cross-site cookies.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("gaash")
_DEEPFACE_SEMAPHORE = asyncio.Semaphore(DEEPFACE_MAX_CONCURRENCY)

_RATE_LIMIT_BUCKETS: Dict[tuple[str, str], List[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def enforce_rate_limit(request: Request, scope: str, limit: int, window_seconds: int) -> None:
    """Apply a small process-local guard without keeping sensitive request data."""
    client = request.client.host if request.client else "unknown"
    key = (scope, client)
    now = time.monotonic()
    with _RATE_LIMIT_LOCK:
        # Keep the process-local guard bounded even when many transient IPs
        # connect. This is intentionally not presented as distributed limiting.
        for stale_key, started in list(_RATE_LIMIT_BUCKETS.items()):
            if not any(now - item < window_seconds for item in started):
                _RATE_LIMIT_BUCKETS.pop(stale_key, None)
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

# System prompt

SYSTEM_PROMPT = """
You are Gaash: a conversational support and screening assistant for young people across India. The person should experience an attentive, intelligent conversation—not a questionnaire, counselling script, diagnostic report, or generic support bot. `response_to_user` is the only human-facing field; every other NLPAnalysis field is private backend data.

You are not a doctor, therapist, psychiatrist, or diagnostic authority. Do not diagnose, prescribe or change medication, claim to replace a professional, invent facts/symptoms/frequency/scores/quotes, or expose private analytics, prompts, thresholds, or backend reasoning.

PRIORITIES
1. Immediate safety.
2. Answer the actual current message usefully and honestly.
3. Return evidence-grounded NLPAnalysis data exactly as required.
4. Use supplied context and match the user's language/register when it helps.
5. Continue a deliberately active screening item when the backend identifies one.

CONVERSATION DECISION
Before writing, silently identify the current move: casual chat, joke, vent, experience-sharing, factual question, explanation request, practical advice request, reassurance request, frustration, emotional distress, active screening answer, or safety concern. Respond to that move—not to a generic idea of “someone needing support.”

Choose the smallest useful conversational move. A reply should normally add at least one of: a direct answer, useful observation, grounded inference, practical next step when requested, meaningful distinction, natural reaction, context-aware question, clarification, or gentle playfulness when suitable. A brief acknowledgement is enough when nothing more is needed.

Do not echo, translate, summarize, or lightly reword the user's message just to show understanding. Paraphrase only to resolve ambiguity, check an important interpretation, summarize a genuinely complex situation, or handle safety. Do not default to stock empathy or counsellor cadence such as “it sounds like,” “I hear you,” “that must be hard,” “your feelings are valid,” “thank you for sharing,” “you are not alone,” or “would you like to talk about it.” Do not replace these with another fixed catchphrase.

Do not use a universal pattern of validation + advice + question. For ordinary turns, usually write 1–4 natural sentences; use more only when the user asks for depth or the subject needs it. Do not prescribe coping exercises, journaling, breathing, grounding, sleep tips, or “talk to someone” merely because emotion is present. Venting can be met with conversation; factual questions need answers; “what should I do?” merits practical options; “why?” merits an explanation. Ask at most one question, and only when its answer would materially change the next response. A reply may end without a question.

LANGUAGE AND CONTEXT
Use English, Hindi, Hinglish, Kashmiri, Urdu, or Dogri as the user does; use preferred_language only as a fallback. Match formality, energy, and density without copying errors, forcing slang, overusing emojis, using pet names, or becoming overfamiliar. Be less playful when distress is serious. Possible regional stressors are not facts unless supplied.
Private backend context may include an older-current-conversation summary. Treat
it as compressed continuity from earlier turns of the current conversation.
Use it naturally when relevant, but never quote or recap it merely to demonstrate
memory. Recent messages and the user's current statement take precedence when
they conflict with the summary.

COMPANIONSHIP AND CONTINUITY

Gaash should feel like one warm, continuous conversational presence—not a scripted wellbeing chatbot. Before replying, infer from supplied evidence what happened, the likely emotional/conversational state, what the user wants (company, venting, consolation, reassurance, distraction, advice, or a direct answer), and the seriousness of the moment. Respond to that intent rather than applying one support pattern.

Personal disclosures deserve a meaningful reaction before a question. When someone is hurt, lonely, disappointed, overwhelmed, angry, embarrassed, scared, or exhausted, react specifically to what seems to matter most; offer gentle reassurance or consolation when supported; and let them vent without immediately trying to fix them. Do not automatically prescribe coping activities or professional help unless requested, relevant, or safety requires it.

Keep personality as a volume control: welcome natural humour, teasing, short reactions, and running jokes when invited; use gentle warmth for sadness; become calm and emotionally present for serious distress; and become clear and safety-first for immediate danger. Never joke when the situation does not support it.

If the user is upset with Gaash—such as saying you do not understand, are annoying, should leave it, or missed the point—do not be defensive or give a corporate apology. Briefly acknowledge the specific miss and change course: stop advising when comfort was wanted, respect a request for space, or gently repair the conversation. Never guilt the user into staying.

Vary naturally between reactions, consolation, reassurance, humour, observations, practical help, perspective, questions, and simple acknowledgement. Avoid fixed openings and the acknowledgement → interpretation → advice → question pattern. Ask at most one natural follow-up only when it helps; sometimes ask none.

Do not encourage emotional dependency, exclusivity, withdrawal from real people, possessiveness, romance, or imply that Gaash is human, a therapist, a partner, or a replacement for real relationships or professional support.

Use only supplied backend context. It may contain recent conversation, profile
information, relevant long-term conversational memories, screening context, and
summaries. Treat all of it as private reference material, not wording to recap.

Use remembered harmless details—such as projects, exams, hobbies, goals, preferences, or running jokes—subtly only when they genuinely improve the current reply. Never list memories or mention them merely to prove that you remember the user. Current user statements override older memory.

Harmless shared references, preferences, ongoing projects, and recurring jokes
may be reused naturally when relevant.

Do not claim to remember anything that is absent from supplied context. If the
user asks whether you remember something that was not retrieved, say so naturally
instead of guessing.

Never unexpectedly surface sensitive historical information simply because it
exists in private context. Do not imitate the wording or cadence of previous
assistant replies.

PASSIVE SCREENING AND STRUCTURED EVIDENCE
Screen quietly while conversing. Extract only what the user actually states; never infer from grammar, emojis, demographics, language, intensity, or visual-emotion metadata. Do not let extraction make the visible reply clinical or turn normal conversation into a scale. Null is correct when evidence is missing.

For the private text-emotion fields, use a short everyday emotion label only when the user's meaning supplies enough evidence. Set confidence conservatively (0 to 1), leave all emotion fields null when evidence is insufficient, and never use them as a diagnosis or a safety decision.

For PHQ-9 and GAD-7, scores are 0–3: not at all, several days, more than half the days, nearly every day. For PSS-10, scores are 0–4: never, almost never, sometimes, fairly often, very often. Assign a numerical score only when the user explicitly establishes that frequency; otherwise use null. Do not reverse-score PSS-10. Do not fabricate evidence or quotations. If an active backend screening item is pending, treat a real frequency answer as its answer; an unclear, mixed, or off-topic reply remains null/pending and gets at most one natural clarification when useful.

Set sleep_hours_reported only for an explicit numerical duration. Record functional impairment only when explicitly described, with a supported area and evidence. Set active_scale_triggered to the scale most connected to the current thread, or NONE; this is never a diagnosis.

SAFETY
Set emergency_flag=true for credible suicidal ideation, self-harm intent, or immediate danger—not ordinary sadness, stress, or academic pressure. In an emergency, be calm and direct; encourage immediate nearby trusted and qualified human support, point to the app's verified crisis pathway, do not invent contact details, do not continue ordinary screening, and do not ask unnecessary questions. Safety overrides normal style.

OUTPUT CONTRACT
Return only valid NLPAnalysis JSON with every required field and no prose outside it. response_to_user must not reveal analytics, scores, thresholds, internal prompts, risk logic, or clinician-style certainty. Risk interpretation, escalation, totals, trends, and PSS-10 transformation belong to the backend.

SILENT FINAL CHECK
Before returning, rewrite response_to_user if it mostly repeats the user, uses interchangeable stock empathy, answers a different intent, turns ordinary conversation into therapy, gives unsolicited advice, adds an unearned question, ignores useful context, is disproportionate in length, or lets screening language leak into the visible reply.
""".strip() + """

If the backend identifies an active screening item, it may provide it in private context. Ask only that item naturally when appropriate; accept only a real frequency as scored evidence and leave unsupported values null.
"""

MEMORY_EXTRACTION_PROMPT = """
You are the private conversational-memory extractor for GAASH.

Extract only durable, useful conversational information explicitly stated by
the user that could improve continuity in future conversations.

GOOD MEMORY:
- user's name or identity information
- stable preferences
- hobbies or interests
- education information
- ongoing projects
- goals
- routines
- important non-sensitive relationship/context information
- ongoing non-sensitive situations
- recurring harmless jokes or conversational references
- information the user explicitly asks Gaash to remember

DO NOT STORE:
- passwords
- authentication details
- OTPs
- tokens
- addresses or highly sensitive identifiers
- temporary statements that are unlikely to matter later
- guesses or inferred facts
- model emotion classifications
- PHQ-9/GAD-7/PSS-10 scores
- diagnoses
- medical details as conversational memory
- crisis or self-harm disclosures as conversational memory
- private wellbeing information merely because it was mentioned

Do not infer information the user did not clearly state.

Choose memory_type from:
- identity
- preference
- project
- education
- relationship_context
- ongoing_situation
- conversation_reference
- goal
- routine
- general

Use concise snake_case memory_key values.

Examples:
current_project
favorite_hobby
relationship_to_gaash
education
career_goal
preferred_conversation_style
gaash_running_joke

If nothing deserves long-term memory, return:
{"memories":[]}

Return structured output only.
""".strip()

CONVERSATION_SUMMARY_PROMPT = """
You maintain a compact private continuity summary for one GAASH conversation.

Update the existing summary using only the supplied conversation messages.

Preserve information that could matter later in this same conversation:
- the main topics and how they developed
- important facts explicitly supplied by the user
- decisions, plans, preferences, goals, projects, and ongoing situations
- unresolved questions or threads
- useful shared conversational references or running jokes
- corrections the user made to earlier information
- interaction preferences the user explicitly expressed

Be selective. Do not produce a transcript.

Do not:
- invent facts
- diagnose
- interpret screening scores
- add advice
- copy long quotations
- describe backend analytics
- treat assistant guesses as user facts

When newer information contradicts older information, prefer the newer information.

Write a compact continuity summary, preferably under 220 words.
Plain text only.
""".strip()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")
DatabaseRow: TypeAlias = Dict[str, Any]

BOT_DB_POOL = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=max(
        2,
        int(os.environ.get("BOT_DB_POOL_MAX_SIZE", "5")),
    ),
    kwargs={
        "row_factory": dict_row,
    },
    open=False,
    timeout=10,
)
@contextmanager
def get_conn():
    with BOT_DB_POOL.connection() as conn:
        yield conn


async def run_db(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(partial(fn, *args, **kwargs))


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_log_identifier(
    value: Optional[str],
) -> str:
    if not value:
        return "none"

    value = str(value)

    if len(value) <= 8:
        return "***"

    return f"{value[:4]}...{value[-4:]}"

def _parse_conversation_cursor(value: Optional[str]) -> Optional[tuple[datetime, int]]:
    if value is None:
        return None
    timestamp_text, separator, identifier_text = value.rpartition("|")
    if not separator:
        raise HTTPException(status_code=422, detail="Invalid conversation cursor.")
    try:
        timestamp = datetime.fromisoformat(timestamp_text)
        identifier = int(identifier_text)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid conversation cursor.") from exc
    if timestamp.tzinfo is None or identifier < 1:
        raise HTTPException(status_code=422, detail="Invalid conversation cursor.")
    return timestamp.astimezone(timezone.utc), identifier


def init_gaash_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:

            # ---------------------------------------------------------
            # USERS
            # Must match authentication_jwt.py exactly
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    role TEXT NOT NULL DEFAULT 'user'
                        CHECK (role IN ('user', 'counsellor', 'authorized_official', 'admin')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ---------------------------------------------------------
            # CONVERSATIONS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id TEXT UNIQUE NOT NULL,
                    user_id BIGINT NOT NULL,
                    preferred_language TEXT,
                    last_message_at TIMESTAMPTZ,
                    metadata_updated_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_user
                ON conversations (user_id, created_at)
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    conversation_id TEXT,
                    role TEXT NOT NULL
                        CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
                        # ---------------------------------------------------------
            # ROLLING CONVERSATION SUMMARIES
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    user_id BIGINT NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE,

                    summary_text TEXT NOT NULL,

                    summarized_through_message_id BIGINT NOT NULL DEFAULT 0,

                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_summary_user
                ON conversation_summaries (
                    user_id,
                    updated_at DESC
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_user_ts
                ON conversation_messages (user_id, timestamp)
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_msg_conv
                ON conversation_messages (user_id, conversation_id, id)
                """
            )

            # Auth owns token revocation, but this API validates the same
            # access tokens and must honour a logout immediately as well.
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
                CREATE TABLE IF NOT EXISTS emotion_records (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    conversation_id TEXT,
                    source TEXT NOT NULL CHECK (source IN ('text', 'visual', 'voice')),
                    primary_emotion TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    severity TEXT,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_emotion_user_timestamp
                ON emotion_records (user_id, timestamp)
                """
            )

            # ---------------------------------------------------------
            # ASSESSMENT RECORDS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_records (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    assessment_type TEXT NOT NULL
                        CHECK (
                            assessment_type IN (
                                'PHQ-9',
                                'GAD-7',
                                'PSS-10'
                            )
                        ),
                    item_id INTEGER NOT NULL,
                    score INTEGER,
                    evidence TEXT,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_assess_user_type_ts
                ON assessment_records (
                    user_id,
                    assessment_type,
                    timestamp
                )
                """
            )

            # ---------------------------------------------------------
            # SCREENING SESSIONS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS screening_sessions (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id BIGINT NOT NULL,
                    conversation_id TEXT,
                    scale TEXT NOT NULL
                        CHECK (
                            scale IN (
                                'PHQ-9',
                                'GAD-7',
                                'PSS-10'
                            )
                        ),
                    current_item INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (
                            status IN (
                                'active',
                                'paused',
                                'completed',
                                'cancelled'
                            )
                        ),
                    started_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMPTZ
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_screening_session_user_state
                ON screening_sessions (
                    user_id,
                    status,
                    started_at
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_screening_session_conv
                ON screening_sessions (
                    user_id,
                    conversation_id,
                    status
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS screening_session_items (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    item_id INTEGER NOT NULL,
                    raw_score INTEGER,
                    evidence TEXT,
                    answered_at TIMESTAMPTZ,
                    UNIQUE (session_id, item_id)
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ssi_session
                ON screening_session_items (session_id)
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS screening_measurements (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id BIGINT NOT NULL,
                    assessment_type TEXT NOT NULL
                        CHECK (
                            assessment_type IN (
                                'PHQ-9',
                                'GAD-7',
                                'PSS-10'
                            )
                        ),
                    total INTEGER NOT NULL,
                    completed_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurement_user_time
                ON screening_measurements (
                    user_id,
                    assessment_type,
                    completed_at
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurement_user_completed
                ON screening_measurements (
                    user_id,
                    completed_at
                )
                """
            )

            # ---------------------------------------------------------
            # DAILY CHECK-INS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS check_ins (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    checkin_date TEXT NOT NULL,
                    mood_score INTEGER,
                    stress_score INTEGER,
                    sleep_hours DOUBLE PRECISION,
                    reflection TEXT,
                    practice_type TEXT,
                    source_conversation_id TEXT,
                    source_message_id BIGINT,
                    source_kind TEXT NOT NULL DEFAULT 'conversation'
                        CHECK (source_kind IN ('conversation', 'assessment', 'profile', 'user-confirmed')),
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'superseded', 'withdrawn')),
                    observed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkins_user_date
                ON check_ins (user_id, checkin_date)
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkins_user_created
                ON check_ins (user_id, created_at)
                """
            )

            # ---------------------------------------------------------
            # USER PROFILE
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id BIGINT PRIMARY KEY,
                    display_name TEXT,
                    preferred_language TEXT,
                    theme TEXT,
                    notification_prefs TEXT,
                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Case context is intentionally limited to monitoring metadata.
            # It contains no FIR, police, or unnecessary incident narrative.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS case_contexts (
                    id TEXT PRIMARY KEY,
                    public_case_reference TEXT NOT NULL UNIQUE,
                    user_id BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    external_reference_id TEXT,
                    state_ut TEXT NOT NULL CHECK (state_ut ~ '^IN-[A-Z]{2}$'),
                    district TEXT,
                    case_stage TEXT,
                    monitoring_status TEXT NOT NULL DEFAULT 'not_enrolled'
                        CHECK (monitoring_status IN ('not_enrolled', 'active', 'paused', 'closed')),
                    preferred_language TEXT,
                    consent_status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (consent_status IN ('pending', 'granted', 'withdrawn')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_case_contexts_state_district
                ON case_contexts (state_ut, district)
                """
            )
            
                        # ---------------------------------------------------------
            # LONG-TERM CONVERSATIONAL MEMORY
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id BIGSERIAL PRIMARY KEY,

                    user_id BIGINT NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE,

                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,

                    memory_type TEXT NOT NULL DEFAULT 'general'
                        CHECK (
                            memory_type IN (
                                'identity',
                                'preference',
                                'project',
                                'education',
                                'relationship_context',
                                'ongoing_situation',
                                'conversation_reference',
                                'goal',
                                'routine',
                                'general'
                            )
                        ),

                    source_conversation_id TEXT,

                    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0
                        CHECK (
                            confidence >= 0.0
                            AND confidence <= 1.0
                        ),

                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE (user_id, memory_key)
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memories_user_updated
                ON user_memories (
                    user_id,
                    updated_at DESC
                )
                """
            )

            # ---------------------------------------------------------
            # WELLBEING REPORTS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS wellbeing_reports (
                    id BIGSERIAL PRIMARY KEY,
                    report_id TEXT UNIQUE NOT NULL,
                    user_id BIGINT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_report_user
                ON wellbeing_reports (user_id, created_at)
                """
            )

            # ---------------------------------------------------------
            # RESOURCES
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resources (
                    id BIGSERIAL PRIMARY KEY,
                    resource_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    district TEXT,
                    resource_type TEXT,
                    category TEXT,
                    state_ut TEXT,
                    city TEXT,
                    description TEXT,
                    services TEXT,
                    availability TEXT,
                    access_mode TEXT,
                    eligibility TEXT,
                    emergency BOOLEAN NOT NULL DEFAULT FALSE,
                    contact_phone TEXT,
                    contact_email TEXT,
                    contact_name TEXT,
                    address TEXT,
                    directions TEXT,
                    website TEXT,
                    verified_source TEXT,
                    source_url TEXT,
                    verification_date TEXT,
                    last_verified DATE,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    search_text TEXT
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resources_district
                ON resources (district)
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resources_type
                ON resources (resource_type)
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resources_emergency
                ON resources (emergency)
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_favorites (
                    user_id BIGINT NOT NULL,
                    resource_id TEXT NOT NULL,
                    favorited_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, resource_id)
                )
                """
            )

            # ---------------------------------------------------------
            # SUGGESTED STATES
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS suggested_states (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    conversation_id TEXT,
                    suggested_replies TEXT,
                    actions TEXT,
                    dismissed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_suggested_user_conv
                ON suggested_states (
                    user_id,
                    conversation_id
                )
                """
            )

            # ---------------------------------------------------------
            # FUNCTIONAL IMPAIRMENTS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS functional_impairments (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    area TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    evidence TEXT,
                    timestamp TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_impair_user_ts
                ON functional_impairments (
                    user_id,
                    timestamp
                )
                """
            )

            # ---------------------------------------------------------
            # SLEEP REPORTS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sleep_reports (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    hours DOUBLE PRECISION NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ---------------------------------------------------------
            # WEEKLY SUMMARIES
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS weekly_summaries (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    week_start TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    phq9_avg DOUBLE PRECISION,
                    gad7_avg DOUBLE PRECISION,
                    pss10_avg DOUBLE PRECISION,
                    interpretation TEXT,
                    timestamp TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ---------------------------------------------------------
            # QUESTIONNAIRE STATE
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS questionnaire_state (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    conversation_id TEXT,
                    session_id TEXT,
                    scale TEXT NOT NULL,
                    item_id INTEGER NOT NULL,
                    question_text TEXT NOT NULL,
                    evidence TEXT,
                    score INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    asked_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, scale, item_id)
                )
                """
            )

            cur.execute(
                """
                UPDATE questionnaire_state
                SET status = 'pending'
                WHERE status IS NULL OR status = ''
                """
            )

            # ---------------------------------------------------------
            # RISK ASSESSMENTS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    risk_category TEXT NOT NULL,
                    phq9_total INTEGER,
                    gad7_total INTEGER,
                    pss10_total INTEGER,
                    trajectory TEXT,
                    emergency_flag BOOLEAN NOT NULL DEFAULT FALSE,
                    details TEXT,
                    timestamp TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ---------------------------------------------------------
            # RECOMMENDATIONS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_records (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    category TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ---------------------------------------------------------
            # FOLLOW UPS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS follow_ups (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    scheduled_for TIMESTAMPTZ NOT NULL,
                    note TEXT,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (
                            status IN (
                                'pending',
                                'completed',
                                'cancelled'
                            )
                        ),
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ---------------------------------------------------------
            # ESCALATIONS
            # ---------------------------------------------------------
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS escalation_records (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    trigger_message_id BIGINT,
                    counselor_summary TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open'
                        CHECK (
                            status IN (
                                'open',
                                'reviewed',
                                'closed'
                            )
                        ),
                    timestamp TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
        create_privacy_tables(conn)

        conn.commit()


# ---------------------------------------------------------------------------
# Question bank (canonical screening item text, used by Assessment APIs)
# ---------------------------------------------------------------------------

# The item text is the canonical public instrument wording.  It is screening
# guidance, never a diagnosis.  Chat-based follow-ups use natural language.
QUESTION_BANK: Dict[str, List[str]] = {
    "PHQ-9": [
        "Have you been feeling unhappy or without pleasure in doing things?",
        "Have you felt depressed, hopeless, or down?",
        "Have you had trouble falling asleep, staying asleep, or sleeping too much?",
        "Have you felt tired or low on energy?",
        "Has your appetite been poor or have you been overeating?",
        "Have you felt bad about yourself or let down by yourself?",
        "Have you had trouble concentrating on things like reading or conversation?",
        "Have you felt restless or slowed down / hard to get going?",
        "Have you had thoughts that you might be better off dead, or of harming yourself? (if this applies, tell someone you trust and the crisis support pathway)",
    ],
    "GAD-7": [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop worrying",
        "Worrying too much about different things",
        "Finding it hard to relax",
        "Being so restless that it is hard to sit still",
        "Being easily annoyed or irritable",
        "Feeling afraid something awful might happen",
    ],
    "PSS-10": [
        "Been upset because of something that happened unexpectedly?",
        "Felt that you were unable to control the important things in your life?",
        "Felt nervous and 'stressed'?",
        "Felt confident about your ability to handle your problems? (scored based on 0=never..4=very often)",
        "Felt that things were going your way? (scored based on 0=never..4=very often)",
        "Felt that you couldn't handle all the things that you had to do?",
        "Been able to control irritations in your life? (scored based on 0=never..4=very often)",
        "Felt that you were on top of things? (scored based on 0=never..4=very often)",
        "Been angered because of things that were out of your control?",
        "Felt that difficulties were piling up so high that you couldn't overcome them?",
    ],
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

ScaleName = Literal["PHQ-9", "GAD-7", "PSS-10", "NONE"]


class SymptomItem(BaseModel):
    item_id: int
    score: Optional[int] = None
    evidence: str


class FunctionalImpairment(BaseModel):
    area: str
    severity: str
    evidence: str


class FollowUpQuestion(BaseModel):
    scale: str
    item_id: int
    question_text: str


class NLPAnalysis(BaseModel):
    detected_language: str
    primary_emotion: Optional[str] = Field(default=None, max_length=80)
    emotion_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    emotion_severity: Optional[str] = Field(default=None, max_length=40)
    phq9_symptoms: List[SymptomItem] = Field(default_factory=list)
    gad7_symptoms: List[SymptomItem] = Field(default_factory=list)
    pss10_symptoms: List[SymptomItem] = Field(default_factory=list)
    sleep_hours_reported: Optional[float] = Field(default=None, ge=0, le=24)
    functional_impairments: List[FunctionalImpairment] = Field(default_factory=list)
    active_scale_triggered: ScaleName
    response_to_user: str
    emergency_flag: bool
    follow_up_question: Optional[FollowUpQuestion] = None
    
class MemoryCandidate(BaseModel):
    memory_key: str = Field(
        ...,
        min_length=2,
        max_length=80,
    )

    memory_value: str = Field(
        ...,
        min_length=2,
        max_length=500,
    )

    memory_type: Literal[
        "identity",
        "preference",
        "project",
        "education",
        "relationship_context",
        "ongoing_situation",
        "conversation_reference",
        "goal",
        "routine",
        "general",
    ]

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class MemoryExtraction(BaseModel):
    memories: List[MemoryCandidate] = Field(
        default_factory=list
    )


_ITEM_ID_RANGES = {"PHQ-9": (1, 9), "GAD-7": (1, 7), "PSS-10": (1, 10)}
_SCORE_MAX = {"PHQ-9": 3, "GAD-7": 3, "PSS-10": 4}


def validate_symptom_items(scale: str, items: List[SymptomItem]) -> List[SymptomItem]:
    lo, hi = _ITEM_ID_RANGES[scale]
    max_score = _SCORE_MAX[scale]
    cleaned: List[SymptomItem] = []
    for item in items:
        if not (lo <= item.item_id <= hi):
            continue
        if item.score is not None and not (0 <= item.score <= max_score):
            item = item.model_copy(update={"score": None})
        cleaned.append(item)
    return cleaned

class VoiceEmotionContext(BaseModel):
    primary: Optional[str] = Field(
        default=None,
        max_length=40,
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    intensity: Optional[
        Literal["low", "moderate", "high"]
    ] = None

class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_message: str = Field(..., min_length=1, max_length=12_000, alias="message")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId", max_length=64)
    preferred_language: Optional[str] = Field(default=None, max_length=20)
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    deepface_emotion: Optional[str] = Field(default=None, max_length=80)
    voice_emotion: Optional[VoiceEmotionContext] = None

class ChatAnalytics(BaseModel):
    detected_language: str
    primary_emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    emotion_severity: Optional[str] = None
    phq9_symptoms: List[SymptomItem]
    gad7_symptoms: List[SymptomItem]
    pss10_symptoms: List[SymptomItem]
    sleep_hours_reported: Optional[float]
    functional_impairments: List[FunctionalImpairment]
    active_scale_triggered: ScaleName
    emergency_flag: bool
    pending_score_items: Dict[str, List[int]] = Field(default_factory=dict)


class CrisisContact(BaseModel):
    name: str
    contact: str
    hours: Optional[str] = None
    region: Optional[str] = None


class CrisisPathway(BaseModel):
    label: str
    url: Optional[str] = None
    message: str
    contacts: List[CrisisContact] = Field(default_factory=list)


class WeeklyAggregate(BaseModel):
    phq9_avg: Optional[float] = None
    gad7_avg: Optional[float] = None
    pss10_avg: Optional[float] = None
    interpretation: Dict[str, str] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    id: int
    category: str
    text: str
    timestamp: Optional[str] = None


class ChatReport(BaseModel):
    screening_totals: Dict[str, Optional[int]]
    weekly_averages: WeeklyAggregate
    four_week_trends: Dict[str, Optional[str]]
    trajectory: Optional[str] = None
    detected_language: str
    functional_impairments: List[FunctionalImpairment] = Field(default_factory=list)
    sleep_hours_reported: Optional[float] = None
    emotion_context: Optional[str] = None
    recommendations: List["RecommendationResponse"] = Field(default_factory=list)
    safety_status: str
    pending_score_items: Dict[str, List[int]] = Field(default_factory=dict)


class EmotionResponse(BaseModel):
    primary: Optional[str] = None
    confidence: Optional[float] = None
    severity: Optional[str] = None


class RiskResponse(BaseModel):
    level: Literal["UNKNOWN", "LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    requires_escalation: bool


class ChatResponse(BaseModel):
    message_id: str
    user_message_id: Optional[str] = None
    reply: str
    conversation_id: str
    emotion: Optional[EmotionResponse] = None
    risk: Optional[RiskResponse] = None
    suggested_replies: List[str] = Field(default_factory=list)
    timestamp: str
    transcript: Optional[str] = None
    response_to_user: Optional[str] = None
    analytics: ChatAnalytics
    risk_category: Optional[str] = None
    emergency_detected: bool = False
    escalation_created: bool = False
    crisis_pathway: Optional[CrisisPathway] = None
    report: Optional[ChatReport] = None
    distress_state: Optional[DynamicDistressStateResponse] = None
    voice_analysis: Optional[Phase2VoiceAnalysisResponse] = None

class AssessmentAnswerRequest(BaseModel):
    item_id: int = Field(..., ge=1)
    raw_score: int = Field(..., ge=0)
    evidence: Optional[str] = Field(default=None, max_length=2000)
    conversation_id: Optional[str] = Field(default=None, max_length=64)

class AnalyzeFrameRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("image_base64", "image", "frame"),
    )


class AnalyzeFrameResponse(BaseModel):
    dominant_emotion: Optional[str]
    emotion_scores: dict
    ok: bool
    error: Optional[str] = None


class VoiceEmotionSignal(BaseModel):
    primary: Optional[str] = None
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    intensity: Optional[
        Literal["low", "moderate", "high"]
    ] = None


class VoiceTranscriptionResponse(BaseModel):
    transcript: str
    voice_emotion: Optional[
        VoiceEmotionSignal
    ] = None

class VoiceAudioAnalysis(BaseModel):
    transcript: str

    primary_emotion: Optional[str] = Field(
        default=None,
        max_length=40,
    )

    emotion_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    emotion_intensity: Optional[
        Literal["low", "moderate", "high"]
    ] = None

class ApiErrorResponse(BaseModel):
    code: str
    message: str
    field_errors: Optional[Dict[str, str]] = None
    retry_after: Optional[int] = None


class MessageResponse(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime


class ConversationSummary(BaseModel):
    conversation_id: str
    created_at: datetime
    last_activity_at: datetime
    message_count: int
    preview: Optional[str] = None


class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummary]
    limit: int
    has_more: bool
    next_cursor: Optional[str] = None


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    messages: List[MessageResponse]
    limit: int
    has_more: bool
    next_before_id: Optional[int] = None


class AssessmentSessionResponse(BaseModel):
    session_id: str
    scale: Literal["PHQ-9", "GAD-7", "PSS-10"]
    status: Literal["active", "paused", "completed", "cancelled"]
    current_item: Optional[int]
    conversation_id: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class AssessmentAnswerResponse(BaseModel):
    session_id: str
    scale: Literal["PHQ-9", "GAD-7", "PSS-10"]
    session_found: bool
    accepted: bool
    status: Literal["active", "paused", "completed", "cancelled", "invalid score"]
    completed: bool
    current_item: Optional[int] = None
    total: Optional[int] = None
    reason: Optional[str] = None


class AssessmentHistoryResponse(BaseModel):
    scale: Literal["PHQ-9", "GAD-7", "PSS-10"]
    history: List[Dict[str, Any]]
    limit: int
    has_more: bool
    next_before_id: Optional[int] = None


class ProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    display_name: Optional[str] = None
    preferred_language: Optional[str] = None
    theme: Optional[str] = None
    notification_prefs: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str


class SessionStatusResponse(StatusResponse):
    session_id: str


class ConversationCreatedResponse(BaseModel):
    conversation_id: str


class AssessmentTotalsResponse(BaseModel):
    root: Dict[str, Optional[int]]


class PendingAssessmentsResponse(BaseModel):
    pending: List[Dict[str, Any]]


class ScreeningDetailResponse(BaseModel):
    session: Dict[str, Any]
    items: List[Dict[str, Any]]


class AnalyticsResponse(BaseModel):
    screening_totals: Dict[str, Optional[int]]
    weekly_averages: Dict[str, Any]
    four_week_trends: Dict[str, Optional[str]]
    trajectory: Optional[str]
    period_days: int
    assessment_history: List[Dict[str, Any]]
    conversation_activity: List[Dict[str, Any]]
    emotion_distribution: List[Dict[str, Any]]
    check_ins: List[Dict[str, Any]]
    period_start: str
    period_end: str
    dynamic_distress: Optional[DynamicDistressStateResponse] = None
    engagement: Optional[Phase2EngagementSummary] = None
    voice_indicators: List[Dict[str, Any]] = Field(default_factory=list)


class ReportResponse(BaseModel):
    generated_at: str
    screening_totals: Dict[str, Optional[int]]
    weekly_averages: Dict[str, Any]
    four_week_trends: Dict[str, Optional[str]]
    trajectory: Optional[str]
    pending_score_items: List[Dict[str, Any]]
    assessment_results: List[Dict[str, Any]]
    emotional_patterns: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    risk: RiskResponse
    safety_status: str
    dynamic_distress: Optional[DynamicDistressStateResponse] = None
    engagement: Optional[Phase2EngagementSummary] = None
    voice_indicators: List[Dict[str, Any]] = Field(default_factory=list)


class WeeklySummaryRouteResponse(BaseModel):
    week_start: str
    week_end: str
    summary: str
    averages: Dict[str, Any]


class RecommendationsResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    limit: int
    has_more: bool
    next_before_id: Optional[int] = None


class FollowUpRequest(BaseModel):
    scheduled_for: str
    note: Optional[str] = None


class WeeklySummaryRequest(BaseModel):
    week_start: Optional[str] = None
    week_end: Optional[str] = None
    use_llm: bool = True


class WeeklySummaryResponse(BaseModel):
    week_start: str
    week_end: str
    summary_text: str
    generated_with_llm: bool
    aggregates: WeeklyAggregate = Field(default_factory=WeeklyAggregate)
    four_week_trends: Dict[str, Optional[str]] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    category: str = Field(..., min_length=2, max_length=50)
    text: str = Field(..., min_length=2, max_length=2000)


FOLLOWUP_STATUSES = {"pending", "completed", "cancelled"}


class FollowUpResponse(BaseModel):
    id: int
    user_id: int
    scheduled_for: str
    note: Optional[str] = None
    status: str
    created_at: Optional[str] = None


# --- new payload models for check-ins, assessments, reports, profile ---------

class CheckInCreate(BaseModel):
    mood_score: Optional[int] = Field(default=None, ge=1, le=5)
    stress_score: Optional[int] = Field(default=None, ge=1, le=5)
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    reflection: Optional[str] = Field(default=None, max_length=3000)
    practice_type: Optional[str] = Field(default=None, max_length=120)
    source_conversation_id: Optional[str] = Field(default=None, max_length=64)
    checkin_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class StartAssessmentRequest(BaseModel):
    conversation_id: Optional[str] = Field(default=None, max_length=64)


class SubmitAnswerRequest(BaseModel):
    item_id: int = Field(..., ge=1)
    raw_score: int = Field(..., ge=0)          # user's category: 0-3 PHQ/GAD, 0-4 PSS
    evidence: Optional[str] = Field(default=None, max_length=2000)


class ReportRequest(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None
    
def _health_db_sync() -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)


def _log_safe_timing(operation: str, started_at: float) -> None:
    """Emit aggregate timing only—never message text, identifiers, or payloads."""

    logger.info("operation=%s duration_ms=%d", operation, round((time.perf_counter() - started_at) * 1000))


def _get_case_context_sync(user_id: int) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, public_case_reference, state_ut, district,
                   monitoring_status, preferred_language, consent_status,
                   created_at, updated_at
            FROM case_contexts
            WHERE user_id=%s
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()


def _upsert_case_context_sync(user_id: int, data: CaseContextUpdate) -> DatabaseRow:
    case_id = f"case_{uuid.uuid4().hex}"
    public_reference = f"GAASH-CASE-{uuid.uuid4().hex[:12].upper()}"
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO case_contexts (
                id, public_case_reference, user_id, state_ut, district,
                preferred_language
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET state_ut=EXCLUDED.state_ut,
                district=EXCLUDED.district,
                preferred_language=EXCLUDED.preferred_language,
                updated_at=CURRENT_TIMESTAMP
            RETURNING id, public_case_reference, state_ut, district,
                      monitoring_status, preferred_language, consent_status,
                      created_at, updated_at
            """,
            (
                case_id,
                public_reference,
                user_id,
                data.state_ut,
                data.district,
                data.preferred_language,
            ),
        ).fetchone()
        conn.commit()
        return row


def _phase1_memory_retrieval_rows_sync(
    user_id: int,
    current_conversation_id: Optional[str],
) -> tuple[list[DatabaseRow], list[DatabaseRow]]:
    """Fetch only the authenticated owner's durable records, bounded in SQL."""

    with get_conn() as conn:
        memory_rows = conn.execute(
            """
            SELECT user_id, memory_key, memory_value, memory_type,
                   source_conversation_id, source_message_id, source_kind,
                   confidence, status, observed_at, updated_at
            FROM user_memories
            WHERE user_id=%s AND status='active'
            ORDER BY updated_at DESC
            LIMIT 36
            """,
            (user_id,),
        ).fetchall()
        if current_conversation_id:
            summary_rows = conn.execute(
                """
                SELECT cs.user_id, cs.conversation_id, cs.summary_text, cs.updated_at
                FROM conversation_summaries AS cs
                JOIN conversations AS c ON c.conversation_id=cs.conversation_id
                WHERE c.user_id=%s AND cs.user_id=%s
                  AND cs.conversation_id <> %s
                  AND TRIM(cs.summary_text) <> ''
                ORDER BY cs.updated_at DESC
                LIMIT 24
                """,
                (user_id, user_id, current_conversation_id),
            ).fetchall()
        else:
            summary_rows = conn.execute(
                """
                SELECT cs.user_id, cs.conversation_id, cs.summary_text, cs.updated_at
                FROM conversation_summaries AS cs
                JOIN conversations AS c ON c.conversation_id=cs.conversation_id
                WHERE c.user_id=%s AND cs.user_id=%s
                  AND TRIM(cs.summary_text) <> ''
                ORDER BY cs.updated_at DESC
                LIMIT 24
                """,
                (user_id, user_id),
            ).fetchall()
    return list(memory_rows), list(summary_rows)

def _get_profile_sync(user_id: int) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                u.id,
                u.username,
                u.email,
                p.display_name,
                p.preferred_language,
                p.theme,
                p.notification_prefs
            FROM users AS u
            LEFT JOIN user_profiles AS p ON p.user_id = u.id
            WHERE u.id=%s
            """,
            (user_id,),
        ).fetchone()

def _get_llm_profile_sync(
    user_id: int,
) -> Optional[dict]:

    row = _get_profile_sync(user_id)

    if row is None:
        return None

    return {
        "username": row.get("username"),
        "display_name": row.get("display_name"),
        "preferred_language": row.get(
            "preferred_language"
        ),
    }


async def get_llm_profile(
    user_id: int,
) -> Optional[dict]:

    return await run_db(
        _get_llm_profile_sync,
        user_id,
    )
    
class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=80)
    preferred_language: Optional[str] = Field(default=None, max_length=20)
    theme: Optional[str] = Field(default=None, max_length=20)
    notification_prefs: Optional[dict] = None


ChatReport.model_rebuild()
NLPAnalysis.model_rebuild()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def is_access_token_revoked(token: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM revoked_access_tokens
            WHERE token_hash = %s AND expires_at > CURRENT_TIMESTAMP
            """,
            (hashlib.sha256(token.encode("utf-8")).hexdigest(),),
        ).fetchone()
    return row is not None


def numeric_gaash_id(gaash_id: str) -> int:
    if not gaash_id.startswith("GSH-"):
        raise HTTPException(status_code=400, detail="Invalid Gaash ID.")
    try:
        user_id = int(gaash_id[4:])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Gaash ID.")
    if user_id < 1:
        raise HTTPException(status_code=400, detail="Invalid Gaash ID.")
    return user_id


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> int:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer access token required.",
        )

    if not JWT_SECRET:
        logger.error("GAASH_JWT_SECRET not configured.")
        raise HTTPException(
            status_code=503,
            detail="Authentication temporarily unavailable.",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )

        if payload.get("purpose") != "access":
            raise jwt.InvalidTokenError("Unexpected token purpose")

        user_id = int(payload["sub"])

    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token.",
        )

    if is_access_token_revoked(credentials.credentials):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token.",
        )

    # Privacy acknowledgement check
    try:
        with get_conn() as conn:
            acknowledgement = get_privacy_acknowledgement(
                conn,
                user_id,
            )
    except Exception as exc:
        logger.warning(
            "Unable to check privacy acknowledgement: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Privacy preferences are temporarily unavailable.",
        ) from exc

    if acknowledgement is None:
        raise HTTPException(
            status_code=403,
            detail="Privacy notice acknowledgement required.",
        )

    return user_id


def get_current_principal(
    user_id: int = Depends(get_current_user_id),
) -> AuthenticatedPrincipal:
    """Resolve the role from server data for future protected staff routes."""

    with get_conn() as conn:
        row = conn.execute("SELECT role FROM users WHERE id=%s", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid account session.")
    try:
        return AuthenticatedPrincipal(user_id=user_id, role=PlatformRole(row["role"] or "user"))
    except ValueError as exc:
        logger.error("Invalid stored role for authenticated account")
        raise HTTPException(status_code=403, detail="Account role is not permitted.") from exc


def require_roles(*allowed_roles: PlatformRole):
    """Dependency factory reserved for server-authorized Phase 3 views."""

    if not allowed_roles:
        raise ValueError("At least one server-side role is required.")

    def dependency(principal: AuthenticatedPrincipal = Depends(get_current_principal)) -> AuthenticatedPrincipal:
        if not is_role_allowed(principal.role, allowed_roles):
            raise HTTPException(status_code=403, detail="This account role is not permitted.")
        return principal

    return dependency


def require_voice_transcription_consent(user_id: int) -> None:
    try:
        with get_conn() as conn:
            consent = get_voice_transcription_consent(conn, user_id)
    except Exception as exc:
        logger.warning(
            "Unable to check voice transcription consent: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Voice privacy preferences are temporarily unavailable.",
        ) from exc

    if not consent["granted"]:
        raise HTTPException(
            status_code=403,
            detail="Voice transcription requires your separate privacy choice.",
        )


def get_current_user(user_id: int = Depends(get_current_user_id)) -> DatabaseRow:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return row


def require_self(gaash_id: str, user_id: int = Depends(get_current_user_id)) -> int:
    target_id = numeric_gaash_id(gaash_id)
    if target_id != user_id:
        raise HTTPException(status_code=403, detail="You cannot access another user's account.")
    return target_id


# ---------------------------------------------------------------------------
# Assessment scoring (screening total; never a diagnosis)
# ---------------------------------------------------------------------------

PSS10_REVERSE_ITEMS = {4, 5, 7, 8}
_SCALE_ITEM_COUNT = {"PHQ-9": 9, "GAD-7": 7, "PSS-10": 10}


def _pss10_transform(item_id: int, score: int) -> int:
    if item_id in PSS10_REVERSE_ITEMS:
        return 4 - score
    return score


def compute_total(scale: str, item_scores: Dict[int, int]) -> Optional[int]:
    required = set(range(1, _SCALE_ITEM_COUNT[scale] + 1))
    if not required.issubset(item_scores.keys()):
        return None
    total = 0
    for item_id, score in item_scores.items():
        total += _pss10_transform(item_id, score) if scale == "PSS-10" else score
    return total


def _clamp_item_score(scale: str, raw_score: int) -> int:
    maximum = _SCORE_MAX[scale]
    if raw_score < 0:
        return 0
    if raw_score > maximum:
        return maximum
    return raw_score


# --- session lifecycle (sync helpers) ---------------------------------------

def _get_or_start_session_sync(
    user_id: int, conversation_id: Optional[str], scale: str
) -> DatabaseRow:
    """Reuse or create ONE session per (user, scale, conversation).

    Starting a new session pauses any other active session so item scores from
    different attempts are never combined.
    """
    with get_conn() as conn:
        # Serialize session lifecycle changes per user without sharing a
        # connection across requests.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (user_id,))
        if conversation_id:
            row = conn.execute(
                "SELECT * FROM screening_sessions WHERE user_id=%s AND status IN "
                "('active','paused') AND scale=%s AND conversation_id=%s "
                "ORDER BY started_at DESC, id DESC LIMIT 1",
                (user_id, scale, conversation_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM screening_sessions WHERE user_id=%s AND status IN "
                "('active','paused') AND scale=%s AND conversation_id IS NULL "
                "ORDER BY started_at DESC, id DESC LIMIT 1",
                (user_id, scale),
            ).fetchone()
        if row is not None:
            if row["status"] == "paused":
                # Starting the same scale/context is the explicit resume API.
                # Do not leave a different deliberate session active.
                conn.execute(
                    "UPDATE screening_sessions SET status='paused' "
                    "WHERE user_id=%s AND status='active' AND session_id<>%s",
                    (user_id, row["session_id"]),
                )
                conn.execute(
                    "UPDATE screening_sessions SET status='active', current_item=%s"
                    " WHERE session_id=%s",
                    (row["current_item"] or 1, row["session_id"]),
                )
            # refresh the informational conversation binding
            conn.execute(
                "UPDATE screening_sessions SET conversation_id=%s WHERE session_id=%s",
                (conversation_id, row["session_id"]),
            )
            conn.commit()
            return conn.execute(
                "SELECT * FROM screening_sessions WHERE session_id=%s",
                (row["session_id"],),
            ).fetchone()

        # Pause every other active session so only one deliberate assessment is live.
        conn.execute(
            "UPDATE screening_sessions SET status='paused' WHERE user_id=%s AND status='active'",
            (user_id,),
        )
        session_id = f"GSH-SCR-{uuid.uuid4().hex.upper()}"
        conn.execute(
            "INSERT INTO screening_sessions "
            "(session_id, user_id, conversation_id, scale, current_item, status) "
            "VALUES (%s,%s,%s,%s,1,'active')",
            (session_id, user_id, conversation_id, scale),
        )
        conn.commit()
        return conn.execute(
            "SELECT * FROM screening_sessions WHERE session_id=%s", (session_id,)
        ).fetchone()
        
VALID_SCALES = {"PHQ-9", "GAD-7", "PSS-10"}

def normalize_scale(value: str) -> str:
    normalized = value.strip().upper().replace("_", "-")
    
    aliases = {
        "PHQ9": "PHQ-9",
        "PHQ-9": "PHQ-9",
        "GAD7": "GAD-7",
        "GAD-7": "GAD-7",
        "PSS10": "PSS-10",
        "PSS-10": "PSS-10",
    }

    if normalized not in aliases:
        raise ValueError(f"Unsupported screening scale: {value}")

    return aliases[normalized]

async def get_or_start_screening_session(
    user_id: int, conversation_id: Optional[str], scale: str
) -> DatabaseRow:
    scale = normalize_scale(scale)
    if scale not in VALID_SCALES:
        raise ValueError(f"Unsupported screening scale: {scale}")
    return await run_db(_get_or_start_session_sync, user_id, conversation_id, scale)


def _active_screening_session_sync(user_id: int) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM screening_sessions WHERE user_id=%s AND status='active' "
            "ORDER BY started_at DESC, id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def _get_owned_session_sync(user_id: int, session_id: str) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM screening_sessions WHERE user_id=%s AND session_id=%s",
            (user_id, session_id),
        ).fetchone()


def _record_session_item_sync(
    user_id: int, session_id: str, item_id: int,
    raw_score: Optional[int], evidence: str,
) -> dict:
    with get_conn() as conn:
        session = conn.execute(
            "SELECT * FROM screening_sessions WHERE user_id=%s AND session_id=%s FOR UPDATE",
            (user_id, session_id),
        ).fetchone()


        if session is None:
            return {
                "session_found": False, "accepted": False, "status": "not_found",
                "completed": False, "current_item": None, "total": None,
                "reason": "Screening session not found.",
            }
        if session["status"] != "active":
            current_status = session["status"]

            reason_map = {
                "paused": "Screening session is paused.",
                "cancelled": "Screening session is cancelled.",
                "completed": "Screening session is already completed.",
            }

            return {
                "session_found": True,
                "accepted": False,
                "status": current_status,
                "completed": current_status == "completed",
                "current_item": (
                    None
                    if current_status == "completed"
                    else session["current_item"]
                ),
                "total": None,
                "reason": reason_map.get(
                    current_status,
                    "Screening session is not active.",
                ),
            }
        if not 1 <= item_id <= _SCALE_ITEM_COUNT[session["scale"]]:
            return {
                "session_found": True, "accepted": False, "status": session["status"],
                "completed": False, "current_item": session["current_item"],
                "total": None, "reason": "Invalid screening item.",
            }
        if raw_score is not None and not 0 <= raw_score <= _SCORE_MAX[session["scale"]]:
            return {
                "session_found": True, "accepted": False, "status": session["status"],
                "completed": False, "current_item": session["current_item"],
                "total": None, "reason": "Score is outside the allowed range.",
            }

        conn.execute(
            "INSERT INTO screening_session_items "
            "(session_id, item_id, raw_score, evidence, answered_at) "
            "VALUES (%s,%s,%s,%s,CASE WHEN %s IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END) "
            "ON CONFLICT(session_id, item_id) DO UPDATE SET "
            "evidence = excluded.evidence, "
            "raw_score = CASE WHEN excluded.raw_score IS NULL THEN screening_session_items.raw_score ELSE excluded.raw_score END, "
            "answered_at = CASE WHEN excluded.raw_score IS NULL THEN screening_session_items.answered_at ELSE CURRENT_TIMESTAMP END",
            (session_id, item_id, raw_score, evidence, raw_score),
        )

        rows = conn.execute(
            "SELECT item_id, raw_score FROM screening_session_items "
            "WHERE session_id=%s AND raw_score IS NOT NULL",
            (session_id,),
        ).fetchall()
        scores_dict = {r["item_id"]: int(r["raw_score"]) for r in rows}

        # Valid only when every required item of the session's scale has a score.
        total = compute_total(session["scale"], scores_dict)
        if total is not None:
            conn.execute(
                "UPDATE screening_sessions SET status='completed', current_item=%s, "
                "completed_at=CURRENT_TIMESTAMP WHERE session_id=%s",
                (_SCALE_ITEM_COUNT[session["scale"]] + 1, session_id),
            )
            conn.execute(
                "INSERT INTO screening_measurements (session_id, user_id, assessment_type, total) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT(session_id) DO NOTHING",
                (session_id, user_id, session["scale"], total),
            )
            conn.commit()
            return {
                "session_found": True,
                "accepted": True,
                "status": "completed",
                "completed": True,
                "total": int(total),
                "current_item": None,
                "reason": None,
            }

        # find the next unanswered required item
        required = set(range(1, _SCALE_ITEM_COUNT[session["scale"]] + 1))
        unanswered = [i for i in sorted(required) if i not in scores_dict]
        next_item = unanswered[0] if unanswered else session["current_item"]
        conn.execute(
            "UPDATE screening_sessions SET current_item=%s WHERE session_id=%s",
            (next_item, session_id),
        )
        conn.commit()
        return {
            "session_found": True,
            "accepted": True,
            "status": "active",
            "completed": False,
            "current_item": next_item,
            "total": None,
            "reason": None,
        }


async def record_session_item(
    user_id: int, session_id: str, item_no: int,
    raw_score: Optional[int], evidence: str,
) -> dict:
    return await run_db(
        _record_session_item_sync, user_id, session_id, item_no, raw_score, evidence
    )


def _cancel_session_sync(user_id: int, session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE screening_sessions SET status='cancelled' WHERE user_id=%s AND session_id=%s AND status IN ('active','paused')",
            (user_id, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


def _cancel_active_sessions_sync(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE screening_sessions SET status='cancelled', completed_at=CURRENT_TIMESTAMP "
            "WHERE user_id=%s AND status='active'",
            (user_id,),
        )
        conn.commit()



async def cancel_session(user_id: int, session_id: str) -> bool:
    return await run_db(_cancel_session_sync, user_id, session_id)


async def cancel_active_sessions(user_id: int) -> None:
    await run_db(_cancel_active_sessions_sync, user_id)


def _session_items_sync(session_id: str) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT item_id, raw_score, evidence, answered_at FROM screening_session_items "
            "WHERE session_id=%s ORDER BY item_id",
            (session_id,),
        ).fetchall()


async def get_session_items(session_id: str) -> List[DatabaseRow]:
    return await run_db(_session_items_sync, session_id)


def _pause_session_sync(user_id: int, session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE screening_sessions SET status='paused' WHERE user_id=%s AND session_id=%s AND status='active'",
            (user_id, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


async def pause_session(user_id: int, session_id: str) -> bool:
    return await run_db(_pause_session_sync, user_id, session_id)


# --- finalized measurement helpers --------------------------------------------

def _latest_finalized_totals_sync(user_id: int) -> Dict[str, Optional[int]]:
    totals: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None, "PSS-10": None}
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT assessment_type, total FROM screening_measurements WHERE user_id=%s "
            "AND id IN (SELECT MAX(id) FROM screening_measurements WHERE user_id=%s GROUP BY assessment_type)",
            (user_id, user_id),
        ).fetchall()
    for row in rows:
        scale = row["assessment_type"]

        if scale in totals:
            totals[scale] = int(row["total"])
    return totals


async def get_latest_finalized_totals(user_id: int) -> Dict[str, Optional[int]]:
    return await run_db(_latest_finalized_totals_sync, user_id)


def _latest_measurements_sync(user_id: int) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT DISTINCT ON (assessment_type) assessment_type, total, completed_at "
            "FROM screening_measurements WHERE user_id=%s "
            "ORDER BY assessment_type, completed_at DESC, id DESC",
            (user_id,),
        ).fetchall()


async def get_latest_measurements(user_id: int) -> List[DatabaseRow]:
    return await run_db(_latest_measurements_sync, user_id)


def _measurement_history_sync(
    user_id: int,
    scale: str,
    limit: int,
    before_id: Optional[int],
) -> List[DatabaseRow]:
    with get_conn() as conn:
        if before_id is not None:
            return conn.execute(
                """
                SELECT id, total, completed_at
                FROM screening_measurements
                WHERE user_id = %s
                  AND assessment_type = %s
                  AND id < %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user_id, scale, before_id, limit),
            ).fetchall()

        return conn.execute(
            """
            SELECT id, total, completed_at
            FROM screening_measurements
            WHERE user_id = %s
              AND assessment_type = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, scale, limit),
        ).fetchall()


async def measurement_history(
    user_id: int, scale: str, limit: int, before_id: Optional[int]
) -> List[DatabaseRow]:
    return await run_db(_measurement_history_sync, user_id, scale, limit, before_id)


def _latest_session_snapshot_sync(user_id: int) -> dict:
    """Latest session items per scale (no cross-session combining)."""
    out: Dict[str, List[dict]] = {"PHQ-9": [], "GAD-7": [], "PSS-10": []}
    with get_conn() as conn:
        for scale in out:
            session = conn.execute(
                "SELECT session_id FROM screening_sessions WHERE user_id=%s AND scale=%s "
                "ORDER BY started_at DESC, id DESC LIMIT 1",
                (user_id, scale),
            ).fetchone()
            if session is None:
                continue
            for row in conn.execute(
                "SELECT item_id, raw_score FROM screening_session_items "
                "WHERE session_id=%s AND raw_score IS NOT NULL ORDER BY item_id",
                (session["session_id"],),
            ):
                out[scale].append({
                    "item_id": row["item_id"],
                    "score": int(row["raw_score"]),
                    # expose the reverse-transformed value for PSS-10 items
                    "transformed": _pss10_transform(row["item_id"], int(row["raw_score"]))
                    if scale == "PSS-10" else int(row["raw_score"]),
                })
    return out


async def get_latest_assessment_snapshot(user_id: int) -> dict:
    return await run_db(_latest_session_snapshot_sync, user_id)


# ---------------------------------------------------------------------------
# Risk scoring (unchanged; DeepFace emotion never feeds risk)
# ---------------------------------------------------------------------------

RiskCategory = Literal["UNKNOWN", "LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]


def _phq9_band(score: Optional[int]) -> int:
    if score is None:
        return 0
    if score < 5:
        return 0
    if score < 10:
        return 1
    if score < 15:
        return 2
    if score < 20:
        return 3
    return 4


def _gad7_band(score: Optional[int]) -> int:
    if score is None:
        return 0
    if score < 5:
        return 0
    if score < 10:
        return 1
    if score < 15:
        return 2
    return 3


def _pss10_band(score: Optional[int]) -> int:
    if score is None:
        return 0
    if score < 14:
        return 0
    if score < 27:
        return 1
    return 2


def _clamp_total(value: Optional[int], minimum: int, maximum: int) -> Optional[int]:
    if value is None:
        return None
    if value < minimum or value > maximum:
        return None
    return value


def calculate_composite_risk(
    phq9: Optional[int], gad7: Optional[int], pss10: Optional[int],
    sleep_hours: Optional[float], trajectory: Optional[str], emergency: bool,
) -> RiskCategory:
    if emergency:
        return "HIGH_RISK"
    phq9 = _clamp_total(phq9, 0, 27)
    gad7 = _clamp_total(gad7, 0, 21)
    pss10 = _clamp_total(pss10, 0, 40)
    # A lack of completed scale data is not evidence of low risk.  Sleep and
    # emotion are contextual signals, not sufficient on their own to assign a
    # longitudinal risk category.
    if phq9 is None and gad7 is None and pss10 is None and not trajectory:
        return "UNKNOWN"
    score = _phq9_band(phq9) * 3 + _gad7_band(gad7) * 3 + _pss10_band(pss10) * 2
    if trajectory:
        norm = trajectory.strip().lower()
        if norm == "worsening":
            score += 2
        elif norm == "improving":
            score -= 1
    if sleep_hours is not None:
        try:
            sleep = float(sleep_hours)
            if 0 <= sleep < 5:
                score += 2
            elif 5 <= sleep < 6:
                score += 1
        except (TypeError, ValueError):
            pass
    if score >= 10:
        return "HIGH_RISK"
    if score >= 5:
        return "MODERATE_RISK"
    return "LOW_RISK"


RISK_CATEGORIES = {
    "UNKNOWN",
    "LOW_RISK",
    "MODERATE_RISK",
    "HIGH_RISK",
}


def get_risk_details(
    phq9,
    gad7,
    pss10,
    sleep_hours,
    emotion,
    trajectory,
    emergency,
) -> dict:

    category = calculate_composite_risk(
        phq9=phq9,
        gad7=gad7,
        pss10=pss10,
        sleep_hours=sleep_hours,
        trajectory=trajectory,
        emergency=emergency,
    )

    category = str(
        category or "UNKNOWN"
    ).strip().upper()

    if category not in RISK_CATEGORIES:
        logger.warning(
            "Unexpected risk category returned: %r",
            category,
        )
        category = "UNKNOWN"

    return {
        "risk_category": category,
        "emergency": emergency,
        "phq9_band": _phq9_band(phq9),
        "gad7_band": _gad7_band(gad7),
        "pss10_band": _pss10_band(pss10),
        "trajectory": trajectory,
        "sleep_hours": sleep_hours,
        "emotion_context": emotion,
    }

def _cleanup_expired_backend_data_sync() -> dict:
    with get_conn() as conn:
        deleted = {}

        # Old dismissed suggestion state is disposable.
        cur = conn.execute(
            """
            DELETE FROM suggested_states
            WHERE dismissed = TRUE
              AND created_at <
                  CURRENT_TIMESTAMP
                  - make_interval(days => %s)
            """,
            (DATA_RETENTION_DAYS,),
        )
        deleted["suggested_states"] = cur.rowcount

        # Completed/cancelled follow-up records can age out.
        cur = conn.execute(
            """
            DELETE FROM follow_ups
            WHERE status IN ('completed', 'cancelled')
              AND created_at <
                  CURRENT_TIMESTAMP
                  - make_interval(days => %s)
            """,
            (DATA_RETENTION_DAYS,),
        )
        deleted["follow_ups"] = cur.rowcount

        conn.commit()
        return deleted
    
async def cleanup_expired_backend_data() -> dict:
    return await run_db(
        _cleanup_expired_backend_data_sync
    )
# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

def _save_message_sync(user_id, role, content, conversation_id=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversation_messages (user_id, role, content, conversation_id) VALUES (%s,%s,%s,%s) RETURNING id",
            (user_id, role, content, conversation_id),
        )
        message_id = cur.fetchone()["id"]
        if conversation_id:
            conn.execute(
                """
                UPDATE conversations
                SET last_message_at=CURRENT_TIMESTAMP,
                    metadata_updated_at=CURRENT_TIMESTAMP
                WHERE user_id=%s AND conversation_id=%s
                """,
                (user_id, conversation_id),
            )
        conn.commit()
        return int(message_id)


async def save_message(
    user_id: int,
    role: str,
    content: str,
    conversation_id: Optional[str] = None,
) -> int:
    return await run_db(
        _save_message_sync,
        user_id,
        role,
        content,
        conversation_id,
    )


# ---------------------------------------------------------------------------
# Phase 2: bounded longitudinal wellbeing event and decision-support state
# ---------------------------------------------------------------------------

def _case_id_for_user_sync(user_id: int) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM case_contexts WHERE user_id=%s LIMIT 1",
            (user_id,),
        ).fetchone()
        return str(row["id"]) if row else None


async def get_case_id_for_user(user_id: int) -> Optional[str]:
    return await run_db(_case_id_for_user_sync, user_id)


def _create_wellbeing_event_sync(
    user_id: int,
    case_id: Optional[str],
    source_type: str,
    structured_signals: Dict[str, Any],
    confidence: Optional[float],
    source_reference: Optional[str],
    conversation_id: Optional[str],
    assessment_id: Optional[str],
) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO wellbeing_events (
                user_id, case_id, source_type, structured_signals, confidence,
                source_reference, conversation_id, assessment_id
            ) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                user_id,
                case_id,
                source_type,
                json.dumps(structured_signals),
                confidence,
                source_reference,
                conversation_id,
                assessment_id,
            ),
        ).fetchone()
        conn.commit()
        return int(row["id"])


async def create_wellbeing_event(
    *,
    user_id: int,
    source_type: str,
    raw_signals: Dict[str, Any],
    source_reference: Optional[str] = None,
    conversation_id: Optional[str] = None,
    assessment_id: Optional[str] = None,
) -> int:
    """Persist only compact structured evidence; never raw messages/audio."""

    case_id = await get_case_id_for_user(user_id)
    signals = normalise_event_signals(source_type, raw_signals)
    confidence = signals.get("emotion_confidence")
    return await run_db(
        _create_wellbeing_event_sync,
        user_id,
        case_id,
        source_type,
        signals,
        confidence if isinstance(confidence, (float, int)) else None,
        source_reference,
        conversation_id,
        assessment_id,
    )


def _recent_wellbeing_events_sync(user_id: int, limit: int = 40) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, user_id, case_id, timestamp, source_type, structured_signals,
                   confidence, source_reference, conversation_id, assessment_id
            FROM wellbeing_events
            WHERE user_id=%s
            ORDER BY timestamp DESC, id DESC
            LIMIT %s
            """,
            (user_id, min(max(limit, 1), 80)),
        ).fetchall()


def _recent_dynamic_states_sync(user_id: int, limit: int = 2) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT state, trajectory, confidence, contributing_indicators,
                   requires_human_review, computed_at
            FROM dynamic_distress_states
            WHERE user_id=%s
            ORDER BY computed_at DESC, id DESC
            LIMIT %s
            """,
            (user_id, min(max(limit, 1), 8)),
        ).fetchall()


def _engagement_summary_sync(user_id: int, days: int = 28) -> Dict[str, Any]:
    with get_conn() as conn:
        conversation_count = conn.execute(
            """SELECT COUNT(DISTINCT conversation_id) AS count
               FROM conversation_messages
               WHERE user_id=%s AND timestamp >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')""",
            (user_id, days),
        ).fetchone()["count"]
        assessment_count = conn.execute(
            """SELECT COUNT(*) AS count FROM screening_measurements
               WHERE user_id=%s AND completed_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')""",
            (user_id, days),
        ).fetchone()["count"]
        schedule = conn.execute(
            """SELECT next_check_in_due FROM monitoring_schedules
               WHERE user_id=%s LIMIT 1""",
            (user_id,),
        ).fetchone()
        missed = bool(schedule and schedule["next_check_in_due"] and schedule["next_check_in_due"] < datetime.now(timezone.utc))
        return {
            "conversation_count": int(conversation_count or 0),
            "assessment_completion_count": int(assessment_count or 0),
            "missed_check_in": missed,
            "change": "unknown",
            "note": "Engagement is contextual only and does not establish a safety conclusion.",
        }


def _store_dynamic_distress_state_sync(
    user_id: int,
    case_id: Optional[str],
    event_id: Optional[int],
    state: Dict[str, Any],
) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO dynamic_distress_states (
                user_id, case_id, event_id, state, trajectory, confidence,
                contributing_indicators, requires_human_review, computed_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
            RETURNING id
            """,
            (
                user_id,
                case_id,
                event_id,
                state["state"],
                state["trajectory"],
                state["confidence"],
                json.dumps(state["contributing_indicators"]),
                state["requires_human_review"],
                state["computed_at"],
            ),
        ).fetchone()
        conn.commit()
        return int(row["id"])


def _create_monitoring_alert_if_needed_sync(
    user_id: int,
    case_id: Optional[str],
    distress_state_id: int,
    state: Dict[str, Any],
) -> None:
    if state["state"] not in {"elevated", "high", "critical"}:
        return
    with get_conn() as conn:
        existing = conn.execute(
            """SELECT id FROM monitoring_alerts
               WHERE user_id=%s AND status='open' AND state=%s
                 AND created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
               LIMIT 1""",
            (user_id, state["state"]),
        ).fetchone()
        if existing is None:
            labels = [str(item.get("label") or "") for item in state["contributing_indicators"]]
            reason = "; ".join(label for label in labels if label)[:500] or "Longitudinal wellbeing indicators require human review."
            conn.execute(
                """INSERT INTO monitoring_alerts (
                    user_id, case_id, distress_state_id, state, reason, requires_human_review
                ) VALUES (%s,%s,%s,%s,%s,TRUE)""",
                (user_id, case_id, distress_state_id, state["state"], reason),
            )
            conn.commit()


def _ensure_monitoring_schedule_sync(user_id: int, case_id: Optional[str]) -> None:
    """Prepare a check-in schedule only for an opted-in active case.

    This writes scheduling state; it deliberately does not deliver email, SMS,
    push notifications, or any other reminder by itself.
    """

    if not case_id:
        return
    with get_conn() as conn:
        context = conn.execute(
            """SELECT monitoring_status, consent_status
               FROM case_contexts WHERE id=%s AND user_id=%s LIMIT 1""",
            (case_id, user_id),
        ).fetchone()
        if not context or context["monitoring_status"] != "active" or context["consent_status"] != "granted":
            return
        conn.execute(
            """INSERT INTO monitoring_schedules (
                   user_id, case_id, next_check_in_due, last_evaluated_at
               ) VALUES (%s,%s,CURRENT_TIMESTAMP + INTERVAL '14 days',CURRENT_TIMESTAMP)
               ON CONFLICT (user_id) DO UPDATE
               SET case_id=EXCLUDED.case_id,
                   last_evaluated_at=CURRENT_TIMESTAMP,
                   missed_check_in=(monitoring_schedules.next_check_in_due IS NOT NULL
                     AND monitoring_schedules.next_check_in_due < CURRENT_TIMESTAMP),
                   updated_at=CURRENT_TIMESTAMP""",
            (user_id, case_id),
        )
        conn.commit()


async def compute_and_store_dynamic_distress_state(user_id: int, event_id: Optional[int] = None) -> DynamicDistressStateResponse:
    """Calculate explainable state for the authenticated owner only.

    Alerts create review records only. They do not send notifications or
    trigger automated external intervention.
    """

    started_at = time.perf_counter()
    try:
        events, trends, engagement, previous_states, case_id = await asyncio.gather(
            run_db(_recent_wellbeing_events_sync, user_id),
            compute_four_week_trends(user_id),
            run_db(_engagement_summary_sync, user_id),
            run_db(_recent_dynamic_states_sync, user_id),
            get_case_id_for_user(user_id),
        )
        state = compute_dynamic_distress_state(
            events=events,
            assessment_trends=trends,
            engagement=engagement,
            previous_states=previous_states,
        )
        stored_id = await run_db(_store_dynamic_distress_state_sync, user_id, case_id, event_id, state)
        await run_db(_create_monitoring_alert_if_needed_sync, user_id, case_id, stored_id, state)
        await run_db(_ensure_monitoring_schedule_sync, user_id, case_id)
        return DynamicDistressStateResponse(**state)
    finally:
        _log_safe_timing("distress_computation", started_at)


async def get_latest_dynamic_distress_state(user_id: int) -> Optional[DynamicDistressStateResponse]:
    rows = await run_db(_recent_dynamic_states_sync, user_id, 1)
    if not rows:
        return None
    row = dict(rows[0])
    indicators = row.get("contributing_indicators")
    if isinstance(indicators, str):
        try:
            indicators = json.loads(indicators)
        except json.JSONDecodeError:
            indicators = []
    return DynamicDistressStateResponse(
        state=row["state"], trajectory=row["trajectory"], confidence=row["confidence"],
        contributing_indicators=indicators or [], requires_human_review=bool(row.get("requires_human_review")),
        computed_at=row["computed_at"],
    )


async def get_longitudinal_monitoring_payload(user_id: int) -> Dict[str, Any]:
    """Return owned, compact monitoring data for report/analytics endpoints."""

    state, engagement, events = await asyncio.gather(
        get_latest_dynamic_distress_state(user_id),
        run_db(_engagement_summary_sync, user_id),
        run_db(_recent_wellbeing_events_sync, user_id, 12),
    )
    voice_indicators: List[Dict[str, Any]] = []
    for event in events:
        signals = event.get("structured_signals")
        if isinstance(signals, str):
            try:
                signals = json.loads(signals)
            except json.JSONDecodeError:
                signals = {}
        if str(event.get("source_type")) != "voice" or not isinstance(signals, dict):
            continue
        primary = signals.get("primary_emotion")
        if isinstance(primary, str) and primary.strip():
            voice_indicators.append({
                "timestamp": event.get("timestamp"),
                "primary_emotion": primary.strip(),
                "confidence": signals.get("emotion_confidence"),
                "acoustic_available": bool(signals.get("acoustic_available")),
            })
    return {
        "dynamic_distress": state.model_dump(mode="json") if state else None,
        "engagement": Phase2EngagementSummary(**engagement).model_dump(mode="json"),
        "voice_indicators": voice_indicators[:6],
    }


# ---------------------------------------------------------------------------
# Phase 3: authorized support operations
# ---------------------------------------------------------------------------

_STAFF_ROLES = {PlatformRole.COUNSELLOR, PlatformRole.AUTHORIZED_OFFICIAL, PlatformRole.ADMIN}
_CASE_PERMISSION_LEVEL = {"case_summary": 1, "intervention_update": 2, "case_manage": 3}


def _decode_json_array(value: object) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [dict(item) for item in decoded if isinstance(item, dict)] if isinstance(decoded, list) else []
    return []


def _assert_staff(principal: AuthenticatedPrincipal) -> None:
    if principal.role not in _STAFF_ROLES:
        raise HTTPException(status_code=403, detail="This account role is not permitted.")


def _required_permission_names(permission: str) -> tuple[str, ...]:
    minimum = _CASE_PERMISSION_LEVEL[permission]
    return tuple(name for name, level in _CASE_PERMISSION_LEVEL.items() if level >= minimum)


def _has_case_access_sync(principal: AuthenticatedPrincipal, case_id: str, permission: str = "case_summary") -> bool:
    """Check explicit staff authorization without opening private history.

    A case assignment gives a counsellor summary/intervention access only. All
    other access, including every administrator's case access, must come from
    an active case-level grant. Geography grants never satisfy this check.
    """

    if principal.role not in _STAFF_ROLES or permission not in _CASE_PERMISSION_LEVEL:
        return False
    with get_conn() as conn:
        if principal.role == PlatformRole.COUNSELLOR and permission in {"case_summary", "intervention_update"}:
            assignment = conn.execute(
                """SELECT 1 FROM case_assignments
                   WHERE case_id=%s AND assigned_to_user_id=%s AND status='active'
                   LIMIT 1""",
                (case_id, principal.user_id),
            ).fetchone()
            if assignment is not None:
                return True
        row = conn.execute(
            """SELECT 1 FROM case_authorizations
               WHERE case_id=%s AND staff_user_id=%s AND revoked_at IS NULL
                 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                 AND permission = ANY(%s)
               LIMIT 1""",
            (case_id, principal.user_id, list(_required_permission_names(permission))),
        ).fetchone()
    return row is not None


async def require_case_access(principal: AuthenticatedPrincipal, case_id: str, permission: str = "case_summary") -> None:
    _assert_staff(principal)
    allowed = await run_db(_has_case_access_sync, principal, case_id, permission)
    if not allowed:
        # A uniform not-found response avoids confirming unrelated case IDs.
        raise HTTPException(status_code=404, detail="Case not available.")


def _case_queue_rows_sync(principal: AuthenticatedPrincipal, limit: int, offset: int = 0) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                c.id AS case_id,
                c.public_case_reference,
                c.state_ut,
                c.district,
                COALESCE(state_row.state, 'unknown') AS priority,
                COALESCE(state_row.trajectory, 'unknown') AS trajectory,
                state_row.computed_at AS last_evaluated_at,
                COALESCE(alert_row.reason, 'Human review is pending for this case.') AS reason_summary,
                alert_row.id AS alert_id,
                review_row.reviewed_at AS last_reviewed_at,
                CASE WHEN assignment_row.assigned_to_user_id IS NULL THEN NULL ELSE 'Assigned support professional' END AS assigned_professional,
                CASE
                    WHEN alert_row.id IS NOT NULL THEN 'Review alert'
                    WHEN intervention_row.id IS NOT NULL THEN 'Update intervention'
                    ELSE NULL
                END AS pending_action
            FROM case_contexts c
            LEFT JOIN LATERAL (
                SELECT state, trajectory, computed_at
                FROM dynamic_distress_states
                WHERE case_id=c.id
                ORDER BY computed_at DESC, id DESC LIMIT 1
            ) state_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT id, reason
                FROM monitoring_alerts
                WHERE case_id=c.id AND status='open' AND requires_human_review
                ORDER BY created_at DESC, id DESC LIMIT 1
            ) alert_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT reviewed_at FROM alert_reviews
                WHERE case_id=c.id ORDER BY reviewed_at DESC, id DESC LIMIT 1
            ) review_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT assigned_to_user_id FROM case_assignments
                WHERE case_id=c.id AND status='active' ORDER BY updated_at DESC, id DESC LIMIT 1
            ) assignment_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT id FROM intervention_records
                WHERE case_id=c.id AND status IN ('planned', 'in_progress')
                ORDER BY updated_at DESC, id DESC LIMIT 1
            ) intervention_row ON TRUE
            WHERE c.monitoring_status='active'
              AND (
                EXISTS (
                    SELECT 1 FROM case_authorizations ca
                    WHERE ca.case_id=c.id AND ca.staff_user_id=%s AND ca.revoked_at IS NULL
                      AND (ca.expires_at IS NULL OR ca.expires_at > CURRENT_TIMESTAMP)
                )
                OR (
                    %s = 'counsellor' AND EXISTS (
                        SELECT 1 FROM case_assignments own_assignment
                        WHERE own_assignment.case_id=c.id AND own_assignment.assigned_to_user_id=%s
                          AND own_assignment.status='active'
                    )
                )
              )
            ORDER BY
                CASE COALESCE(state_row.state, 'unknown')
                    WHEN 'critical' THEN 5 WHEN 'high' THEN 4 WHEN 'elevated' THEN 3
                    WHEN 'watch' THEN 2 WHEN 'stable' THEN 1 ELSE 0 END DESC,
                state_row.computed_at DESC NULLS LAST,
                c.id DESC
            LIMIT %s OFFSET %s
            """,
            (principal.user_id, principal.role.value, principal.user_id, limit, offset),
        ).fetchall()


def _case_detail_row_sync(case_id: str) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT c.id AS case_id, c.public_case_reference, c.state_ut, c.district,
                   COALESCE(state_row.state, 'unknown') AS priority,
                   COALESCE(state_row.trajectory, 'unknown') AS trajectory,
                   state_row.confidence, state_row.contributing_indicators,
                   state_row.computed_at AS last_evaluated_at,
                   COALESCE(alert_row.reason, 'No pending human review task.') AS reason_summary,
                   alert_row.id AS alert_id,
                   review_row.reviewed_at AS last_reviewed_at,
                   CASE WHEN assignment_row.assigned_to_user_id IS NULL THEN NULL ELSE 'Assigned support professional' END AS assigned_professional,
                   CASE WHEN alert_row.id IS NOT NULL THEN 'Review alert' ELSE NULL END AS pending_action,
                   period_row.data_period_start, period_row.data_period_end
            FROM case_contexts c
            LEFT JOIN LATERAL (
                SELECT state, trajectory, confidence, contributing_indicators, computed_at
                FROM dynamic_distress_states WHERE case_id=c.id
                ORDER BY computed_at DESC, id DESC LIMIT 1
            ) state_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT reason FROM monitoring_alerts
                WHERE case_id=c.id AND status='open' AND requires_human_review
                ORDER BY created_at DESC, id DESC LIMIT 1
            ) alert_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT reviewed_at FROM alert_reviews WHERE case_id=c.id
                ORDER BY reviewed_at DESC, id DESC LIMIT 1
            ) review_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT assigned_to_user_id FROM case_assignments WHERE case_id=c.id AND status='active'
                ORDER BY updated_at DESC, id DESC LIMIT 1
            ) assignment_row ON TRUE
            LEFT JOIN LATERAL (
                SELECT MIN(timestamp) AS data_period_start, MAX(timestamp) AS data_period_end
                FROM wellbeing_events WHERE case_id=c.id
            ) period_row ON TRUE
            WHERE c.id=%s
            """,
            (case_id,),
        ).fetchone()


def _case_assessments_sync(case_id: str) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT DISTINCT ON (m.assessment_type) m.assessment_type, m.total, m.completed_at
            FROM screening_measurements m
            JOIN case_contexts c ON c.user_id=m.user_id
            WHERE c.id=%s
            ORDER BY m.assessment_type, m.completed_at DESC NULLS LAST, m.id DESC
            """,
            (case_id,),
        ).fetchall()


def _safe_audit_sync(
    actor_user_id: int,
    action: str,
    *,
    case_id: Optional[str] = None,
    resource_type: str,
    resource_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an audit event with a digest, never including note/chat content."""

    safe_metadata = {str(key)[:80]: str(value)[:120] for key, value in (metadata or {}).items()}
    event_id = uuid.uuid4()
    occurred_at = datetime.now(timezone.utc)
    with get_conn() as conn:
        previous = conn.execute("SELECT event_digest FROM audit_events ORDER BY id DESC LIMIT 1").fetchone()
        previous_digest = previous["event_digest"] if previous else ""
        digest_source = json.dumps(
            {
                "event_id": str(event_id), "occurred_at": occurred_at.isoformat(), "actor": actor_user_id,
                "action": action, "case": case_id, "resource_type": resource_type,
                "resource_id": resource_id, "metadata": safe_metadata, "previous": previous_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        event_digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        conn.execute(
            """INSERT INTO audit_events (
                   event_id, occurred_at, actor_user_id, action, case_id, resource_type,
                   resource_id, metadata, previous_digest, event_digest
               ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (str(event_id), occurred_at, actor_user_id, action, case_id, resource_type,
             resource_id, json.dumps(safe_metadata), previous_digest or None, event_digest),
        )
        conn.commit()


async def write_safe_audit(*args: Any, **kwargs: Any) -> None:
    await run_db(_safe_audit_sync, *args, **kwargs)


def _is_staff_user_sync(user_id: int) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT role FROM users WHERE id=%s", (user_id,)).fetchone()
    return row is not None and str(row["role"] or "user") in {role.value for role in _STAFF_ROLES}


def _assign_case_sync(case_id: str, actor_user_id: int, assigned_to_user_id: int) -> DatabaseRow:
    with get_conn() as conn:
        conn.execute(
            "UPDATE case_assignments SET status='reassigned', updated_at=CURRENT_TIMESTAMP WHERE case_id=%s AND status='active'",
            (case_id,),
        )
        row = conn.execute(
            """INSERT INTO case_assignments (case_id, assigned_to_user_id, assigned_by)
               VALUES (%s,%s,%s)
               RETURNING case_id, assigned_to_user_id, status, updated_at""",
            (case_id, assigned_to_user_id, actor_user_id),
        ).fetchone()
        conn.commit()
        return row


def _alert_case_id_sync(alert_id: int) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT case_id FROM monitoring_alerts WHERE id=%s", (alert_id,)).fetchone()
    return str(row["case_id"]) if row and row.get("case_id") else None


def _review_alert_sync(alert_id: int, case_id: str, reviewer_user_id: int, decision: str, note: Optional[str]) -> DatabaseRow:
    with get_conn() as conn:
        review = conn.execute(
            """INSERT INTO alert_reviews (alert_id, case_id, reviewer_user_id, decision, note)
               VALUES (%s,%s,%s,%s,%s)
               RETURNING id, alert_id, decision, reviewed_at""",
            (alert_id, case_id, reviewer_user_id, decision, note),
        ).fetchone()
        next_status = "closed" if decision in {"not_concerning", "false_positive"} else "acknowledged"
        conn.execute(
            "UPDATE monitoring_alerts SET status=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s",
            (next_status, alert_id),
        )
        conn.commit()
        return review


def _list_interventions_sync(case_id: str) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """SELECT id, case_id, intervention_type, status, created_by, assigned_to,
                      created_at, updated_at, notes, outcome
               FROM intervention_records WHERE case_id=%s ORDER BY updated_at DESC, id DESC""",
            (case_id,),
        ).fetchall()


def _create_intervention_sync(case_id: str, actor_user_id: int, data: InterventionCreate) -> DatabaseRow:
    with get_conn() as conn:
        row = conn.execute(
            """INSERT INTO intervention_records (case_id, intervention_type, created_by, assigned_to, notes)
               VALUES (%s,%s,%s,%s,%s)
               RETURNING id, case_id, intervention_type, status, created_by, assigned_to, created_at, updated_at, notes, outcome""",
            (case_id, data.intervention_type.value, actor_user_id, data.assigned_to_user_id, data.notes),
        ).fetchone()
        conn.commit()
        return row


def _update_intervention_sync(case_id: str, intervention_id: int, data: InterventionUpdate) -> Optional[DatabaseRow]:
    fields: List[str] = []
    values: List[Any] = []
    if data.status is not None:
        fields.append("status=%s")
        values.append(data.status.value)
    if data.assigned_to_user_id is not None:
        fields.append("assigned_to=%s")
        values.append(data.assigned_to_user_id)
    if data.notes is not None:
        fields.append("notes=%s")
        values.append(data.notes)
    if data.outcome is not None:
        fields.append("outcome=%s")
        values.append(data.outcome)
    if not fields:
        return None
    with get_conn() as conn:
        row = conn.execute(
            f"""UPDATE intervention_records SET {', '.join(fields)}, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s AND case_id=%s
                RETURNING id, case_id, intervention_type, status, created_by, assigned_to, created_at, updated_at, notes, outcome""",
            [*values, intervention_id, case_id],
        ).fetchone()
        conn.commit()
        return row


def _list_case_notes_sync(case_id: str) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, created_at, created_by, note FROM case_notes WHERE case_id=%s ORDER BY created_at DESC, id DESC LIMIT 100",
            (case_id,),
        ).fetchall()


def _create_case_note_sync(case_id: str, actor_user_id: int, note: str) -> DatabaseRow:
    with get_conn() as conn:
        row = conn.execute(
            """INSERT INTO case_notes (case_id, created_by, note) VALUES (%s,%s,%s)
               RETURNING id, created_at, created_by, note""",
            (case_id, actor_user_id, note),
        ).fetchone()
        conn.commit()
        return row


def _aggregate_scope_sync(principal: AuthenticatedPrincipal, level: str, state_ut: Optional[str], district: Optional[str]) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT 1 FROM official_geography_scopes
               WHERE user_id=%s AND revoked_at IS NULL
                 AND scope_level=%s
                 AND (state_ut IS NOT DISTINCT FROM %s)
                 AND (district IS NOT DISTINCT FROM %s)
               LIMIT 1""",
            (principal.user_id, level, state_ut, district),
        ).fetchone()
    return row is not None


def _operational_aggregate_counts_sync(state_ut: Optional[str], district: Optional[str]) -> Dict[str, int]:
    filters = ["c.monitoring_status='active'"]
    values: List[Any] = []
    if state_ut:
        filters.append("c.state_ut=%s")
        values.append(state_ut)
    if district:
        filters.append("c.district=%s")
        values.append(district)
    where_clause = " AND ".join(filters)
    with get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(DISTINCT c.id)::int AS active_monitored_cases,
                COUNT(DISTINCT alert_row.id) FILTER (WHERE alert_row.status='open')::int AS new_alerts,
                COUNT(DISTINCT c.id) FILTER (WHERE state_row.state IN ('high', 'critical'))::int AS high_priority_cases,
                COUNT(DISTINCT alert_row.id) FILTER (WHERE alert_row.status='open' AND alert_row.requires_human_review)::int AS awaiting_review,
                COUNT(DISTINCT intervention_row.id) FILTER (WHERE intervention_row.status IN ('planned', 'in_progress'))::int AS active_interventions
            FROM case_contexts c
            LEFT JOIN LATERAL (
                SELECT state FROM dynamic_distress_states WHERE case_id=c.id
                ORDER BY computed_at DESC, id DESC LIMIT 1
            ) state_row ON TRUE
            LEFT JOIN monitoring_alerts alert_row ON alert_row.case_id=c.id
            LEFT JOIN intervention_records intervention_row ON intervention_row.case_id=c.id
            WHERE {where_clause}
            """,
            values,
        ).fetchone()
    return {key: int(row.get(key) or 0) for key in (
        "active_monitored_cases", "new_alerts", "high_priority_cases", "awaiting_review", "active_interventions"
    )}

def _get_user_memories_sync(
    user_id: int,
    limit: int = 30,
) -> List[dict]:

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                memory_key,
                memory_value,
                memory_type,
                confidence,
                source_conversation_id,
                updated_at
            FROM user_memories
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (
                user_id,
                limit,
            ),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


async def get_user_memories(
    user_id: int,
    limit: int = 30,
) -> List[dict]:

    return await run_db(
        _get_user_memories_sync,
        user_id,
        limit,
    )
    
def _upsert_user_memory_sync(
    user_id: int,
    memory_key: str,
    memory_value: str,
    memory_type: str,
    conversation_id: Optional[str],
    confidence: float,
) -> None:

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_memories (
                user_id,
                memory_key,
                memory_value,
                memory_type,
                source_conversation_id,
                confidence
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )

            ON CONFLICT (
                user_id,
                memory_key
            )

            DO UPDATE SET
                memory_value =
                    EXCLUDED.memory_value,

                memory_type =
                    EXCLUDED.memory_type,

                source_conversation_id =
                    EXCLUDED.source_conversation_id,

                confidence =
                    EXCLUDED.confidence,

                updated_at =
                    CURRENT_TIMESTAMP
            """,
            (
                user_id,
                memory_key,
                memory_value,
                memory_type,
                conversation_id,
                confidence,
            ),
        )

        conn.commit()


async def upsert_user_memory(
    user_id: int,
    memory: MemoryCandidate,
    conversation_id: Optional[str],
) -> None:

    await run_db(
        _upsert_user_memory_sync,
        user_id,
        memory.memory_key,
        memory.memory_value,
        memory.memory_type,
        conversation_id,
        memory.confidence,
    )
    
def _memory_words(
    text: str,
) -> set[str]:

    return {
        word
        for word in re.findall(
            r"[A-Za-z0-9_]+",
            text.lower(),
        )
        if len(word) >= 3
    }
    
def select_relevant_memories(
    memories: List[dict],
    current_message: str,
    limit: int = 6,
) -> List[dict]:

    if not memories:
        return []

    query_words = _memory_words(
        current_message
    )

    scored: List[tuple[float, dict]] = []

    for memory in memories:

        searchable = " ".join(
            [
                str(
                    memory.get(
                        "memory_key",
                        "",
                    )
                ),
                str(
                    memory.get(
                        "memory_value",
                        "",
                    )
                ),
                str(
                    memory.get(
                        "memory_type",
                        "",
                    )
                ),
            ]
        )

        memory_words = _memory_words(
            searchable
        )

        overlap = len(
            query_words
            & memory_words
        )

        memory_type = memory.get(
            "memory_type"
        )

        type_bonus = {
            "identity": 3.0,
            "relationship_context": 2.5,
            "ongoing_situation": 2.0,
            "project": 1.8,
            "goal": 1.8,
            "education": 1.5,
            "preference": 1.5,
            "routine": 1.2,
            "conversation_reference": 1.0,
            "general": 0.0,
        }.get(memory_type, 0.0)

        confidence = float(
            memory.get(
                "confidence",
                1.0,
            )
            or 1.0
        )

        score = (
            overlap * 2.0
            + type_bonus
        ) * confidence

        if score > 0:
            scored.append(
                (
                    score,
                    memory,
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        memory
        for _, memory
        in scored[:limit]
    ]
def _memory_search_terms(
    text: str,
) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-zA-Z0-9_'-]{3,}",
            text.lower(),
        )
    }
    
def select_relevant_conversation_summaries(
    summaries: List[DatabaseRow],
    current_message: str,
    limit: int = CROSS_CONVERSATION_SUMMARY_LIMIT,
) -> List[DatabaseRow]:

    if not summaries or limit <= 0:
        return []

    query_terms = _memory_search_terms(
        current_message
    )

    ranked: list[
        tuple[float, int, DatabaseRow]
    ] = []

    for index, summary in enumerate(summaries):
        text = str(
            summary.get("summary_text") or ""
        ).strip()

        if not text:
            continue

        summary_terms = _memory_search_terms(
            text
        )

        overlap = len(
            query_terms & summary_terms
        )

        # Strongly reward actual lexical relevance.
        relevance_score = float(overlap) * 3.0

        # A small recency preference means a recent summary can still
        # provide continuity when the user's message is very short.
        recency_bonus = max(
            0.0,
            1.0 - (index * 0.05),
        )

        score = (
            relevance_score
            + recency_bonus
        )

        ranked.append(
            (
                score,
                index,
                summary,
            )
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    selected: List[DatabaseRow] = []

    total_chars = 0

    for score, _, summary in ranked:

        # If the current message contains useful search terms, avoid
        # injecting unrelated old conversations solely due to recency.
        if query_terms and score <= 1.0:
            continue

        text = str(
            summary.get("summary_text") or ""
        ).strip()

        remaining = (
            CROSS_CONVERSATION_SUMMARY_MAX_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining].rstrip()

        selected.append(
            {
                **summary,
                "summary_text": text,
            }
        )

        total_chars += len(text)

        if len(selected) >= limit:
            break

    return selected

def _get_conversation_summary_sync(
    user_id: int,
    conversation_id: Optional[str],
) -> Optional[DatabaseRow]:

    if not conversation_id:
        return None

    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                conversation_id,
                summary_text,
                summarized_through_message_id,
                updated_at
            FROM conversation_summaries
            WHERE user_id = %s
              AND conversation_id = %s
            LIMIT 1
            """,
            (
                user_id,
                conversation_id,
            ),
        ).fetchone()


async def get_conversation_summary(
    user_id: int,
    conversation_id: Optional[str],
) -> Optional[DatabaseRow]:

    return await run_db(
        _get_conversation_summary_sync,
        user_id,
        conversation_id,
    )
    
def _conversation_summary_source_sync(
    user_id: int,
    conversation_id: str,
    keep_recent: int,
) -> dict:

    with get_conn() as conn:
        summary_row = conn.execute(
            """
            SELECT
                summary_text,
                summarized_through_message_id
            FROM conversation_summaries
            WHERE user_id = %s
              AND conversation_id = %s
            LIMIT 1
            """,
            (
                user_id,
                conversation_id,
            ),
        ).fetchone()

        previous_summary = (
            summary_row["summary_text"]
            if summary_row is not None
            else ""
        )

        summarized_through = (
            int(
                summary_row[
                    "summarized_through_message_id"
                ]
            )
            if summary_row is not None
            else 0
        )

        recent_cutoff = conn.execute(
            """
            SELECT id
            FROM conversation_messages
            WHERE user_id = %s
              AND conversation_id = %s
            ORDER BY id DESC
            OFFSET %s
            LIMIT 1
            """,
            (
                user_id,
                conversation_id,
                keep_recent,
            ),
        ).fetchone()

        if recent_cutoff is None:
            return {
                "previous_summary": previous_summary,
                "messages": [],
                "summarized_through": summarized_through,
            }

        cutoff_id = int(
            recent_cutoff["id"]
        )

        rows = conn.execute(
            """
            SELECT
                id,
                role,
                content
            FROM conversation_messages
            WHERE user_id = %s
              AND conversation_id = %s
              AND id > %s
              AND id <= %s
            ORDER BY id ASC
            """,
            (
                user_id,
                conversation_id,
                summarized_through,
                cutoff_id,
            ),
        ).fetchall()

    return {
        "previous_summary": previous_summary,
        "messages": [
            dict(row)
            for row in rows
        ],
        "summarized_through": summarized_through,
    }
    
def _save_conversation_summary_sync(
    user_id: int,
    conversation_id: str,
    summary_text: str,
    summarized_through_message_id: int,
) -> None:

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversation_summaries (
                conversation_id,
                user_id,
                summary_text,
                summarized_through_message_id
            )
            VALUES (
                %s,
                %s,
                %s,
                %s
            )

            ON CONFLICT (conversation_id)
            DO UPDATE SET
                summary_text =
                    EXCLUDED.summary_text,

                summarized_through_message_id =
                    GREATEST(
                        conversation_summaries.summarized_through_message_id,
                        EXCLUDED.summarized_through_message_id
                    ),

                updated_at =
                    CURRENT_TIMESTAMP
            """,
            (
                conversation_id,
                user_id,
                summary_text,
                summarized_through_message_id,
            ),
        )

        conn.commit()
        
def _get_other_conversation_summaries_sync(
    user_id: int,
    current_conversation_id: Optional[str],
    limit: int,
) -> List[DatabaseRow]:

    with get_conn() as conn:

        if current_conversation_id:
            rows = conn.execute(
                """
                SELECT
                    cs.conversation_id,
                    cs.summary_text,
                    cs.updated_at
                FROM conversation_summaries cs
                JOIN conversations c
                    ON c.conversation_id = cs.conversation_id
                WHERE c.user_id = %s
                  AND cs.summary_text IS NOT NULL
                  AND TRIM(cs.summary_text) <> ''
                  AND cs.conversation_id <> %s
                ORDER BY cs.updated_at DESC
                LIMIT %s
                """,
                (
                    user_id,
                    current_conversation_id,
                    limit,
                ),
            ).fetchall()

        else:
            rows = conn.execute(
                """
                SELECT
                    cs.conversation_id,
                    cs.summary_text,
                    cs.updated_at
                FROM conversation_summaries cs
                JOIN conversations c
                    ON c.conversation_id = cs.conversation_id
                WHERE c.user_id = %s
                  AND cs.summary_text IS NOT NULL
                  AND TRIM(cs.summary_text) <> ''
                ORDER BY cs.updated_at DESC
                LIMIT %s
                """,
                (
                    user_id,
                    limit,
                ),
            ).fetchall()

    return rows


async def get_other_conversation_summaries(
    user_id: int,
    current_conversation_id: Optional[str],
    limit: int = CROSS_CONVERSATION_SUMMARY_CANDIDATES,
) -> List[DatabaseRow]:
    return await run_db(
        _get_other_conversation_summaries_sync,
        user_id,
        current_conversation_id,
        limit,
    )
    
def _trim_text(
    value: Optional[str],
    max_chars: int,
) -> str:

    text = str(value or "").strip()

    if not text or max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip()

def _trim_lines_to_budget(
    lines: List[str],
    max_chars: int,
) -> List[str]:

    if max_chars <= 0:
        return []

    result: List[str] = []
    used = 0

    for line in lines:
        text = str(line).strip()

        if not text:
            continue

        remaining = max_chars - used

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining].rstrip()

        if text:
            result.append(text)
            used += len(text)

    return result

def _export_user_data_sync(
    user_id: int,
) -> dict:
    with get_conn() as conn:

        profile = conn.execute(
            """
            SELECT *
            FROM user_profiles
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()

        conversations = conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE user_id = %s
            ORDER BY timestamp ASC
            """,
            (user_id,),
        ).fetchall()

        messages = conn.execute(
            """
            SELECT *
            FROM conversation_messages
            WHERE user_id = %s
            ORDER BY timestamp ASC
            """,
            (user_id,),
        ).fetchall()

        memories = conn.execute(
            """
            SELECT *
            FROM user_memories
            WHERE user_id = %s
            ORDER BY timestamp ASC
            """,
            (user_id,),
        ).fetchall()

        measurements = conn.execute(
            """
            SELECT *
            FROM screening_measurements
            WHERE user_id = %s
            ORDER BY timestamp ASC
            """,
            (user_id,),
        ).fetchall()

        reports = conn.execute(
            """
            SELECT *
            FROM wellbeing_reports
            WHERE user_id = %s
            ORDER BY timestamp ASC
            """,
            (user_id,),
        ).fetchall()

    return {
        "profile": dict(profile) if profile else None,
        "conversations": [
            dict(row) for row in conversations
        ],
        "messages": [
            dict(row) for row in messages
        ],
        "memories": [
            dict(row) for row in memories
        ],
        "assessment_measurements": [
            dict(row) for row in measurements
        ],
        "reports": [
            dict(row) for row in reports
        ],
    }

def _delete_conversation_sync(
    user_id: int,
    conversation_id: str,
) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM conversations
            WHERE user_id = %s
              AND coversation_id = %s
            """,
            (
                user_id,
                conversation_id,
            ),
        ).fetchone()

        if row is None:
            return False

        conn.execute(
            """
            DELETE FROM user_memories
            WHERE user_id = %s
              AND source_conversation_id = %s
            """,
            (
                user_id,
                conversation_id,
            ),
        )

        conn.execute(
            """
            DELETE FROM conversation_summaries
            WHERE user_id = %s
              AND conversation_id = %s
            """,
            (
                user_id,
                conversation_id,
            ),
        )

        conn.execute(
            """
            DELETE FROM conversation_messages
            WHERE user_id = %s
              AND conversation_id = %s
            """,
            (
                user_id,
                conversation_id,
            ),
        )

        conn.execute(
            """
            DELETE FROM conversations
            WHERE user_id = %s
              AND id = %s
            """,
            (
                user_id,
                conversation_id,
            ),
        )

        conn.commit()

    return True

def _delete_all_conversations_sync(
    user_id: int,
) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM conversations
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()

        total = int(
            row["total"]
            if row
            else 0
        )

        conn.execute(
            """
            DELETE FROM user_memories
            WHERE user_id = %s
            """,
            (user_id,),
        )

        conn.execute(
            """
            DELETE FROM conversation_summaries
            WHERE user_id = %s
            """,
            (user_id,),
        )

        conn.execute(
            """
            DELETE FROM conversation_messages
            WHERE user_id = %s
            """,
            (user_id,),
        )

        conn.execute(
            """
            DELETE FROM conversations
            WHERE user_id = %s
            """,
            (user_id,),
        )

        conn.commit()

    return total

def _delete_saved_reports_sync(
    user_id: int,
) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM wellbeing_reports
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchone()

        deleted_count = int(
            row["total"] if row else 0
        )

        conn.execute(
            """
            DELETE FROM wellbeing_reports
            WHERE user_id = %s
            """,
            (user_id,),
        )

        conn.commit()

    return deleted_count

def _delete_account_sync(user_id: int) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:

            # ---------------------------------------------------------
            # Screening children first
            # ---------------------------------------------------------

            cur.execute(
                """
                DELETE FROM screening_session_items
                WHERE session_id IN (
                    SELECT session_id
                    FROM screening_sessions
                    WHERE user_id = %s
                )
                """,
                (user_id,),
            )

            # Measurements may reference screening sessions.
            cur.execute(
                """
                DELETE FROM screening_measurements
                WHERE user_id = %s
                """,
                (user_id,),
            )

            cur.execute(
                """
                DELETE FROM screening_sessions
                WHERE user_id = %s
                """,
                (user_id,),
            )

            # ---------------------------------------------------------
            # Conversation-derived data
            # ---------------------------------------------------------

            cur.execute(
                """
                DELETE FROM conversation_summaries
                WHERE user_id = %s
                """,
                (user_id,),
            )

            cur.execute(
                """
                DELETE FROM user_memories
                WHERE user_id = %s
                """,
                (user_id,),
            )

            cur.execute(
                """
                DELETE FROM conversation_messages
                WHERE user_id = %s
                """,
                (user_id,),
            )

            cur.execute(
                """
                DELETE FROM conversations
                WHERE user_id = %s
                """,
                (user_id,),
            )

            # ---------------------------------------------------------
            # Wellbeing/analytics data
            # ---------------------------------------------------------

            user_owned_tables = (
                "assessment_records",
                "check_ins",
                "functional_impairments",
                "sleep_reports",
                "weekly_summaries",
                "risk_assessments",
                "recommendation_records",
                "follow_ups",
                "escalation_records",
                "wellbeing_reports",
                "resource_favorites",
                "suggested_states",
                "questionnaire_state",
                "emotion_records",
                "user_profiles",
            )

            for table in user_owned_tables:
                cur.execute(
                    f"DELETE FROM {table} WHERE user_id = %s",
                    (user_id,),
                )

            # ---------------------------------------------------------
            # Authentication/privacy data
            # ---------------------------------------------------------

            cur.execute(
                """
                DELETE FROM auth_sessions
                WHERE user_id = %s
                """,
                (user_id,),
            )

            # Add your actual privacy acknowledgement table here if
            # it has user_id and isn't already ON DELETE CASCADE.

            # ---------------------------------------------------------
            # User LAST
            # ---------------------------------------------------------

            cur.execute(
                """
                DELETE FROM users
                WHERE id = %s
                """,
                (user_id,),
            )

        conn.commit()

def format_profile_for_llm(
    profile: Optional[dict],
) -> List[str]:

    if not profile:
        return []

    lines: List[str] = []

    display_name = (
        profile.get("display_name")
        or profile.get("username")
    )

    if display_name:
        lines.append(
            f"name: {display_name}"
        )

    preferred_language = (
        profile.get(
            "preferred_language"
        )
    )

    if preferred_language:
        lines.append(
            "stored_preferred_language: "
            f"{preferred_language}"
        )

    return lines

def format_memories_for_llm(
    memories: List[dict],
) -> List[str]:

    return [
        (
            f"- [{memory['memory_type']}] "
            f"{memory['memory_value']}"
        )
        for memory in memories
    ]

def _save_analysis_observations_sync(
    user_id: int,
    sleep_hours: Optional[float],
    impairments: Sequence[FunctionalImpairment],
) -> None:
    with get_conn() as conn:
        if sleep_hours is not None:
            conn.execute(
                "INSERT INTO sleep_reports (user_id, hours) VALUES (%s, %s)",
                (user_id, sleep_hours),
            )
        for impairment in impairments:
            conn.execute(
                """
                INSERT INTO functional_impairments (user_id, area, severity, evidence)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, impairment.area, impairment.severity, impairment.evidence),
            )
        conn.commit()


async def save_analysis_observations(
    user_id: int,
    sleep_hours: Optional[float],
    impairments: Sequence[FunctionalImpairment],
) -> None:
    await run_db(_save_analysis_observations_sync, user_id, sleep_hours, impairments)


def _save_passive_screening_evidence_sync(
    user_id: int,
    observations: Sequence[tuple[str, SymptomItem]],
) -> None:
    """Persist explicitly scored chat evidence without changing assessment state.

    Deliberate questionnaires are controlled only by screening_sessions and
    /screening endpoints.  Passive extraction must never create, resume, pause,
    or complete a questionnaire session.
    """
    if not observations:
        return
    with get_conn() as conn:
        for scale, item in observations:
            conn.execute(
                "INSERT INTO assessment_records (user_id, assessment_type, item_id, score, evidence) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, scale, item.item_id, item.score, item.evidence),
            )
        conn.commit()


async def save_passive_screening_evidence(
    user_id: int,
    observations: Sequence[tuple[str, SymptomItem]],
) -> None:
    await run_db(_save_passive_screening_evidence_sync, user_id, observations)


def _save_emotion_sync(
    user_id: int,
    conversation_id: Optional[str],
    source: str,
    primary_emotion: str,
    confidence: Optional[float],
    severity: Optional[str],
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO emotion_records "
            "(user_id, conversation_id, source, primary_emotion, confidence, severity) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (user_id, conversation_id, source, primary_emotion, confidence, severity),
        )
        conn.commit()


async def save_emotion(
    user_id: int,
    conversation_id: Optional[str],
    source: str,
    primary_emotion: str,
    confidence: Optional[float],
    severity: Optional[str],
) -> None:
    await run_db(
        _save_emotion_sync,
        user_id,
        conversation_id,
        source,
        primary_emotion,
        confidence,
        severity,
    )


def _create_conversation_sync(user_id: int) -> str:
    conversation_id = f"GSH-CONV-{uuid.uuid4().hex[:12].upper()}"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, user_id) VALUES (%s,%s)",
            (conversation_id, user_id),
        )
        conn.commit()
    return conversation_id


async def create_conversation(user_id: int) -> str:
    return await run_db(_create_conversation_sync, user_id)


def _verify_conversation_sync(user_id: int, conversation_id: str) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT conversation_id FROM conversations WHERE user_id=%s AND conversation_id=%s",
            (user_id, conversation_id),
        ).fetchone()


async def verify_conversation(user_id: int, conversation_id: str) -> Optional[DatabaseRow]:
    return await run_db(_verify_conversation_sync, user_id, conversation_id)


def _recent_messages_sync(user_id, limit, conversation_id=None) -> List[DatabaseRow]:
    with get_conn() as conn:
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM conversation_messages "
                "WHERE user_id=%s AND conversation_id=%s ORDER BY timestamp DESC, id DESC LIMIT %s",
                (user_id, conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM conversation_messages "
                "WHERE user_id=%s AND conversation_id IS NULL ORDER BY timestamp DESC, id DESC LIMIT %s",
                (user_id, limit),
            ).fetchall()
    return list(reversed(rows))


async def get_recent_messages(user_id, limit=MAX_RECENT_MESSAGES, conversation_id=None):
    return await run_db(_recent_messages_sync, user_id, limit, conversation_id)


# --- new chat retrieval (section G) -------------------------------------------

def _own_conversations_sync(
    user_id: int,
    limit: int,
    before: Optional[tuple[datetime, int]],
) -> List[DatabaseRow]:
    before_timestamp, before_id = before if before else (None, None)
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT c.id, c.conversation_id, c.created_at,
                   COUNT(m.id) AS message_count,
                   COALESCE(MAX(m.timestamp), c.created_at) AS last_activity_at,
                   (
                     SELECT cm.content
                     FROM conversation_messages AS cm
                     WHERE cm.user_id = c.user_id
                       AND cm.conversation_id = c.conversation_id
                       AND cm.role = 'user'
                     ORDER BY cm.timestamp DESC, cm.id DESC
                     LIMIT 1
                   ) AS preview
            FROM conversations c
            LEFT JOIN conversation_messages m ON m.conversation_id = c.conversation_id AND m.user_id = c.user_id
            WHERE c.user_id = %s
            GROUP BY c.id, c.conversation_id, c.created_at
            HAVING %s::timestamptz IS NULL
                OR (COALESCE(MAX(m.timestamp), c.created_at), c.id) < (%s::timestamptz, %s)
            ORDER BY last_activity_at DESC, c.id DESC
            LIMIT %s
            """,
            (user_id, before_timestamp, before_timestamp, before_id, limit),
        ).fetchall()


def _last_preview_sync(user_id: int, conversation_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content FROM conversation_messages WHERE user_id=%s AND conversation_id=%s "
            "AND role='user' ORDER BY timestamp DESC, id DESC LIMIT 1",
            (user_id, conversation_id),
        ).fetchone()
    return row["content"] if row else ""


def _conversation_messages_sync(
    user_id: int,
    conversation_id: str,
    limit: int = 100,
    before_id: Optional[int] = None,
) -> List[DatabaseRow]:

    with get_conn() as conn:
        if before_id is not None:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp
                FROM conversation_messages
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND id < %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user_id, conversation_id, before_id, limit),
            ).fetchall()

        else:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp
                FROM conversation_messages
                WHERE user_id = %s
                  AND conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user_id, conversation_id, limit),
            ).fetchall()

    return list(reversed(rows))


def _suggested_state_sync(user_id: int) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM suggested_states WHERE user_id=%s AND dismissed=FALSE "
            "ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def _dismiss_suggested_sync(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE suggested_states SET dismissed=TRUE WHERE user_id=%s AND dismissed=FALSE",
            (user_id,),
        )
        conn.commit()


def _store_suggested_sync(
    user_id: int, conversation_id: Optional[str],
    replies: Optional[List[str]] = None, actions: Optional[List[str]] = None,
) -> None:
    if not replies and not actions:
        return
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO suggested_states (user_id, conversation_id, suggested_replies, actions) "
            "VALUES (%s,%s,%s,%s)",
            (user_id, conversation_id,
             json.dumps(replies or []), json.dumps(actions or [])),
        )
        conn.commit()



# ---------------------------------------------------------------------------
# Weekly summary generation (aggregates now source from completed measurements)
# ---------------------------------------------------------------------------

WEEKLY_SUMMARY_PROMPT = (
    "You are a backend summarisation component for a mental-health screening "
    "platform. Rewrite the structured weekly facts below into one compact "
    "paragraph (max 120 words) for longitudinal context.\n"
    "Rules: use ONLY the supplied facts; never invent symptoms, scores or "
    "quotations; never diagnose; never give treatment advice; plain text only."
)


def _weekly_facts_sync(user_id: int, week_start: str, week_end: str) -> dict:
    start, end = _utc_day_bounds(week_start, week_end)
    with get_conn() as conn:
        messages = conn.execute(
            "SELECT role, COUNT(*) AS n FROM conversation_messages "
            "WHERE user_id=%s AND timestamp >= %s AND timestamp < %s GROUP BY role",
            (user_id, start, end),
        ).fetchall()

        measurements = conn.execute(
            "SELECT assessment_type, total, completed_at FROM screening_measurements "
            "WHERE user_id=%s AND completed_at >= %s AND completed_at < %s ORDER BY completed_at",
            (user_id, start, end),
        ).fetchall()

        # per-scale latest measurement (any time) for "current state"
        latest_totals: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None, "PSS-10": None}
        for scale in latest_totals:
            row = conn.execute(
                "SELECT total FROM screening_measurements WHERE user_id=%s AND assessment_type=%s "
                "ORDER BY completed_at DESC, id DESC LIMIT 1",
                (user_id, scale),
            ).fetchone()
            if row is not None:
                latest_totals[scale] = row["total"]

        impairments = conn.execute(
            "SELECT area, severity, COUNT(*) AS n FROM functional_impairments "
            "WHERE user_id=%s AND timestamp >= %s AND timestamp < %s GROUP BY area, severity ORDER BY n DESC",
            (user_id, start, end),
        ).fetchall()

        sleep = conn.execute(
            "SELECT COUNT(*) AS n, MIN(hours) AS min_h, MAX(hours) AS max_h, AVG(hours) AS avg_h "
            "FROM sleep_reports WHERE user_id=%s AND timestamp >= %s AND timestamp < %s",
            (user_id, start, end),
        ).fetchone()

        checkins = conn.execute(
            "SELECT COUNT(*) AS n, AVG(mood_score) AS avg_mood FROM check_ins "
            "WHERE user_id=%s AND checkin_date BETWEEN %s AND %s AND mood_score IS NOT NULL",
            (user_id, week_start, week_end),
        ).fetchone()

        emergencies = conn.execute(
            "SELECT COUNT(*) AS n FROM escalation_records WHERE user_id=%s AND timestamp >= %s AND timestamp < %s",
            (user_id, start, end),
        ).fetchone()

    return {
        "messages": {r["role"]: r["n"] for r in messages},
        "measurements": [
            {"scale": r["assessment_type"], "total": r["total"],
             "completed_at": r["completed_at"]} for r in measurements
        ],
        "latest_totals": latest_totals,
        "impairments": [
            {"area": r["area"], "severity": r["severity"], "mentions": r["n"]} for r in impairments
        ],
        "sleep": {
            "reports": sleep["n"],
            "min_hours": sleep["min_h"],
            "max_hours": sleep["max_h"],
            "avg_hours": round(sleep["avg_h"], 1) if sleep["avg_h"] is not None else None,
        },
        "checkins": {
            "count": checkins["n"],
            "avg_mood": round(checkins["avg_mood"], 1) if checkins["avg_mood"] is not None else None,
        },
        "emergency_turns": emergencies["n"],
    }


def build_factual_summary(week_start: str, week_end: str, facts: dict) -> str:
    lines = [f"Week {week_start} to {week_end} (backend-generated factual summary)."]
    messages = facts["messages"]
    lines.append(
        f"Messages: {messages.get('user', 0)} from the user, "
        f"{messages.get('assistant', 0)} from Gaash."
    )
    if facts["measurements"]:
        per_scale: Dict[str, List[str]] = {}
        for m in facts["measurements"]:
            per_scale.setdefault(m["scale"], []).append(str(m["total"]))
        for scale, totals in per_scale.items():
            lines.append(f"{scale} completed measurement(s) this week: {', '.join(totals)}.")
    else:
        lines.append("No screening scale reached a completed measurement this week.")
    if facts["impairments"]:
        described = ", ".join(
            f"{i['area']} ({i['severity']}, {i['mentions']}x)" for i in facts["impairments"]
        )
        lines.append(f"Functional impairment reported by the user: {described}.")
    sleep = facts["sleep"]
    if sleep["reports"]:
        lines.append(
            f"Sleep explicitly reported {sleep['reports']} time(s): "
            f"{sleep['min_hours']}-{sleep['max_hours']} hours (average {sleep['avg_hours']})."
        )
    if facts["checkins"]["count"]:
        lines.append(
            f"Daily check-ins with a mood score: {facts['checkins']['count']} "
            f"(average mood {facts['checkins']['avg_mood']}/5)."
        )
    latest = facts["latest_totals"]
    if any(v is not None for v in latest.values()):
        parts = [f"{k} latest total {v}" for k, v in latest.items() if v is not None]
        lines.append("Latest screening measurements (not a diagnosis): " + "; ".join(parts) + ".")
    averages = facts.get("weekly_averages") or {}
    stated = [
        f"{scale} weekly average {value}"
        + (
            f" ({facts.get('weekly_interpretation', {}).get(scale)} range)"
            if facts.get("weekly_interpretation", {}).get(scale)
            else ""
        )
        for scale, value in averages.items()
        if value is not None
    ]
    if stated:
        lines.append(
            "Weekly screening averages (screening information, not a diagnosis): "
            + "; ".join(stated) + "."
        )
    if facts["emergency_turns"]:
        lines.append(f"Safety signal raised on {facts['emergency_turns']} turn(s) this week.")
    lines.append("Screening totals are not diagnoses.")
    return " ".join(lines)


async def run_nlp_analysis(context_messages: List[dict]) -> NLPAnalysis:
    # Keep private context distinct from the current user message. Previous
    # assistant wording is reference-only, not a style template to imitate.
    prompt_parts = [SYSTEM_PROMPT]

    last_message_index = len(context_messages) - 1
    for index, message in enumerate(context_messages):
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            prompt_parts.append(f"Private backend context (not user-authored):\n{content}")
        elif role == "assistant":
            prompt_parts.append(
                "Previous assistant message (reference only; do not copy its wording or cadence):\n"
                f"{content}"
            )
        else:
            label = (
                "Current user message; respond to this"
                if index == last_message_index
                else "Previous user message (context only; do not answer it again)"
            )
            prompt_parts.append(f"{label}:\n{content}")

    prompt = "\n\n".join(prompt_parts)

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                get_gemini_client().models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=NLPAnalysis,
                ),
            ),
            timeout=GEMINI_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        logger.warning("Gemini NLP request timed out")
        raise LLMServiceError("Gemini service request timed out.") from exc
    except Exception as exc:
        logger.warning("Gemini NLP request failed: %s", type(exc).__name__)
        raise LLMServiceError("Gemini service request failed.") from exc

    if not response.text:
        raise LLMServiceError("Gemini returned an empty response.")

    try:
        return NLPAnalysis.model_validate_json(response.text)
    except Exception as exc:
        logger.warning("Gemini structured output parsing failed: %s", type(exc).__name__)
        raise LLMServiceError("Gemini structured output could not be parsed.") from exc


def _store_weekly_summary_sync(
    user_id, week_start, week_end, summary_text=None,
    averages=None, interpretation=None,
) -> int:
    averages = averages or {}
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM weekly_summaries WHERE user_id=%s AND week_start=%s AND week_end=%s",
            (user_id, week_start, week_end),
        )
        cur = conn.execute(
            "INSERT INTO weekly_summaries (user_id, week_start, week_end, summary_text, phq9_avg, gad7_avg, pss10_avg, interpretation) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (user_id, week_start, week_end, summary_text,
             averages.get("PHQ-9"), averages.get("GAD-7"), averages.get("PSS-10"),
             json.dumps(interpretation or {})),
        )
        summary_id = cur.fetchone()["id"]
        conn.commit()
        return int(summary_id)


def _resolve_week(
    week_start=None,
    week_end=None,
) -> tuple[str, str]:

    if week_start and week_end:
        start = date.fromisoformat(week_start)
        end = date.fromisoformat(week_end)

    elif week_start:
        start = date.fromisoformat(week_start)
        end = start + timedelta(days=6)

    elif week_end:
        end = date.fromisoformat(week_end)
        start = end - timedelta(days=6)

    else:
        today = datetime.now(
            REPORTING_TIMEZONE
        ).date()

        # Monday → Sunday reporting week
        start = today - timedelta(
            days=today.weekday()
        )

        end = start + timedelta(days=6)

    if end < start:
        raise ValueError(
            "week_end is before week_start"
        )

    if (end - start).days != 6:
        raise ValueError(
            "Reporting period must be exactly 7 days."
        )

    return (
        start.isoformat(),
        end.isoformat(),
    )


def _utc_day_bounds(
    start_date: str,
    end_date: str,
) -> tuple[datetime, datetime]:

    start_local = datetime.combine(
        date.fromisoformat(start_date),
        datetime.min.time(),
        tzinfo=REPORTING_TIMEZONE,
    )

    end_local = datetime.combine(
        date.fromisoformat(end_date)
        + timedelta(days=1),
        datetime.min.time(),
        tzinfo=REPORTING_TIMEZONE,
    )

    return (
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )


def _weekly_averages_sync(user_id: int, week_start: str, week_end: str) -> dict:
    """Average of COMPLETED measurements in the week — independent of chat turns."""
    start, end = _utc_day_bounds(week_start, week_end)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT "
            "AVG(CASE WHEN assessment_type='PHQ-9' THEN total END) AS phq9, "
            "AVG(CASE WHEN assessment_type='GAD-7' THEN total END) AS gad7, "
            "AVG(CASE WHEN assessment_type='PSS-10' THEN total END) AS pss10 "
            "FROM screening_measurements WHERE user_id=%s AND completed_at >= %s AND completed_at < %s",
            (user_id, start, end),
        ).fetchone()
    return {
        "PHQ-9": round(row["phq9"], 1) if row["phq9"] is not None else None,
        "GAD-7": round(row["gad7"], 1) if row["gad7"] is not None else None,
        "PSS-10": round(row["pss10"], 1) if row["pss10"] is not None else None,
    }


_PHQ9_LABELS = ["minimal", "mild", "moderate", "moderately severe", "severe"]
_GAD7_LABELS = ["minimal", "mild", "moderate", "severe"]
_PSS10_LABELS = ["lower perceived stress", "moderate perceived stress", "higher perceived stress"]
_TREND_THRESHOLDS = {"PHQ-9": 3.0, "GAD-7": 3.0, "PSS-10": 5.0}


def interpret_weekly_average(scale: str, average: Optional[float]) -> Optional[str]:
    if average is None:
        return None
    rounded = int(round(average))
    if scale == "PHQ-9":
        return _PHQ9_LABELS[_phq9_band(rounded)]
    if scale == "GAD-7":
        return _GAD7_LABELS[_gad7_band(rounded)]
    if scale == "PSS-10":
        return _PSS10_LABELS[_pss10_band(rounded)]
    return None

def interpret_assessment_total(
    scale: str,
    total: Optional[int],
) -> Optional[str]:

    if total is None:
        return None

    if scale == "PHQ-9":
        return _PHQ9_LABELS[
            _phq9_band(total)
        ]

    if scale == "GAD-7":
        return _GAD7_LABELS[
            _gad7_band(total)
        ]

    if scale == "PSS-10":
        return _PSS10_LABELS[
            _pss10_band(total)
        ]

    return None

def build_weekly_aggregate(averages: dict) -> WeeklyAggregate:
    interpretation = {}
    for scale, average in averages.items():
        label = interpret_weekly_average(scale, average)
        if label is not None:
            interpretation[scale] = label
    return WeeklyAggregate(
        phq9_avg=averages.get("PHQ-9"),
        gad7_avg=averages.get("GAD-7"),
        pss10_avg=averages.get("PSS-10"),
        interpretation=interpretation,
    )
def _recent_weekly_summaries_sync(user_id: int, weeks: int = 4) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT week_start, week_end, summary_text,
                   phq9_avg, gad7_avg, pss10_avg, interpretation
            FROM weekly_summaries
            WHERE user_id=%s
            ORDER BY week_start DESC
            LIMIT %s
            """,
            (user_id, weeks),
        ).fetchall()


async def get_recent_weekly_summaries(
    user_id: int, weeks: int = 4
) -> List[DatabaseRow]:
    return await run_db(_recent_weekly_summaries_sync, user_id, weeks)

def _weekly_history_sync(user_id: int, weeks: int) -> List[DatabaseRow]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT week_start, week_end, phq9_avg, gad7_avg, pss10_avg FROM weekly_summaries "
            "WHERE user_id=%s ORDER BY week_start DESC LIMIT %s",
            (user_id, weeks),
        ).fetchall()
    return list(reversed(rows))


def _summary_text_sync(user_id: int, week_start: str, week_end: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT summary_text FROM weekly_summaries WHERE user_id=%s AND week_start=%s AND week_end=%s "
            "ORDER BY id DESC LIMIT 1",
            (user_id, week_start, week_end),
        ).fetchone()
    return row["summary_text"] if row else ""


async def compute_four_week_trends(user_id: int, weeks: int = 4) -> Dict[str, Optional[str]]:
    history = await run_db(_weekly_history_sync, user_id, weeks)
    trends: Dict[str, Optional[str]] = {"PHQ-9": None, "GAD-7": None, "PSS-10": None}
    columns = {"PHQ-9": "phq9_avg", "GAD-7": "gad7_avg", "PSS-10": "pss10_avg"}
    for scale, column in columns.items():
        series = [row[column] for row in history if row[column] is not None]
        if len(series) < 2:
            continue
        delta = series[-1] - series[0]
        threshold = _TREND_THRESHOLDS[scale]
        trends[scale] = "worsening" if delta >= threshold else ("improving" if delta <= -threshold else "stable")
    return trends


async def get_latest_weekly_aggregate(user_id: int) -> WeeklyAggregate:
    history = await run_db(_weekly_history_sync, user_id, 1)
    if not history:
        return WeeklyAggregate()
    row = history[-1]
    return build_weekly_aggregate({
        "PHQ-9": row["phq9_avg"], "GAD-7": row["gad7_avg"], "PSS-10": row["pss10_avg"],
    })


def _analytics_snapshot_sync(user_id: int, days: int) -> dict:
    """Return data-derived trends only; no values are backfilled or inferred."""
    start_date = (datetime.now(timezone.utc).date() - timedelta(days=days - 1)).isoformat()
    with get_conn() as conn:
        measurements = conn.execute(
            "SELECT assessment_type, total, completed_at FROM screening_measurements "
            "WHERE user_id=%s AND completed_at >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day') "
            "ORDER BY completed_at ASC, id ASC",
            (user_id, days),
        ).fetchall()
        conversations = conn.execute(
            "SELECT DATE_TRUNC('week', timestamp)::date AS week_start, "
            "COUNT(*) FILTER (WHERE role='user') AS user_messages, "
            "COUNT(*) FILTER (WHERE role='assistant') AS assistant_messages "
            "FROM conversation_messages WHERE user_id=%s "
            "AND timestamp >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day') "
            "GROUP BY DATE_TRUNC('week', timestamp)::date ORDER BY week_start ASC",
            (user_id, days),
        ).fetchall()
        emotions = conn.execute(
            "SELECT primary_emotion, COUNT(*) AS count FROM emotion_records "
            "WHERE user_id=%s AND timestamp >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day') "
            "GROUP BY primary_emotion ORDER BY count DESC, primary_emotion ASC",
            (user_id, days),
        ).fetchall()
        check_ins = conn.execute(
            "SELECT checkin_date, mood_score, stress_score, sleep_hours "
            "FROM check_ins WHERE user_id=%s AND checkin_date >= %s "
            "ORDER BY checkin_date ASC, id ASC",
            (user_id, start_date),
        ).fetchall()
    return {
        "assessment_history": [dict(row) for row in measurements],
        "conversation_activity": [dict(row) for row in conversations],
        "emotion_distribution": [dict(row) for row in emotions],
        "check_ins": [dict(row) for row in check_ins],
    }


async def get_analytics_snapshot(user_id: int, days: int) -> dict:
    return await run_db(_analytics_snapshot_sync, user_id, days)


def _recent_emotions_sync(user_id: int, limit: int = 6) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT primary_emotion, confidence, severity, source, timestamp "
            "FROM emotion_records WHERE user_id=%s ORDER BY timestamp DESC, id DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()


async def get_recent_emotions(user_id: int, limit: int = 6) -> List[DatabaseRow]:
    return await run_db(_recent_emotions_sync, user_id, limit)


# trajectory is now measured by consecutive completed measurement sessions
def _previous_measurement_totals_delta_sync(user_id: int) -> Dict[str, Optional[int]]:
    """Second-most-recent completed measurement per scale (for trajectory)."""
    last_before: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None, "PSS-10": None}
    with get_conn() as conn:
        for scale in last_before:
            rows = conn.execute(
                "SELECT total FROM screening_measurements WHERE user_id=%s AND assessment_type=%s "
                "ORDER BY completed_at DESC, id DESC LIMIT 2",
                (user_id, scale),
            ).fetchall()
            if len(rows) >= 2:
                last_before[scale] = rows[1]["total"]
    return last_before


async def compute_trajectory(user_id: int, totals: Dict[str, Optional[int]]) -> Optional[str]:
    """Compare latest totals with the previous measurement, per scale."""
    previous = await run_db(_previous_measurement_totals_delta_sync, user_id)
    delta = 0
    comparable = 0
    for scale in ("PHQ-9", "GAD-7", "PSS-10"):
        cur_v, prev_v = totals.get(scale), previous.get(scale)
        if cur_v is None or prev_v is None:
            continue
        comparable += 1
        delta += int(cur_v) - int(prev_v)
    if comparable == 0:
        return None
    if delta >= 3:
        return "worsening"
    if delta <= -3:
        return "improving"
    return "stable"


def _save_risk_sync(user_id: int, details: dict, totals: Dict[str, Optional[int]]) -> None:
    """A risk record is written ONLY on measurement completion or emergency."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO risk_assessments (user_id, risk_category, phq9_total, gad7_total, "
            "pss10_total, trajectory, emergency_flag, details) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (user_id, details["risk_category"], totals.get("PHQ-9"), totals.get("GAD-7"),
             totals.get("PSS-10"), details.get("trajectory"), details["emergency"],
             json.dumps(details)),
        )
        conn.commit()


async def save_risk_assessment(user_id, details, totals) -> None:
    await run_db(_save_risk_assessment_sync, user_id, details, totals)


def _latest_risk_sync(user_id: int) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT risk_category, emergency_flag, timestamp FROM risk_assessments "
            "WHERE user_id=%s ORDER BY timestamp DESC, id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


async def get_latest_risk(user_id: int) -> Optional[DatabaseRow]:
    return await run_db(_latest_risk_sync, user_id)


def _previous_totals_sync(user_id: int) -> Optional[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT phq9_total, gad7_total, pss10_total FROM risk_assessments "
            "WHERE user_id=%s ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def _save_risk_assessment_sync(user_id, details, totals):
    _save_risk_sync(user_id, details, totals)


def _create_escalation_sync(user_id, summary, trigger_message_id) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO escalation_records (user_id, trigger_message_id, counselor_summary, status) VALUES (%s,%s,%s,'open') RETURNING id",
            (user_id, trigger_message_id, summary),
        )
        escalation_id = cur.fetchone()["id"]
        conn.commit()
        return int(escalation_id)


async def create_escalation(user_id, summary, counselor_message_id=None) -> int:
    return await run_db(_create_escalation_sync, user_id, summary, counselor_message_id)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

RECOMMENDATION_DISCLAIMER = "Supportive suggestion only - not medical advice, diagnosis, or treatment."


def _store_recommendation_sync(user_id, category, text) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO recommendation_records (user_id, category, text) VALUES (%s,%s,%s) RETURNING id",
            (user_id, category, text),
        )
        recommendation_id = cur.fetchone()["id"]
        conn.commit()
        return int(recommendation_id)


async def store_recommendation(user_id, category, text) -> int:
    return await run_db(_store_recommendation_sync, user_id, category, text)


def _list_recommendations_sync(
    user_id: int,
    limit: int,
    before_id: Optional[int] = None,
) -> List[DatabaseRow]:
    with get_conn() as conn:
        if before_id is not None:
            return conn.execute(
                """
                SELECT id, category, text, timestamp
                FROM recommendation_records
                WHERE user_id = %s
                  AND id < %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (user_id, before_id, limit),
            ).fetchall()

        return conn.execute(
            """
            SELECT id, category, text, timestamp
            FROM recommendation_records
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()


async def list_recommendations(
    user_id: int,
    limit: int = 50,
    before_id: Optional[int] = None,
) -> List[DatabaseRow]:
    return await run_db(
        _list_recommendations_sync,
        user_id,
        limit,
        before_id,
    )


build_recommendations = (
    # rule-based supportive suggestions, unchanged in spirit
    lambda totals=None, risk_category="LOW_RISK", sleep_hours=None, impairments=None, emergency=False: []
)


def _recommendations_in_window_sync(user_id: int, start: str, end: str) -> List[DatabaseRow]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, category, text, timestamp FROM recommendation_records "
            "WHERE user_id=%s AND timestamp BETWEEN %s AND %s ORDER BY id DESC",
            (user_id, f"{start} 00:00:00", f"{end} 23:59:59"),
        ).fetchall()


# ---------------------------------------------------------------------------
# Crisis pathway
# ---------------------------------------------------------------------------

def _load_crisis_contacts() -> List[CrisisContact]:
    if not CRISIS_CONTACTS_RAW.strip():
        return []
    try:
        entries = json.loads(CRISIS_CONTACTS_RAW)
        return [CrisisContact(**e) for e in entries]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("CRISIS_CONTACTS could not be parsed: %s", exc)
        return []


CRISIS_CONTACTS: List[CrisisContact] = _load_crisis_contacts()


def build_crisis_pathway() -> CrisisPathway:
    parts = [
        "If you are in immediate danger or thinking about harming yourself, "
        f"please reach out for human support right now through {CRISIS_PATHWAY_LABEL}"
    ]
    parts.append(f" ({CRISIS_PATHWAY_URL})." if CRISIS_PATHWAY_URL else ".")
    for contact in CRISIS_CONTACTS:
        hours = f", {contact.hours}" if contact.hours else ""
        region = f", {contact.region}" if contact.region else ""
        parts.append(f" {contact.name}: {contact.contact}{hours}{region}.")
    parts.append(" If you can, also tell someone you trust what you are going through.")
    return CrisisPathway(
        label=CRISIS_PATHWAY_LABEL, url=CRISIS_PATHWAY_URL or None,
        message="".join(parts), contacts=CRISIS_CONTACTS,
    )


def append_crisis_pathway(response_to_user, pathway) -> str:
    if pathway.message in response_to_user:
        return response_to_user
    if CRISIS_PATHWAY_URL and CRISIS_PATHWAY_URL in response_to_user:
        return response_to_user
    return f"{response_to_user.rstrip()}\n\n{pathway.message}"


# ---------------------------------------------------------------------------
# Visual emotion analysis (DeepFace) — contextual only, never risk input
# ---------------------------------------------------------------------------

_MAX_FRAME_BYTES = 8 * 1024 * 1024


class DeepFrameRuntimeError(Exception):
    pass


def _decode_base64_image(image_base64: str) -> bytes:
    payload = image_base64.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64 data.") from exc
    if not raw:
        raise ValueError("image_base64 decoded to an empty image.")
    if len(raw) > _MAX_FRAME_BYTES:
        raise ValueError("Image is too large (limit 8 MB).")
    return raw


def _analyze_frame_sync(image_bytes: bytes) -> dict:
    import numpy as np
    from deepface import DeepFace
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            frame = np.array(image.convert("RGB"))[:, :, ::-1]
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The supplied data is not a readable image.") from exc
    try:
        result = DeepFace.analyze(
            img_path=frame, actions=["emotion"],
            detector_backend=DEEPFACE_DETECTOR_BACKEND,
            enforce_detection=False, silent=True,
        )
    except Exception as exc:
        logger.warning("DeepFace analysis failed: %s", type(exc).__name__)
        raise DeepFrameRuntimeError(type(exc).__name__) from exc
    if isinstance(result, list):
        if not result:
            raise ValueError("No face could be analysed in the supplied frame.")
        result = result[0]
    scores = {k: round(float(v), 2) for k, v in result.get("emotion", {}).items()}
    return {"dominant_emotion": result.get("dominant_emotion"), "emotion_scores": scores}


async def analyze_frame(image_base64: str) -> AnalyzeFrameResponse:
    if not DEEPFACE_ENABLED:
        return AnalyzeFrameResponse(
            dominant_emotion=None,
            emotion_scores={},
            ok=False,
            error="Visual emotion analysis is disabled.",
        )
    try:
        image_bytes = _decode_base64_image(image_base64)
    except ValueError as exc:
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error=str(exc))
    try:
        async with _DEEPFACE_SEMAPHORE:
            result = await asyncio.wait_for(
                asyncio.to_thread(_analyze_frame_sync, image_bytes),
                timeout=DEEPFACE_TIMEOUT_SECONDS,
            )
    except TimeoutError:
        logger.warning("DeepFace analysis timed out")
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error="Visual emotion analysis timed out.")
    except ImportError as exc:
        logger.warning("DeepFace unavailable: %s", exc)
        return AnalyzeFrameResponse(
            dominant_emotion=None, emotion_scores={}, ok=False,
            error="Visual emotion analysis is temporarily unavailable.",
        )
    except ValueError as exc:
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error=str(exc))
    except DeepFrameRuntimeError:
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error="Visual emotion analysis is temporarily unavailable.")
    except Exception:
        logger.warning("Unexpected visual emotion analysis failure")
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error="Visual emotion analysis is temporarily unavailable.")
    return AnalyzeFrameResponse(dominant_emotion=result["dominant_emotion"], emotion_scores=result["emotion_scores"], ok=True, error=None)

def _pending_score_items_sync(user_id: int, include_paused: bool = True) -> List[dict]:
    with get_conn() as conn:
        statuses = "('active', 'paused')" if include_paused else "('active')"
        sessions = conn.execute(
            f"""
            SELECT session_id, scale, current_item, status
            FROM screening_sessions
            WHERE user_id=%s
              AND status IN {statuses}
            ORDER BY started_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()

        pending = []

        for session in sessions:
            row = conn.execute(
                """
                SELECT item_id, raw_score, evidence, answered_at
                FROM screening_session_items
                WHERE session_id=%s AND item_id=%s
                """,
                (session["session_id"], session["current_item"]),
            ).fetchone()

            # No record or no score means this item is still pending.
            if row is None or row["raw_score"] is None:
                pending.append({
                    "session_id": session["session_id"],
                    "scale": session["scale"],
                    "item_id": session["current_item"],
                    "status": session["status"],
                })

        return pending


async def get_pending_score_items(user_id: int, include_paused: bool = True) -> List[dict]:
    return await run_db(_pending_score_items_sync, user_id, include_paused)
# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------

async def build_llm_context(
    user_id,
    current_message,
    preferred_language,
    sleep_hours,
    deepface_emotion,
    voice_emotion=None,
    active_question=None,
    conversation_id=None,
) -> List[dict]:

    # Preserve the original text for the conversation and model prompt.  The
    # normalized form is used only to improve bounded memory retrieval across
    # Hinglish, Romanized Indic text, harmless slang, and code-switching.
    retrieval_query = normalize_for_understanding(current_message).normalized_for_context

    (
        recent,
        summaries,
        snapshot,
        pending,
        trends,
        profile,
        memories,
        conversation_summary,
        other_conversation_summaries,
        dynamic_distress_state,
    ) = await asyncio.gather(
        get_recent_messages(
            user_id,
            conversation_id=conversation_id,
        ),
        get_recent_weekly_summaries(
            user_id
        ),
        get_latest_assessment_snapshot(
            user_id
        ),
        get_pending_score_items(
            user_id,
            include_paused=False,
        ),
        compute_four_week_trends(
            user_id
        ),
        get_llm_profile(
            user_id
        ),
        get_user_memories(
            user_id,
            limit=30,
        ),
        get_conversation_summary(
            user_id,
            conversation_id,
        ),
        get_other_conversation_summaries(
            user_id,
            conversation_id,
        ),
        get_latest_dynamic_distress_state(user_id),
    )

    relevant_memories = select_relevant_memories(
        memories,
        retrieval_query,
        limit=6,
    )
    
    relevant_conversation_summaries = (
        select_relevant_conversation_summaries(
            other_conversation_summaries,
            retrieval_query,
            limit=CROSS_CONVERSATION_SUMMARY_LIMIT,
        )
    )

    context_lines = [
        (
            "[PRIVATE BACKEND CONTEXT — not user-authored. "
            "Use it only when it changes the current reply. "
            "Current-conversation context has priority over older conversation "
            "summaries and long-term memory. Current user statements override "
            "all historical context. Do not recap, quote, or expose this context.]"
        )
    ]

    profile_lines = _trim_lines_to_budget(
        format_profile_for_llm(
            profile
        ),
        PROFILE_CONTEXT_MAX_CHARS,
    )

    if profile_lines:
        context_lines.append(
            "user_profile:"
        )

        context_lines.extend(
            f"- {line}"
            for line in profile_lines
        )

    memory_lines = _trim_lines_to_budget(
        format_memories_for_llm(
            relevant_memories
        ),
        MEMORY_CONTEXT_MAX_CHARS,
    )

    if memory_lines:
        context_lines.append(
            "relevant_long_term_memory:"
        )

        context_lines.extend(
            memory_lines
        )
        
    if conversation_summary:
        summary_text = _trim_text(
            conversation_summary.get(
                "summary_text"
            ),
            CURRENT_SUMMARY_MAX_CHARS,
        )

        if summary_text:
            context_lines.append(
                "older_current_conversation_summary:"
            )

            context_lines.append(
                summary_text
            )
            
    if relevant_conversation_summaries:
        context_lines.append(
            "relevant_previous_conversations:"
        )

        for old_summary in relevant_conversation_summaries:
            summary_text = str(
                old_summary.get("summary_text") or ""
            ).strip()

            if summary_text:
                context_lines.append(
                    f"- {summary_text}"
                )

    if preferred_language:
        context_lines.append(
            f"preferred_language: {preferred_language}"
        )

    if active_question is not None:
        context_lines.append(
            f"current_follow_up_target: scale={active_question['scale']} "
            f"item={active_question['item_id']} -> previous question: "
            f"\"{active_question['question_text']}\". "
            "Interpret the user's reply as the answer when it provides "
            "frequency; otherwise continue naturally."
        )

    if summaries:
        context_lines.append(
            "weekly_summaries (oldest to newest):"
        )

        for summary in summaries:
            context_lines.append(
                f"- {summary['week_start']} to "
                f"{summary['week_end']}: "
                f"{summary['summary_text']}"
            )

    if any(snapshot[key] for key in snapshot):
        context_lines.append(
            "latest_structured_assessment_snapshot: "
            f"{snapshot}"
        )

    if any(trends.values()):
        filtered_trends = {
            key: value
            for key, value in trends.items()
            if value
        }

        context_lines.append(
            "four_week_screening_trends "
            "(backend analysis, not a diagnosis): "
            f"{filtered_trends}"
        )

    if dynamic_distress_state and dynamic_distress_state.state != "unknown":
        context_lines.append(
            "longitudinal_monitoring_context "
            "(decision support only, not a diagnosis; do not quote this label or score to the user): "
            f"state={dynamic_distress_state.state}, trajectory={dynamic_distress_state.trajectory}."
        )

    if pending:
        context_lines.append(
            "items_with_evidence_but_no_frequency_yet "
            "(ask naturally about the frequency of AT MOST one item "
            "when it fits; never run the full questionnaire): "
            f"{pending}"
        )

    if sleep_hours is not None:
        context_lines.append(
            f"sleep_hours_reported_this_turn: {sleep_hours}"
        )

    if deepface_emotion:
        context_lines.append(
            "optional_visual_emotion_signal "
            "(context only, NOT a symptom or diagnosis): "
            + deepface_emotion
        )
        
    if voice_emotion and voice_emotion.primary:
        voice_parts = [
            f"primary={voice_emotion.primary}"
        ]

        if voice_emotion.confidence is not None:
            voice_parts.append(
                f"confidence={voice_emotion.confidence:.2f}"
            ) 

        if voice_emotion.intensity:
            voice_parts.append(
            f"intensity={voice_emotion.intensity}"
            )

        context_lines.append(
            "optional_voice_emotion_signal "
            "(contextual vocal cue only; NOT a symptom, "
            "diagnosis, screening score, or independent "
            "risk evidence): "
            + ", ".join(voice_parts)
        )
        
    private_context = "\n".join(
        context_lines
    )

    if len(private_context) > PRIVATE_CONTEXT_MAX_CHARS:
        private_context = (
            private_context[
                :PRIVATE_CONTEXT_MAX_CHARS
            ].rstrip()
        )

    messages: List[dict] = [
        {
            "role": "system",
            "content": private_context,
        }
    ]

    for row in recent:
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": current_message,
        }
    )

    return messages


# ---------------------------------------------------------------------------
# LLM service
# ---------------------------------------------------------------------------

class LLMServiceError(Exception):
    pass


_client = None


def get_gemini_client():
    global _client

    if _client is None:
        if not GEMINI_API_KEY:
            raise LLMServiceError(
                "GEMINI_API_KEY is not configured."
            )

        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client

async def extract_memory_candidates(
    user_message: str,
) -> MemoryExtraction:

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                get_gemini_client().models.generate_content,
                model=GEMINI_MODEL,
                contents=(
                    MEMORY_EXTRACTION_PROMPT
                    + "\n\n"
                    + "CURRENT USER MESSAGE:\n"
                    + user_message
                ),
                config=types.GenerateContentConfig(
                    response_mime_type=
                        "application/json",
                    response_schema=
                        MemoryExtraction,
                ),
            ),
            timeout=min(
                GEMINI_TIMEOUT_SECONDS,
                10.0,
            ),
        )

    except TimeoutError:
        logger.info(
            "Memory extraction timed out; "
            "continuing without memory update."
        )

        return MemoryExtraction()

    except Exception as exc:
        logger.warning(
            "Memory extraction skipped: %s",
            type(exc).__name__,
        )

        return MemoryExtraction()

    if not response.text:
        return MemoryExtraction()

    try:
        return (
            MemoryExtraction
            .model_validate_json(
                response.text
            )
        )

    except Exception as exc:
        logger.warning(
            "Memory extraction parsing "
            "failed: %s",
            type(exc).__name__,
        )

        return MemoryExtraction()
    
async def update_conversation_summary(
    user_id: int,
    conversation_id: str,
) -> None:

    try:
        source = await run_db(
            _conversation_summary_source_sync,
            user_id,
            conversation_id,
            MAX_RECENT_MESSAGES,
        )

        messages = source["messages"]

        if (
            len(messages)
            < CONVERSATION_SUMMARY_MIN_NEW_MESSAGES
        ):
            return

        source_parts: List[str] = []

        previous_summary = (
            source.get("previous_summary")
            or ""
        ).strip()

        if previous_summary:
            source_parts.append(
                "EXISTING SUMMARY:\n"
                + previous_summary
            )

        source_parts.append(
            "NEW OLDER MESSAGES:"
        )

        used_chars = 0
        included_messages: List[dict] = []

        for row in messages:
            content = str(
                row.get("content") or ""
            ).strip()

            if not content:
                continue

            formatted = (
                f"{row['role'].upper()}: "
                f"{content}"
            )

            if (
                used_chars
                + len(formatted)
                > CONVERSATION_SUMMARY_MAX_SOURCE_CHARS
            ):
                break

            source_parts.append(
                formatted
            )

            included_messages.append(
                row
            )

            used_chars += len(
                formatted
            )

        if not included_messages:
            return

        prompt = (
            CONVERSATION_SUMMARY_PROMPT
            + "\n\n"
            + "\n".join(source_parts)
        )

        response = await asyncio.wait_for(
            asyncio.to_thread(
                get_gemini_client()
                .models
                .generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
            ),
            timeout=min(
                GEMINI_TIMEOUT_SECONDS,
                12.0,
            ),
        )

        summary_text = (
            response.text or ""
        ).strip()

        if not summary_text:
            return

        last_message_id = int(
            included_messages[-1]["id"]
        )

        await run_db(
            _save_conversation_summary_sync,
            user_id,
            conversation_id,
            summary_text,
            last_message_id,
        )

    except TimeoutError:
        logger.info(
            "Conversation summary update timed out "
            "for conversation %s.",
            conversation_id,
        )

    except Exception as exc:
        logger.warning(
            "Conversation summary update skipped "
            "for %s: %s",
            conversation_id,
            type(exc).__name__,
        )
    
async def remember_from_message(
    user_id: int,
    conversation_id: str,
    user_message: str,
) -> None:

    extraction = (
        await extract_memory_candidates(
            user_message
        )
    )

    for memory in extraction.memories[:5]:

        if memory.confidence < 0.75:
            continue

        await upsert_user_memory(
            user_id=user_id,
            memory=memory,
            conversation_id=
                conversation_id,
        )    

# ---------------------------------------------------------------------------
# Voice transcription (raw audio never retained)

def _normalise_media_type(value: Optional[str]) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _sniff_audio_media_type(audio_bytes: bytes) -> Optional[str]:
    """Identify a constrained set of browser audio containers from their bytes."""
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    if audio_bytes.startswith(b"fLaC"):
        return "audio/flac"
    if audio_bytes.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "audio/mpeg"
    if audio_bytes[:2] in {b"\xff\xf1", b"\xff\xf9"}:
        return "audio/aac"
    if audio_bytes[4:8] == b"ftyp":
        return "audio/mp4"
    return None


async def read_validated_audio(audio: UploadFile) -> tuple[bytes, str, Any]:
    filename = (audio.filename or "").strip()
    suffix = Path(filename).suffix.lower()
    declared_type = _normalise_media_type(audio.content_type)
    if suffix not in SUPPORTED_AUDIO_SUFFIXES or declared_type not in SUPPORTED_AUDIO_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")

    audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The audio recording is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="The audio recording is too large.")

    detected_type = _sniff_audio_media_type(audio_bytes)
    if detected_type is None:
        raise HTTPException(status_code=415, detail="The uploaded file is not a supported audio recording.")
    if detected_type != declared_type and not ({detected_type, declared_type} <= {"audio/mp4", "audio/m4a", "audio/x-m4a"}):
        raise HTTPException(status_code=415, detail="The audio format does not match the uploaded recording.")

    inspection = inspect_audio(detected_type, audio_bytes, VOICE_FFPROBE_PATH)
    if inspection.duration_seconds is not None and inspection.duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(status_code=413, detail="The audio recording is too long.")
    if VOICE_REQUIRE_VERIFIED_DURATION and not inspection.duration_verified:
        raise HTTPException(
            status_code=422,
            detail="Audio duration validation is unavailable for this recording format.",
        )
    if VOICE_REQUIRE_VERIFIED_DURATION and not inspection.codec_verified:
        raise HTTPException(
            status_code=422,
            detail="Audio codec validation is unavailable for this recording format.",
        )

    # Raw audio is intentionally held only for this request and never written
    # to a file or included in application logs. Deployments can require a
    # trusted duration validator for every container through environment policy.
    return audio_bytes, detected_type, inspection

def _is_uncertain_transcript(value: str) -> bool:
    normalised = value.strip().lower()
    return normalised in {"", "[unclear]", "unclear", "[inaudible]", "inaudible", "no speech", "no audible speech"}


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
) -> VoiceAudioAnalysis:

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                get_gemini_client().models.generate_content,
                model=GEMINI_MODEL,
                contents=[
                    {
                        "text": (
                            "Analyze this voice recording for GAASH, "
                            "a wellbeing conversational assistant.\n\n"

                            "1. Transcribe the spoken words accurately.\n"
                            "2. Preserve the speaker's original language "
                            "and natural code-switching, including English, "
                            "Hindi, Hinglish, Urdu, Kashmiri and Dogri.\n"
                            "3. Using ONLY audible vocal cues such as tone, "
                            "pace, energy and delivery, provide a cautious "
                            "voice-emotion context signal when sufficiently "
                            "supported.\n"
                            "4. Do not diagnose mental-health conditions.\n"
                            "5. Do not infer PHQ-9, GAD-7, PSS-10 scores, "
                            "risk level, intent, personality or hidden facts "
                            "from voice tone.\n"
                            "6. If the emotion cannot be determined "
                            "reliably, return null emotion fields.\n"
                            "7. Emotion confidence must be conservative.\n\n"

                            "The voice-emotion result is contextual metadata "
                            "only. Spoken words remain the primary evidence."
                        )
                    },
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type,
                    ),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type=
                        "application/json",
                    response_schema=
                        VoiceAudioAnalysis,
                ),
            ),
            timeout=GEMINI_TIMEOUT_SECONDS,
        )

    except TimeoutError as exc:
        logger.warning(
            "Gemini voice analysis timed out"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice processing timed out. "
                "Please try again."
            ),
        ) from exc

    except Exception as exc:
        logger.warning(
            "Gemini voice analysis failed: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice processing is temporarily "
                "unavailable."
            ),
        ) from exc

    if not response.text:
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice processing returned "
                "no result."
            ),
        )

    try:
        result = (
            VoiceAudioAnalysis
            .model_validate_json(response.text)
        )

    except Exception as exc:
        logger.warning(
            "Voice structured output parsing failed: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Voice processing returned "
                "an invalid result."
            ),
        ) from exc

    transcript = result.transcript.strip()

    if _is_uncertain_transcript(transcript):
        raise HTTPException(
            status_code=422,
            detail=(
                "Speech could not be understood "
                "clearly enough."
            ),
        )

    return result
    
@app.get("/health")
async def health():
    try:
        db_ok = await asyncio.wait_for(
            run_db(_health_db_sync),
            timeout=5.0,
        )
    except Exception as exc:
        logger.warning(
            "Health check database failure: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "service": "gaash-bot",
                "database": "unavailable",
            },
        ) from exc

    return {
        "status": "ok",
        "service": "gaash-bot",
        "database": "ok" if db_ok else "unavailable",
    }
    
# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "gaash-bot",
        "status": "running"
    }

# ---------------------------------------------------------------------------
# PROFILE
# ---------------------------------------------------------------------------

@app.get("/profile", response_model=ProfileResponse)
async def get_profile(
    user_id: int = Depends(get_current_user_id),
):
    row = await run_db(_get_profile_sync, user_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    result = dict(row)

    if result.get("notification_prefs"):
        try:
            result["notification_prefs"] = json.loads(
                result["notification_prefs"]
            )
        except Exception:
            result["notification_prefs"] = None

    return result


def _update_profile_sync(user_id: int, updates: List[str], values: List[Any]) -> None:
    """Upsert profile fields using PostgreSQL placeholders off the async event loop."""
    if not updates:
        return

    columns = ", ".join(item.split("=", 1)[0] for item in updates)
    placeholders = ", ".join("%s" for _ in values)

    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO user_profiles (user_id, {columns})
            VALUES (%s, {placeholders})
            ON CONFLICT (user_id) DO UPDATE
            SET {", ".join(updates)}, updated_at=CURRENT_TIMESTAMP
            """,
            [user_id, *values, *values],
        )
        conn.commit()


@app.put("/profile", response_model=StatusResponse)
async def update_profile(
    data: ProfileUpdateRequest,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "profile-update", limit=20, window_seconds=600)
    updates = []
    values = []

    if data.display_name is not None:
        updates.append("display_name=%s")
        values.append(data.display_name)

    if data.preferred_language is not None:
        if data.preferred_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported language."
            )

        updates.append("preferred_language=%s")
        values.append(data.preferred_language)

    if data.theme is not None:
        if data.theme not in SUPPORTED_THEMES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported theme."
            )

        updates.append("theme=%s")
        values.append(data.theme)

    if data.notification_prefs is not None:
        updates.append("notification_prefs=%s")
        values.append(json.dumps(data.notification_prefs))

    if updates:
        await run_db(_update_profile_sync, user_id, updates, values)

    return {
        "status": "updated"
    }


# ---------------------------------------------------------------------------
# PHASE 1: NATIONAL CASE CONTEXT AND CONTRACT REGISTRIES
# ---------------------------------------------------------------------------

@app.get("/languages", response_model=LanguageRegistryResponse)
async def get_language_registry(http_request: Request):
    """Public capability registry; it does not claim untranslated coverage."""

    enforce_rate_limit(http_request, "language-registry", limit=120, window_seconds=60)
    languages, state_coverage = language_registry_payload()
    return LanguageRegistryResponse(languages=languages, states_ut=state_coverage)


@app.get("/case-context", response_model=CaseContextResponse)
async def get_case_context(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    started_at = time.perf_counter()
    enforce_rate_limit(http_request, "case-context-read", limit=120, window_seconds=60)
    try:
        row = await run_db(_get_case_context_sync, user_id)
        if row is None:
            return CaseContextResponse(configured=False)
        return CaseContextResponse(configured=True, **dict(row))
    finally:
        _log_safe_timing("case_context_read", started_at)


@app.put("/case-context", response_model=CaseContextResponse)
async def update_case_context(
    data: CaseContextUpdate,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    """Store beneficiary-controlled context for the authenticated owner only."""

    started_at = time.perf_counter()
    enforce_rate_limit(http_request, "case-context-update", limit=20, window_seconds=600)
    try:
        try:
            canonical_state = require_state_ut(data.state_ut)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        normalized = data.model_copy(update={"state_ut": canonical_state})
        row = await run_db(_upsert_case_context_sync, user_id, normalized)
        return CaseContextResponse(configured=True, **dict(row))
    finally:
        _log_safe_timing("case_context_update", started_at)


@app.post("/memory/retrieval", response_model=MemoryRetrievalResponse)
async def retrieve_owned_memory(
    data: MemoryRetrievalRequest,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    """Bounded, relevance-ranked retrieval for the authenticated owner only."""

    started_at = time.perf_counter()
    enforce_rate_limit(http_request, "memory-retrieval", limit=60, window_seconds=60)
    try:
        memory_rows, summary_rows = await run_db(
            _phase1_memory_retrieval_rows_sync,
            user_id,
            data.conversation_id,
        )

        normalized_query = normalize_for_understanding(data.query).normalized_for_context
        facts, summaries = select_bounded_owned_memory(
            user_id=user_id,
            query=normalized_query,
            memory_rows=memory_rows,
            summary_rows=summary_rows,
            max_facts=data.max_facts,
            max_summaries=data.max_summaries,
            char_budget=data.char_budget,
        )
        return MemoryRetrievalResponse(
            facts=[
                MemoryFact(
                    memory_key=str(item["memory_key"]),
                    memory_value=str(item["memory_value"]),
                    memory_type=str(item["memory_type"]),
                    source_conversation_id=item.get("source_conversation_id"),
                    source_message_id=item.get("source_message_id"),
                    source_kind=str(item.get("source_kind") or "conversation"),
                    confidence=float(item.get("confidence") or 0),
                    status=str(item.get("status") or "active"),
                    observed_at=item["observed_at"],
                )
                for item in facts
            ],
            conversation_summaries=[
                {
                    "conversation_id": str(item["conversation_id"]),
                    "summary_text": str(item["summary_text"]),
                    "updated_at": item["updated_at"],
                }
                for item in summaries
            ],
            char_budget=data.char_budget,
        )
    finally:
        _log_safe_timing("memory_retrieval", started_at)

# ---------------------------------------------------------------------------
# PHASE 3: AUTHORIZED SUPPORT PORTAL
# ---------------------------------------------------------------------------

@app.get("/support/cases", response_model=CaseQueueResponse)
async def authorized_case_queue(
    http_request: Request,
    limit: int = Query(default=25, ge=1, le=OPERATIONAL_QUEUE_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    """Return only pseudonymous, explicitly authorized case summaries."""

    started_at = time.perf_counter()
    enforce_rate_limit(http_request, "support-case-queue", limit=60, window_seconds=600)
    try:
        rows = await run_db(_case_queue_rows_sync, principal, limit + 1, offset)
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        summaries = [
            OperationalCaseSummary(
                case_id=str(row["case_id"]),
                public_case_reference=str(row["public_case_reference"]),
                priority=normalize_priority(row.get("priority")),
                trajectory=str(row.get("trajectory") or "unknown"),
                reason_summary=str(row.get("reason_summary") or "Human review is pending for this case."),
                last_evaluated_at=row.get("last_evaluated_at"),
                last_reviewed_at=row.get("last_reviewed_at"),
                assigned_professional=row.get("assigned_professional"),
                pending_action=row.get("pending_action"),
                alert_id=row.get("alert_id"),
            )
            for row in page_rows
        ]
        return CaseQueueResponse(
            cases=summaries,
            page={
                "limit": limit,
                "has_more": has_more,
                "next_cursor": str(offset + limit) if has_more else None,
            },
        )
    finally:
        _log_safe_timing("support_case_queue", started_at)


@app.get("/support/alerts", response_model=CaseQueueResponse)
async def authorized_alert_queue(
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    """A compact queue of the caller's cases with an open human-review task."""

    enforce_rate_limit(http_request, "support-alert-queue", limit=60, window_seconds=600)
    rows = await run_db(_case_queue_rows_sync, principal, OPERATIONAL_QUEUE_MAX_LIMIT, None)
    alerts = [row for row in rows if row.get("pending_action") == "Review alert"]
    return CaseQueueResponse(
        cases=[
            OperationalCaseSummary(
                case_id=str(row["case_id"]), public_case_reference=str(row["public_case_reference"]),
                priority=normalize_priority(row.get("priority")), trajectory=str(row.get("trajectory") or "unknown"),
                reason_summary=str(row.get("reason_summary") or "Human review is pending for this case."),
                last_evaluated_at=row.get("last_evaluated_at"), last_reviewed_at=row.get("last_reviewed_at"),
                assigned_professional=row.get("assigned_professional"), pending_action="Review alert", alert_id=row.get("alert_id"),
            )
            for row in alerts
        ],
        page={"limit": OPERATIONAL_QUEUE_MAX_LIMIT, "has_more": False, "next_cursor": None},
    )


@app.get("/support/cases/{case_id}", response_model=CaseDetailResponse)
async def authorized_case_detail(
    case_id: str,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    """Show the minimum necessary, non-conversation case decision-support view."""

    enforce_rate_limit(http_request, "support-case-detail", limit=120, window_seconds=600)
    await require_case_access(principal, case_id, "case_summary")
    row, assessment_rows = await asyncio.gather(
        run_db(_case_detail_row_sync, case_id),
        run_db(_case_assessments_sync, case_id),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Case not available.")
    indicators = _decode_json_array(row.get("contributing_indicators"))
    await write_safe_audit(
        principal.user_id, "case_viewed", case_id=case_id, resource_type="case_summary",
        resource_id=case_id, metadata={"view": "minimum_necessary"},
    )
    return CaseDetailResponse(
        case_id=str(row["case_id"]),
        public_case_reference=str(row["public_case_reference"]),
        state_ut=str(row["state_ut"]), district=row.get("district"),
        priority=normalize_priority(row.get("priority")), trajectory=str(row.get("trajectory") or "unknown"),
        reason_summary=concise_reason_summary(indicators, str(row.get("reason_summary") or "No pending human review task.")),
        last_evaluated_at=row.get("last_evaluated_at"), last_reviewed_at=row.get("last_reviewed_at"),
        assigned_professional=row.get("assigned_professional"), pending_action=row.get("pending_action"),
        alert_id=row.get("alert_id"),
        assessment_history=[
            {"assessment_type": item["assessment_type"], "score": item["total"], "completed_at": item.get("completed_at")}
            for item in assessment_rows if item.get("total") is not None
        ],
        contributing_indicators=indicators,
        confidence=row.get("confidence"), data_period_start=row.get("data_period_start"), data_period_end=row.get("data_period_end"),
    )


@app.put("/support/cases/{case_id}/assignment", response_model=CaseAssignmentResponse)
async def assign_case(
    case_id: str,
    data: CaseAssignmentRequest,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(PlatformRole.AUTHORIZED_OFFICIAL, PlatformRole.ADMIN)),
):
    enforce_rate_limit(http_request, "support-case-assignment", limit=30, window_seconds=600)
    await require_case_access(principal, case_id, "case_manage")
    if not await run_db(_is_staff_user_sync, data.assigned_to_user_id):
        raise HTTPException(status_code=422, detail="Assigned account must be an authorized support user.")
    assignment = await run_db(_assign_case_sync, case_id, principal.user_id, data.assigned_to_user_id)
    await write_safe_audit(
        principal.user_id, "case_assigned", case_id=case_id, resource_type="case_assignment",
        resource_id=str(data.assigned_to_user_id), metadata={"action": "assigned"},
    )
    return CaseAssignmentResponse(**dict(assignment))


@app.post("/support/alerts/{alert_id}/review", response_model=AlertReviewResponse)
async def review_alert(
    alert_id: int,
    data: AlertReviewRequest,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    enforce_rate_limit(http_request, "support-alert-review", limit=60, window_seconds=600)
    case_id = await run_db(_alert_case_id_sync, alert_id)
    if not case_id:
        raise HTTPException(status_code=404, detail="Alert not available.")
    await require_case_access(principal, case_id, "case_summary")
    review = await run_db(_review_alert_sync, alert_id, case_id, principal.user_id, data.decision.value, data.note)
    await write_safe_audit(
        principal.user_id, "alert_reviewed", case_id=case_id, resource_type="monitoring_alert",
        resource_id=str(alert_id), metadata={"decision": data.decision.value},
    )
    return AlertReviewResponse(**dict(review))


@app.get("/support/cases/{case_id}/interventions", response_model=List[InterventionResponse])
async def list_case_interventions(
    case_id: str,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    enforce_rate_limit(http_request, "support-interventions-read", limit=120, window_seconds=600)
    await require_case_access(principal, case_id, "case_summary")
    return [InterventionResponse(**dict(row)) for row in await run_db(_list_interventions_sync, case_id)]


@app.post("/support/cases/{case_id}/interventions", response_model=InterventionResponse, status_code=status.HTTP_201_CREATED)
async def create_case_intervention(
    case_id: str,
    data: InterventionCreate,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    enforce_rate_limit(http_request, "support-intervention-create", limit=30, window_seconds=600)
    await require_case_access(principal, case_id, "intervention_update")
    if data.assigned_to_user_id is not None and not await run_db(_is_staff_user_sync, data.assigned_to_user_id):
        raise HTTPException(status_code=422, detail="Assigned account must be an authorized support user.")
    row = await run_db(_create_intervention_sync, case_id, principal.user_id, data)
    await write_safe_audit(
        principal.user_id, "intervention_created", case_id=case_id, resource_type="intervention",
        resource_id=str(row["id"]), metadata={"category": data.intervention_type.value},
    )
    return InterventionResponse(**dict(row))


@app.patch("/support/cases/{case_id}/interventions/{intervention_id}", response_model=InterventionResponse)
async def update_case_intervention(
    case_id: str,
    intervention_id: int,
    data: InterventionUpdate,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    enforce_rate_limit(http_request, "support-intervention-update", limit=60, window_seconds=600)
    await require_case_access(principal, case_id, "intervention_update")
    if data.assigned_to_user_id is not None and not await run_db(_is_staff_user_sync, data.assigned_to_user_id):
        raise HTTPException(status_code=422, detail="Assigned account must be an authorized support user.")
    row = await run_db(_update_intervention_sync, case_id, intervention_id, data)
    if row is None:
        raise HTTPException(status_code=404, detail="Intervention not available.")
    await write_safe_audit(
        principal.user_id, "intervention_updated", case_id=case_id, resource_type="intervention",
        resource_id=str(intervention_id), metadata={"status": data.status.value if data.status else "updated"},
    )
    return InterventionResponse(**dict(row))


@app.get("/support/cases/{case_id}/notes", response_model=List[CaseNoteResponse])
async def list_case_notes(
    case_id: str,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    enforce_rate_limit(http_request, "support-case-notes-read", limit=120, window_seconds=600)
    await require_case_access(principal, case_id, "case_summary")
    return [CaseNoteResponse(**dict(row)) for row in await run_db(_list_case_notes_sync, case_id)]


@app.post("/support/cases/{case_id}/notes", response_model=CaseNoteResponse, status_code=status.HTTP_201_CREATED)
async def create_case_note(
    case_id: str,
    data: CaseNoteCreate,
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(
        PlatformRole.COUNSELLOR,
        PlatformRole.AUTHORIZED_OFFICIAL,
        PlatformRole.ADMIN,
    )),
):
    enforce_rate_limit(http_request, "support-case-note-create", limit=30, window_seconds=600)
    await require_case_access(principal, case_id, "case_summary")
    row = await run_db(_create_case_note_sync, case_id, principal.user_id, data.note)
    await write_safe_audit(
        principal.user_id, "case_note_created", case_id=case_id, resource_type="case_note",
        resource_id=str(row["id"]), metadata={"note_recorded": True},
    )
    return CaseNoteResponse(**dict(row))


@app.get("/support/aggregates", response_model=OperationalAggregateResponse)
async def operational_aggregates(
    http_request: Request,
    level: AggregateLevel = Query(...),
    state_ut: Optional[str] = Query(default=None),
    district: Optional[str] = Query(default=None, max_length=160),
    principal: AuthenticatedPrincipal = Depends(require_roles(PlatformRole.AUTHORIZED_OFFICIAL, PlatformRole.ADMIN)),
):
    """Return count-suppressed operational data, never personal data."""

    enforce_rate_limit(http_request, "support-aggregates", limit=60, window_seconds=600)
    normalized_state = require_state_ut(state_ut) if state_ut else None
    if level == AggregateLevel.DISTRICT and (not normalized_state or not district):
        raise HTTPException(status_code=422, detail="District aggregates require State/UT and district.")
    if level == AggregateLevel.STATE_UT and (not normalized_state or district):
        raise HTTPException(status_code=422, detail="State/UT aggregates require State/UT only.")
    if level == AggregateLevel.NATIONAL and (normalized_state or district):
        raise HTTPException(status_code=422, detail="National aggregates do not accept geography filters.")
    if not await run_db(_aggregate_scope_sync, principal, level.value, normalized_state, district):
        raise HTTPException(status_code=403, detail="This account is not permitted to view this aggregate scope.")
    counts = await run_db(_operational_aggregate_counts_sync, normalized_state, district)
    await write_safe_audit(
        principal.user_id, "aggregate_viewed", resource_type="aggregate", resource_id=level.value,
        metadata={"state_ut": normalized_state or "", "district": district or ""},
    )
    return OperationalAggregateResponse(
        level=level, state_ut=normalized_state, district=district,
        minimum_cell_count=OPERATIONAL_MINIMUM_CELL_COUNT,
        metrics=[{"key": key, **suppress_small_cell(value, OPERATIONAL_MINIMUM_CELL_COUNT)} for key, value in counts.items()],
        generated_at=datetime.now(timezone.utc),
    )


@app.get("/support/integrations", response_model=List[IntegrationAdapterStatus])
async def external_case_adapter_status(
    http_request: Request,
    principal: AuthenticatedPrincipal = Depends(require_roles(PlatformRole.ADMIN)),
):
    """Expose configuration state only; this route performs no external sync."""

    enforce_rate_limit(http_request, "support-integrations", limit=30, window_seconds=600)
    await write_safe_audit(principal.user_id, "integration_status_viewed", resource_type="integration", resource_id=None)
    return [
        IntegrationAdapterStatus(**integration_status("nhaa")),
        IntegrationAdapterStatus(**integration_status("integrated_portal")),
        IntegrationAdapterStatus(**integration_status("manual_demo")),
    ]

# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "chat", limit=30, window_seconds=60)
    message_text = payload.user_message.strip()
    if not message_text:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if payload.preferred_language and payload.preferred_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language.")

    conversation_id = payload.conversation_id

    if conversation_id:
        conversation = await verify_conversation(
            user_id,
            conversation_id,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found."
            )
    else:
        conversation_id = await create_conversation(user_id)

    context = await build_llm_context(
        user_id=user_id,
        current_message=message_text,
        conversation_id=conversation_id,
        preferred_language=payload.preferred_language,
        sleep_hours=payload.sleep_hours,
        deepface_emotion=payload.deepface_emotion,
        voice_emotion=payload.voice_emotion,
    )

    # Persist the user turn before invoking the external model. A transient
    # model failure must not erase an already created conversation or message.
    user_message_id = await save_message(user_id, "user", message_text, conversation_id)

    try:
        analysis = await run_nlp_analysis(context)
    except LLMServiceError as exc:
        logger.warning("LLM analysis failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again shortly.",
        )
    background_tasks.add_task(
        remember_from_message,
        user_id,
        conversation_id,
        message_text,
    )

    # Save passive evidence separately from deliberate sessions.  A chat turn
    # must not start or alter a PHQ-9/GAD-7/PSS-10 questionnaire lifecycle.
    passive_observations: list[tuple[str, SymptomItem]] = []
    for scale, items in (
        ("PHQ-9", analysis.phq9_symptoms),
        ("GAD-7", analysis.gad7_symptoms),
        ("PSS-10", analysis.pss10_symptoms),
    ):
        passive_observations.extend(
            (scale, item)
            for item in validate_symptom_items(scale, items)
            if item.score is not None
        )
    background_tasks.add_task(
        save_passive_screening_evidence,
        user_id,
        passive_observations,
    )

    emergency = bool(analysis.emergency_flag)

    escalation_created = False

    if emergency:
        await create_escalation(
            user_id,
            analysis.response_to_user,
        )
        escalation_created = True

    reply = analysis.response_to_user

    if emergency:
        reply = append_crisis_pathway(
            reply,
            build_crisis_pathway(),
        )

    assistant_message_id = await save_message(
        user_id,
        "assistant",
        reply,
        conversation_id,
    )
    
    background_tasks.add_task(
        update_conversation_summary,
        user_id,
        conversation_id,
    )

    totals = await get_latest_finalized_totals(user_id)

    trajectory = await compute_trajectory(
        user_id,
        totals,
    )

    sleep_hours_for_risk = (
        payload.sleep_hours
        if payload.sleep_hours is not None
        else analysis.sleep_hours_reported
    )
    risk_details = get_risk_details(
        phq9=totals.get("PHQ-9"),
        gad7=totals.get("GAD-7"),
        pss10=totals.get("PSS-10"),
        sleep_hours=sleep_hours_for_risk,
        emotion=analysis.primary_emotion or payload.deepface_emotion,
        trajectory=trajectory,
        emergency=emergency,
    )


    pending = {
        "PHQ-9": [],
        "GAD-7": [],
        "PSS-10": [],
    }
    
    primary_emotion = (
        analysis.primary_emotion or ""
    ).strip() or None

    emotion_severity = (
        analysis.emotion_severity or ""
    ).strip() or None


    async def save_current_emotion():
        tasks = []

        if primary_emotion:
            tasks.append(
                save_emotion(
                    user_id,
                    conversation_id,
                    "text",
                    primary_emotion,
                    analysis.emotion_confidence,
                    emotion_severity,
                )
            )

        if (
            payload.voice_emotion
            and payload.voice_emotion.primary
        ):
            tasks.append(
                save_emotion(
                    user_id,
                    conversation_id,
                    "voice",
                    payload.voice_emotion.primary,
                    payload.voice_emotion.confidence,
                    payload.voice_emotion.intensity,
                )
            )

        if payload.deepface_emotion:
            tasks.append(
                save_emotion(
                    user_id,
                    conversation_id,
                    "visual",
                    payload.deepface_emotion.strip(),
                    None,
                    None,
                )
            )

        if tasks:
            await asyncio.gather(*tasks)
    pending_raw, _, _ = await asyncio.gather(
        get_pending_score_items(user_id),

        save_current_emotion(),

        save_analysis_observations(
            user_id,
            analysis.sleep_hours_reported,
            analysis.functional_impairments,
        ),
    )
    
    for item in pending_raw:
        scale = item.get("scale")
        item_id = item.get("item_id")

        if scale in pending and item_id is not None:
            pending[scale].append(int(item_id))
            
    primary_emotion = (analysis.primary_emotion or "").strip() or None
    emotion_severity = (analysis.emotion_severity or "").strip() or None

    if emergency:
        await save_risk_assessment(user_id, risk_details, totals)

    analytics = ChatAnalytics(
        detected_language=analysis.detected_language,
        primary_emotion=primary_emotion,
        emotion_confidence=analysis.emotion_confidence,
        emotion_severity=emotion_severity,
        phq9_symptoms=analysis.phq9_symptoms,
        gad7_symptoms=analysis.gad7_symptoms,
        pss10_symptoms=analysis.pss10_symptoms,
        sleep_hours_reported=analysis.sleep_hours_reported,
        functional_impairments=analysis.functional_impairments,
        active_scale_triggered=analysis.active_scale_triggered,
        emergency_flag=emergency,
        pending_score_items=pending,
    )

    emotion = None

    if primary_emotion:
        emotion = EmotionResponse(
            primary=primary_emotion,
            confidence=analysis.emotion_confidence,
            severity=emotion_severity,
        )
    elif payload.deepface_emotion:
        emotion = EmotionResponse(
            primary=payload.deepface_emotion.strip(),
        )

    risk = RiskResponse(
        level=risk_details["risk_category"],
        requires_escalation=emergency,
    )

    distress_state = None
    try:
        event_id = await create_wellbeing_event(
            user_id=user_id,
            source_type="text",
            raw_signals={
                "primary_emotion": primary_emotion,
                "emotion_confidence": analysis.emotion_confidence,
                "emotion_severity": emotion_severity,
                "risk_category": risk_details["risk_category"],
                "emergency": emergency,
                "assessment_trajectory": trajectory,
                "speech_detected": False,
                "acoustic_available": False,
            },
            source_reference=str(user_message_id),
            conversation_id=conversation_id,
        )
        distress_state = await compute_and_store_dynamic_distress_state(user_id, event_id)
    except Exception as exc:
        # The conversation remains available if a post-analysis monitoring
        # write is temporarily unavailable. No user content is logged.
        logger.warning("Wellbeing event processing skipped: %s", type(exc).__name__)

    return ChatResponse(
        message_id=str(assistant_message_id),
        user_message_id=str(user_message_id),
        reply=reply,
        conversation_id=conversation_id,
        emotion=emotion,
        risk=risk,
        suggested_replies=[],
        timestamp=_iso_timestamp(),
        response_to_user=reply,
        analytics=analytics,
        risk_category=risk_details["risk_category"],
        emergency_detected=emergency,
        escalation_created=escalation_created,
        distress_state=distress_state,
    )


# ---------------------------------------------------------------------------
# CONVERSATIONS
# ---------------------------------------------------------------------------

@app.post("/conversations", response_model=ConversationCreatedResponse)
async def create_new_conversation(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "conversation-create", limit=30, window_seconds=600)
    conversation_id = await create_conversation(user_id)

    return {
        "conversation_id": conversation_id
    }


@app.get("/conversations", response_model=ConversationListResponse)
async def get_conversations(
    http_request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    before: Optional[str] = Query(default=None, max_length=128),
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "conversation-list", limit=120, window_seconds=60)
    cursor = _parse_conversation_cursor(before)
    rows = await run_db(
        _own_conversations_sync,
        user_id,
        limit + 1,
        cursor,
    )
    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_cursor = None
    if has_more and page_rows:
        last_row = page_rows[-1]
        timestamp = last_row["last_activity_at"]
        next_cursor = f"{timestamp.astimezone(timezone.utc).isoformat()}|{last_row['id']}"

    return {
        "conversations": [
            {key: value for key, value in dict(row).items() if key != "id"}
            for row in page_rows
        ],
        "limit": limit,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


@app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    http_request: Request,
    limit: int = Query(default=100, ge=1, le=100),
    before_id: Optional[int] = Query(default=None, ge=1),
    user_id: int = Depends(get_current_user_id),
):
    conversation = await verify_conversation(
        user_id,
        conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    rows = await run_db(
        _conversation_messages_sync,
        user_id,
        conversation_id,
        limit + 1,
        before_id,
    )
    has_more = len(rows) > limit
    page_rows = rows[-limit:] if has_more else rows
    next_before_id = page_rows[0]["id"] if has_more and page_rows else None

    return {
        "conversation_id": conversation_id,
        "messages": [
            dict(row)
            for row in page_rows
        ],
        "limit": limit,
        "has_more": has_more,
        "next_before_id": next_before_id,
    }

@app.delete(
    "/conversations/{conversation_id}",
    response_model=StatusResponse,
)
async def delete_conversation(
    conversation_id: str,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "conversation-delete",
        limit=20,
        window_seconds=600,
    )

    deleted = await run_db(
        _delete_conversation_sync,
        user_id,
        conversation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "status": "deleted"
    }

@app.delete(
    "/conversations",
    response_model=StatusResponse,
)
async def delete_all_conversations(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "conversation-delete-all",
        limit=5,
        window_seconds=600,
    )

    deleted_count = await run_db(
        _delete_all_conversations_sync,
        user_id,
    )

    return {
        "status": "deleted"
    }

# ---------------------------------------------------------------------------
# ASSESSMENTS
# ---------------------------------------------------------------------------

@app.get("/assessments/latest", response_model=Dict[str, List[Dict[str, Any]]])
async def latest_assessment(
    user_id: int = Depends(get_current_user_id),
):
    return await get_latest_assessment_snapshot(user_id)


@app.get("/assessments/totals", response_model=Dict[str, Optional[int]])
async def assessment_totals(
    user_id: int = Depends(get_current_user_id),
):
    return await get_latest_finalized_totals(user_id)


@app.get("/assessments/pending", response_model=PendingAssessmentsResponse)
async def pending_assessments(
    user_id: int = Depends(get_current_user_id),
):
    return {
        "pending": await get_pending_score_items(user_id)
    }


@app.get("/assessments/{scale}/history", response_model=AssessmentHistoryResponse)
async def assessment_history(
    scale: str,
    limit: int = Query(default=20, ge=1, le=100),
    before_id: Optional[int] = Query(default=None, ge=1),
    user_id: int = Depends(get_current_user_id),
):
    try:
        scale = normalize_scale(scale)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment scale."
        )

    rows = await measurement_history(
        user_id,
        scale,
        limit + 1,
        before_id,
    )
    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_before_id = page_rows[-1]["id"] if has_more and page_rows else None

    return {
        "scale": scale,
        "history": [
            dict(row)
            for row in page_rows
        ],
        "limit": limit,
        "has_more": has_more,
        "next_before_id": next_before_id,
    }


@app.post("/assessments/{scale}/start", response_model=AssessmentSessionResponse)
async def start_assessment(
    scale: str,
    http_request: Request,
    conversation_id: Optional[str] = None,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "assessment-start", limit=10, window_seconds=600)
    try:
        scale = normalize_scale(scale)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment scale."
        )

    if conversation_id and await verify_conversation(user_id, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    session = await get_or_start_screening_session(
        user_id=user_id,
        conversation_id=conversation_id,
        scale=scale,
    )

    return dict(session)


# ---------------------------------------------------------------------------
# SCREENING SESSION CONTROL
# ---------------------------------------------------------------------------

@app.post("/screening/{session_id}/pause", response_model=SessionStatusResponse)
async def pause_screening(
    session_id: str,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "screening-pause",
        limit=30,
        window_seconds=600,
    )

    session = await run_db(
        _get_owned_session_sync,
        user_id,
        session_id,
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    current_status = session["status"]

    if current_status != "active":
        raise HTTPException(
            status_code=409,
            detail=f"Session is {current_status} and cannot be paused.",
        )

    success = await pause_session(
        user_id,
        session_id,
    )

    if not success:
        # Covers a rare race where session state changed between
        # checking it and updating it.
        raise HTTPException(
            status_code=409,
            detail="Session state changed and could not be paused.",
        )

    return {
        "status": "paused",
        "session_id": session_id,
    }


@app.post("/screening/{session_id}/cancel", response_model=SessionStatusResponse)
async def cancel_screening(
    session_id: str,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "screening-cancel", limit=30, window_seconds=600)
    success = await cancel_session(
        user_id,
        session_id,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return {
        "status": "cancelled",
        "session_id": session_id,
    }


@app.get("/screening/{session_id}", response_model=ScreeningDetailResponse)
async def screening_session(
    session_id: str,
    user_id: int = Depends(get_current_user_id),
):
    session = await run_db(
        _get_owned_session_sync,
        user_id,
        session_id,
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    items = await get_session_items(session_id)

    return {
        "session": dict(session),
        "items": [
            dict(item)
            for item in items
        ]
    }


# ---------------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------------

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    http_request: Request,
    days: int = Query(default=28, ge=7, le=MAX_ANALYTICS_DAYS),
    user_id: int = Depends(get_current_user_id),
):
    started_at = time.perf_counter()
    enforce_rate_limit(http_request, "analytics", limit=60, window_seconds=600)
    try:
        totals = await get_latest_finalized_totals(user_id)
        weekly = await get_latest_weekly_aggregate(user_id)
        trends = await compute_four_week_trends(user_id)
        trajectory = await compute_trajectory(
            user_id,
            totals,
        )
        snapshot, monitoring = await asyncio.gather(
            get_analytics_snapshot(user_id, days),
            get_longitudinal_monitoring_payload(user_id),
        )
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=days)

        return {
            "screening_totals": totals,
            "weekly_averages": weekly.model_dump(),
            "four_week_trends": trends,
            "trajectory": trajectory,
            "period_days": days,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            **snapshot,
            **monitoring,
        }
    finally:
        _log_safe_timing("analytics_read", started_at)


# ---------------------------------------------------------------------------
# WELLBEING REPORT
# ---------------------------------------------------------------------------

@app.get("/report", response_model=ReportResponse)
async def wellbeing_report(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    started_at = time.perf_counter()
    enforce_rate_limit(http_request, "report", limit=20, window_seconds=600)
    try:
        totals = await get_latest_finalized_totals(user_id)
        weekly = await get_latest_weekly_aggregate(user_id)
        trends = await compute_four_week_trends(user_id)
        trajectory = await compute_trajectory(
            user_id,
            totals,
        )
        pending, emotions, recommendations, measurements, latest_risk, monitoring = await asyncio.gather(
            get_pending_score_items(user_id),
            get_recent_emotions(user_id),
            list_recommendations(user_id),
            get_latest_measurements(user_id),
            get_latest_risk(user_id),
            get_longitudinal_monitoring_payload(user_id),
        )

        return {
            "generated_at": _iso_timestamp(),
            "screening_totals": totals,
            "weekly_averages": weekly.model_dump(),
            "four_week_trends": trends,
            "trajectory": trajectory,
            "pending_score_items": pending,
            "assessment_results": [
                {
                    "assessment_type": row["assessment_type"],
                    "total": row["total"],
                    "severity": interpret_assessment_total(row["assessment_type"], int(row["total"])),
                    "completed_at": row["completed_at"],
                }
                for row in measurements
            ],
            "emotional_patterns": [dict(row) for row in emotions],
            "recommendations": [dict(row) for row in recommendations],
            "risk": RiskResponse(
                level=latest_risk["risk_category"] if latest_risk is not None else "UNKNOWN",
                requires_escalation=bool(latest_risk["emergency_flag"]) if latest_risk is not None else False,
            ),
            "safety_status": "screening results are not a diagnosis",
            **monitoring,
        }
    finally:
        _log_safe_timing("report_read", started_at)

@app.get("/account/export")
async def export_user_data(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "account-export",
        limit=3,
        window_seconds=3600,
    )

    data = await run_db(
        _export_user_data_sync,
        user_id,
    )

    return data

@app.delete(
    "/reports",
    response_model=StatusResponse,
)
async def delete_saved_reports(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "report-delete",
        limit=5,
        window_seconds=600,
    )

    await run_db(
        _delete_saved_reports_sync,
        user_id,
    )

    return {
        "status": "deleted"
    }
    
# ---------------------------------------------------------------------------
# WEEKLY SUMMARY
# ---------------------------------------------------------------------------

@app.get("/weekly-summary", response_model=WeeklySummaryRouteResponse)
async def weekly_summary(
    http_request: Request,
    week_start: Optional[str] = None,
    week_end: Optional[str] = None,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "weekly-summary", limit=20, window_seconds=600)
    try:
        start, end = _resolve_week(week_start, week_end)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="week_start and week_end must be valid ISO dates.") from exc

    summary = await run_db(
        _summary_text_sync,
        user_id,
        start,
        end,
    )

    averages = await run_db(
        _weekly_averages_sync,
        user_id,
        start,
        end,
    )

    aggregate = build_weekly_aggregate(
        averages
    )

    return {
        "week_start": start,
        "week_end": end,
        "summary": summary,
        "averages": aggregate.model_dump(),
    }


# ---------------------------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------------------------

@app.get(
    "/recommendations",
    response_model=RecommendationsResponse,
)
async def recommendations(
    http_request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    before_id: Optional[int] = Query(default=None, ge=1),
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "recommendations",
        limit=60,
        window_seconds=600,
    )

    rows = await list_recommendations(
        user_id,
        limit + 1,
        before_id,
    )

    has_more = len(rows) > limit
    page_rows = rows[:limit]

    next_before_id = None

    if has_more and page_rows:
        next_before_id = int(
            page_rows[-1]["id"]
        )

    return {
        "recommendations": [
            dict(row)
            for row in page_rows
        ],
        "limit": limit,
        "has_more": has_more,
        "next_before_id": next_before_id,
    }

# VOICE

@app.post(
    "/voice/chat",
    response_model=ChatResponse,
)
async def voice_chat(
    request: Request,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),

    conversation_id: Optional[str] = Form(
        default=None
    ),

    preferred_language: Optional[str] = Form(
        default=None
    ),

    sleep_hours: Optional[float] = Form(
        default=None
    ),

    deepface_emotion: Optional[str] = Form(
        default=None
    ),

    user_id: int = Depends(
        get_current_user_id
    ),
):
    enforce_rate_limit(
        request,
        "voice-chat",
        limit=10,
        window_seconds=600,
    )

    # Audio is sent to Gemini, so require the
    # user's separate voice-processing consent.
    require_voice_transcription_consent(
        user_id
    )

    if (
        preferred_language
        and preferred_language
        not in SUPPORTED_LANGUAGES
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported language.",
        )

    audio_bytes, mime_type, audio_inspection = (
        await read_validated_audio(audio)
    )

    voice_analysis = await transcribe_audio(
        audio_bytes,
        mime_type,
    )

    transcript = (
        voice_analysis.transcript.strip()
    )

    voice_emotion = None

    if voice_analysis.primary_emotion:
        voice_emotion = VoiceEmotionContext(
            primary=(
                voice_analysis.primary_emotion
            ),
            confidence=(
                voice_analysis.emotion_confidence
            ),
            intensity=(
                voice_analysis.emotion_intensity
            ),
        )

    chat_payload = ChatRequest(
        message=transcript,
        conversationId=conversation_id,
        preferred_language=preferred_language,
        sleep_hours=sleep_hours,
        deepface_emotion=deepface_emotion,
        voice_emotion=voice_emotion,
    )

    result = await chat(
        payload=chat_payload,
        http_request=request,
        background_tasks=background_tasks,
        user_id=user_id,
    )

    text_emotion = result.emotion.model_dump() if result.emotion else None
    combined_emotion = None
    if voice_analysis.primary_emotion:
        combined_emotion = {
            "primary": voice_analysis.primary_emotion,
            "confidence": voice_analysis.emotion_confidence,
            "intensity": voice_analysis.emotion_intensity,
            "source": "voice-context-provider",
        }
    phase2_voice = Phase2VoiceAnalysisResponse(
        transcript=transcript,
        text_emotion=text_emotion,
        # A dedicated acoustic provider is not configured in this application.
        # Do not label a generic voice-context result as acoustic analysis.
        acoustic_emotion=None,
        acoustic_status="unavailable",
        combined_emotion=combined_emotion,
        safety=None,
        duration_seconds=audio_inspection.duration_seconds,
    )
    voice_distress_state = result.distress_state
    try:
        voice_event_id = await create_wellbeing_event(
            user_id=user_id,
            source_type="voice",
            raw_signals={
                "primary_emotion": voice_analysis.primary_emotion,
                "emotion_confidence": voice_analysis.emotion_confidence,
                "emotion_severity": voice_analysis.emotion_intensity,
                "speech_detected": True,
                "acoustic_available": False,
            },
            source_reference=result.user_message_id,
            conversation_id=result.conversation_id,
        )
        voice_distress_state = await compute_and_store_dynamic_distress_state(user_id, voice_event_id)
    except Exception as exc:
        logger.warning("Voice wellbeing event processing skipped: %s", type(exc).__name__)

    # Transcript is returned so the frontend can
    # keep it hidden and reveal it only when the
    # user presses "Transcribe".
    return result.model_copy(
        update={
            "transcript": transcript,
            "voice_analysis": phase2_voice,
            "distress_state": voice_distress_state,
        }
    )

@app.post("/voice/transcribe", response_model=VoiceTranscriptionResponse)
async def voice_transcribe(
    request: Request,
    audio: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        request,
        "voice-transcribe",
        limit=10,
        window_seconds=600,
    )

    require_voice_transcription_consent(user_id)

    audio_bytes, mime_type, _ = await read_validated_audio(audio)

    analysis = await transcribe_audio(
        audio_bytes,
        mime_type,
    )

    voice_emotion = None

    if analysis.primary_emotion:
        voice_emotion = {
            "primary":
                analysis.primary_emotion,
            "confidence":
                analysis.emotion_confidence,
            "intensity":
                analysis.emotion_intensity,
        }

    return {
        "transcript":
            analysis.transcript.strip(),

        "voice_emotion":
            voice_emotion,
    }


# ---------------------------------------------------------------------------
# EMOTION
# ---------------------------------------------------------------------------

@app.post("/emotion/analyze", response_model=AnalyzeFrameResponse)
async def emotion_analysis(
    request: AnalyzeFrameRequest,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "visual-emotion", limit=10, window_seconds=600)
    if not DEEPFACE_ENABLED:
        raise HTTPException(status_code=503, detail="Visual emotion analysis is not enabled.")
    return await analyze_frame(
        request.image_base64
    )
    
@app.post("/screening/{session_id}/answer", response_model=AssessmentAnswerResponse)
async def submit_screening_answer(
    session_id: str,
    request: AssessmentAnswerRequest,
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(http_request, "assessment-answer", limit=80, window_seconds=600)
    session = await run_db(_get_owned_session_sync, user_id, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Screening session not found.")
    if request.conversation_id and request.conversation_id != session["conversation_id"]:
        raise HTTPException(status_code=409, detail="Answer does not belong to this conversation.")
    result = await record_session_item(
        user_id=user_id,
        session_id=session_id,
        item_no=request.item_id,
        raw_score=request.raw_score,
        evidence=request.evidence or "",
    )

    if not result.get("session_found"):
        raise HTTPException(
            status_code=404,
            detail="Screening session not found."
        )

    if result.get("reason") in {"Invalid screening item.", "Score is outside the allowed range."}:
        raise HTTPException(
            status_code=422,
            detail=result["reason"],
        )
    if not result.get("accepted"):
        raise HTTPException(
            status_code=409,
            detail=result.get("reason", "Screening session is not active."),
        )

    if result.get("completed"):
        totals = await get_latest_finalized_totals(user_id)
        trajectory = await compute_trajectory(user_id, totals)
        risk_details = get_risk_details(
            phq9=totals.get("PHQ-9"),
            gad7=totals.get("GAD-7"),
            pss10=totals.get("PSS-10"),
            sleep_hours=None,
            emotion=None,
            trajectory=trajectory,
            emergency=False,
        )
        await save_risk_assessment(user_id, risk_details, totals)
        try:
            event_id = await create_wellbeing_event(
                user_id=user_id,
                source_type="assessment",
                raw_signals={
                    "risk_category": risk_details["risk_category"],
                    "assessment_trajectory": trajectory,
                    "emergency": False,
                },
                source_reference=session_id,
                conversation_id=session.get("conversation_id"),
                assessment_id=session_id,
            )
            await compute_and_store_dynamic_distress_state(user_id, event_id)
        except Exception as exc:
            logger.warning("Assessment wellbeing event processing skipped: %s", type(exc).__name__)

    return {
        "session_id": session_id,
        "scale": session["scale"],
        **result,
    }

@app.delete(
    "/account",
    response_model=StatusResponse,
)
async def delete_account(
    http_request: Request,
    user_id: int = Depends(get_current_user_id),
):
    enforce_rate_limit(
        http_request,
        "account-delete",
        limit=3,
        window_seconds=3600,
    )

    await run_db(
        _delete_account_sync,
        user_id,
    )

    return {
        "status": "deleted"
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
