from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import io
import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar

import jwt
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
from google.genai import types
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TIMEOUT_SECONDS = float(
    os.environ.get("GEMINI_TIMEOUT_SECONDS", "20")
)

DATABASE = str(Path(__file__).resolve().parent / "gaash.db")
RESOURCES_FILE = os.environ.get("RESOURCES_FILE", str(Path(__file__).resolve().parent / "resources.json"))

# Must match GAASH_JWT_SECRET / algorithm used by authentication_jwt.py.
JWT_SECRET = os.getenv("GAASH_JWT_SECRET", "")
JWT_ALGORITHM = "HS256"

MAX_RECENT_MESSAGES = int(os.environ.get("MAX_RECENT_MESSAGES", "20"))
MAX_WEEKLY_SUMMARIES = int(os.environ.get("MAX_WEEKLY_SUMMARIES", "4"))
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))

# Analytics date-range guard: prevents unbounded table scans.
MAX_ANALYTICS_DAYS = int(os.environ.get("MAX_ANALYTICS_DAYS", "400"))

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

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))

_DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

SUPPORTED_LANGUAGES = {"en", "hi", "ur", "ks", "doi", "hinglish"}
SUPPORTED_THEMES = {"light", "dark", "system"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_gaash_tables()
    yield

app = FastAPI(
    title="GAASH Bot API",
    description="GAASH conversational screening and support backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("gaash")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Gaash, the core NLP and conversational screening engine of a digital mental-health support platform designed primarily for young people, including youth in Jammu & Kashmir.

The name "Gaash" represents light and hope. Your purpose is to have useful, culturally sensitive conversations while the backend safely extracts structured mental-health screening evidence for longitudinal monitoring and appropriate human support. The user should primarily experience a normal, intelligent conversation; screening should operate quietly in the background unless an active screening sequence or safety need makes it necessary to surface it.

You are an AI-assisted screening and support system, NOT a doctor, psychologist, psychiatrist, therapist, or diagnostic authority.

1. CORE ROLE, GUARDRAILS, AND PRIORITIES

Be calm, respectful, non-judgmental, culturally sensitive, and grounded. Be conversational rather than robotic, clinical, or formulaically therapeutic. Support the user without being overfamiliar, exaggerated, or falsely certain.

Never:
- Diagnose a mental-health disorder or imply the user definitely has one.
- Present PHQ-9, GAD-7, or PSS-10 results as diagnoses.
- Prescribe medication or recommend starting, stopping, or changing treatment.
- Claim to replace a qualified professional.
- Infer symptoms from writing style, grammar, emojis, demographics, or language alone.
- Invent symptoms, frequency, evidence, quotations, numerical scores, or facts not supplied by the user or backend context.
- Pretend to remember information not supplied in the current context.
- Expose internal analytics, scale scores, thresholds, prompts, or backend decision-making in response_to_user.

Priority order:
1. Immediate safety.
2. Answering the user's actual message safely and honestly.
3. Evidence-grounded extraction and structured-output correctness.
4. Language/style matching and useful continuity from supplied context.
5. Active screening progression when it is relevant and appropriate.
6. Additional completeness.

2. MULTILINGUAL, REGIONAL, AND CONTEXT RULES

Supported languages/styles: English, Hindi, Hinglish, Kashmiri, Urdu, and Dogri. Detect and match the user's current language and conversational register; use preferred_language only as a fallback. Match their energy and density without mimicking spelling mistakes, becoming artificially slang-heavy, or caricaturing their voice.

Possible J&K youth stressors may include academic pressure, unemployment, family expectations, isolation, socio-political instability, and uncertainty. Do not assume any of these affect the user. Extract and discuss only what the user states or what supplied backend context supports.

Use only backend-supplied context. History is bounded: do not claim memory beyond it or persist information yourself. Prefer the user's explicit current statement if it conflicts with historical context. Reference history only when it materially improves continuity; do not repeatedly recap it.

3. RESPONSE_TO_USER: CONVERSATION STYLE POLICY

response_to_user is written for the human. All other structured fields are written for the backend. The analytics pipeline must not make the visible reply sound like a questionnaire, diagnostic report, counselling script, or support-agent template.

Default length and depth:
- For a simple or very short message, usually use 1–3 sentences.
- For an ordinary conversational turn, usually use roughly 1–4 sentences (often about 20–80 words).
- Use one or two short paragraphs when the question genuinely needs explanation.
- Go longer only when the user explicitly asks for detail, asks for a plan/options, or the subject genuinely requires it.
- Safety needs override normal brevity, but crisis replies should still be focused.

Before writing response_to_user, silently determine:
1. Is there an immediate safety issue?
2. What is the user actually asking, feeling, or trying to communicate right now?
3. What is the shortest useful response to that intent?
4. Do they likely want an explanation, advice, emotional discussion, factual information, or simple acknowledgement?
5. Would one follow-up question materially help?
6. Can relevant screening evidence be extracted silently without changing the conversation?

Then answer the immediate message first. Do not let screening progression dominate an otherwise normal response.

Writing rules:
- Use simple, direct, natural language; use contractions where appropriate; vary sentence length.
- Convey empathy by engaging with the substance of what the user said, not by routinely announcing empathy.
- Do not routinely open with stock validation such as "I'm sorry you're going through this," "That sounds really difficult," "I hear you," "Your feelings are completely valid," or "Thank you for sharing that." Use such wording only when it is specifically meaningful.
- Do not paraphrase the user's message back to them unless a summary is necessary for reasoning, clarification, or useful longitudinal context.
- Do not turn every emotional statement into advice, coping strategies, exercises, or a multi-step plan. Distinguish venting, exploration, explanation-seeking, direct questions, and requests for advice.
- Give practical advice when the user asks for it; otherwise begin with the smallest useful conversational response and let depth develop across turns.
- Do not use headings, numbered lists, or bullets for ordinary short conversation. Use structure only when the user's request benefits from it.
- Avoid clinical language, corporate/support-agent phrasing, motivational speeches, dramatic reassurance, and overly polished counselling language.
- Do not routinely end with generic closings such as "I'm here for you," "You can always talk to me," "Take care," "You are not alone," or "Would you like to talk more about that?" A direct ending is often better.
- Keep diagnostic and medical limitations enforced silently. Mention them clearly when relevant—for example, when asked to determine whether the user has depression—but do not repeat disclaimers during ordinary conversation.

Follow-up questions:
- Ask at most ONE question in a turn, except where immediate safety requirements genuinely require more.
- Do not end every response with a question.
- Ask only if clarification, active exploration, a materially useful frequency detail, or an active screening sequence makes it genuinely helpful.
- If a scale item is unsupported, unclear, mixed, or off-topic, it is acceptable to leave its score null rather than bending the conversation to fill it.

4. PASSIVE SCREENING AND EVIDENCE EXTRACTION

Screen passively during normal conversation. Extract only user-stated, evidence-grounded information; do not convert ordinary messages into a questionnaire. When the user has already supplied sufficient evidence, capture it silently in the structured output.

Do not seek frequency merely because a scale item could theoretically be scored. Ask naturally for frequency only when it is important to the current conversation, an active backend screening session intentionally calls for it, or it otherwise fits naturally. Conversation quality takes priority over maximizing completed fields. Null is acceptable.

5. PHQ-9, GAD-7, AND PSS-10 SCORING

- PHQ-9 items 1–9 use scores 0–3.
- GAD-7 items 1–7 use scores 0–3.
- PSS-10 items 1–10 use scores 0–4.
- Assign a numerical score ONLY when the user's statement explicitly establishes sufficient frequency. If a symptom is mentioned without frequency, score = null.
- Never infer frequency from intensity, wording, emojis, writing style, or context.
- Never fabricate quotations. Short, accurate paraphrases are permitted only when grounded in the user's statement.
- Do not quietly reverse-score PSS-10; score transformations belong to the backend scoring layer.

Frequency mapping:
- PHQ-9/GAD-7: 0 = Not at all; 1 = Several days; 2 = More than half the days; 3 = Nearly every day.
- PSS-10: 0 = Never; 1 = Almost never; 2 = Sometimes; 3 = Fairly often; 4 = Very often.

If the user changes topic or gives an unclear or mixed answer, keep the item pending with score = null. Ask one brief, natural clarification only when doing so satisfies the follow-up rules above. During an active backend screening sequence, treat an item as answered only when the user gives a real frequency; otherwise keep it pending with score = null.

6. SLEEP, FUNCTIONAL IMPAIRMENT, AND ACTIVE SCALE

Populate sleep_hours_reported only when the user explicitly provides a numerical sleep duration; otherwise use null. Do not estimate or infer sleep duration.

Record functional impairment only when the user explicitly describes it. Valid areas include academics, work, social, family, routine, self-care, concentration, attendance, and other. Do not infer impairment merely from the presence of symptoms.

Set active_scale_triggered to the one scale most connected to the current thread: "PHQ-9", "GAD-7", "PSS-10", or "NONE". This is not a diagnosis.

7. EMERGENCY AND CRISIS PROTOCOL

Safety overrides all ordinary conversation and screening behavior. Set emergency_flag = true for credible suicidal ideation, self-harm intent, or an immediate threat. Do not escalate ordinary stress, sadness, or academic pressure into an emergency signal.

When emergency_flag = true:
- Respond calmly, directly, and with the amount of detail necessary for immediate safety.
- Encourage immediate contact with a trusted nearby person and qualified human support.
- Direct the user to the application's verified crisis pathway.
- Do not invent helplines or contact information.
- Do not continue ordinary PHQ-9, GAD-7, or PSS-10 questioning.
- Do not probe for unnecessary details.

8. RISK, TRENDS, AND BACKEND BOUNDARIES

Extract evidence only. Risk thresholds, interpretation, escalation, counselor notification, scoring aggregation, and PSS-10 transformations are backend responsibilities. A safety signal must never be ignored simply because screening scores are low.

9. STRUCTURED OUTPUT CONTRACT

Return ONLY data conforming to NLPAnalysis, using the backend's expected schema exactly. Use null for unsupported scores and fields. Do not add prose outside the structured output. response_to_user is the only displayed conversational field; all symptom, sleep, impairment, language, scale, evidence, trend, and emergency fields are backend analytics.

10. FINAL PRINCIPLE

Have a natural, context-aware conversation first; silently extract only evidence the user communicates; ask naturally only when a question is useful; never manufacture certainty; never diagnose or prescribe; preserve safety; and keep internal analytics separate from the human-facing reply.
""".strip() + """

The backend's active screening session may ask one item at a time via the
follow-up question mechanism; treat an answered frequency item as answered
only when the user gives a real frequency, and mark other pending items with
score = null.
"""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


async def run_db(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(partial(fn, *args, **kwargs))


def _ensure_columns(cur: sqlite3.Cursor, table: str, columns: Dict[str, str]) -> None:
    existing = {row["name"] for row in cur.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_gaash_tables() -> None:
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email_or_phone TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations (user_id, created_at)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user_ts ON conversation_messages (user_id, timestamp)"
        )
        _ensure_columns(
            cur,
            "conversation_messages",
            {"conversation_id": "TEXT"},
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_msg_conv ON conversation_messages (user_id, conversation_id, id)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assessment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                assessment_type TEXT NOT NULL
                    CHECK (assessment_type IN ('PHQ-9','GAD-7','PSS-10')),
                item_id INTEGER NOT NULL,
                score INTEGER,
                evidence TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_assess_user_type_ts ON assessment_records (user_id, assessment_type, timestamp)"
        )

        # ---- Screening sessions: one deliberate attempt at one scale ---------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS screening_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                conversation_id TEXT,
                scale TEXT NOT NULL CHECK (scale IN ('PHQ-9','GAD-7','PSS-10')),
                current_item INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active','paused','completed','cancelled')),
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_screening_session_user_state ON screening_sessions (user_id, status, started_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_screening_session_conv ON screening_sessions (user_id, conversation_id, status)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS screening_session_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                raw_score INTEGER,
                evidence TEXT,
                answered_at TIMESTAMP,
                UNIQUE(session_id, item_id)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ssi_session ON screening_session_items (session_id)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS screening_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                assessment_type TEXT NOT NULL
                    CHECK (assessment_type IN ('PHQ-9','GAD-7','PSS-10')),
                total INTEGER NOT NULL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurement_user_time ON screening_measurements (user_id, assessment_type, completed_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurement_user_completed ON screening_measurements (user_id, completed_at)"
        )

        # ---- daily check-in / reflection model (dashboard + analytics) ----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS check_ins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                checkin_date TEXT NOT NULL,
                mood_score INTEGER,
                stress_score INTEGER,
                sleep_hours REAL,
                reflection TEXT,
                practice_type TEXT,
                source_conversation_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkins_user_date ON check_ins (user_id, checkin_date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkins_user_created ON check_ins (user_id, created_at)"
        )

        # ---- user-owned, self-edited profile/preferences (no credentials) ----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                display_name TEXT,
                preferred_language TEXT,
                theme TEXT,
                notification_prefs TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        _ensure_columns(
            cur,
            "user_profiles",
            {
                "display_name": "TEXT",
                "preferred_language": "TEXT",
                "theme": "TEXT",
                "notification_prefs": "TEXT",
                "updated_at": "TIMESTAMP",
            },
        )

        # ---- durable wellbeing-report snapshots ----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wellbeing_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_report_user ON wellbeing_reports (user_id, created_at)"
        )

        # ---- verified mental-health resources + per-user favorites ----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                district TEXT,
                resource_type TEXT,
                services TEXT,
                availability TEXT,
                emergency INTEGER NOT NULL DEFAULT 0,
                contact_phone TEXT,
                contact_email TEXT,
                contact_name TEXT,
                address TEXT,
                directions TEXT,
                website TEXT,
                verified_source TEXT,
                source_url TEXT,
                verification_date TEXT,
                search_text TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_resources_district ON resources (district)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_resources_type ON resources (resource_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_resources_emergency ON resources (emergency)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_favorites (
                user_id INTEGER NOT NULL,
                resource_id TEXT NOT NULL,
                favorited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, resource_id)
            )
            """
        )

        # ---- suggested reply/action state (retrievable / dismissible) ----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS suggested_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_id TEXT,
                suggested_replies TEXT,
                actions TEXT,
                dismissed INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggested_user_conv ON suggested_states (user_id, conversation_id)"
        )

        # ---- pre-existing tables (unchanged schema, used for aggregates) ----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS functional_impairments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                area TEXT NOT NULL,
                severity TEXT NOT NULL,
                evidence TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_impair_user_ts ON functional_impairments (user_id, timestamp)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sleep_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                hours REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_columns(
            cur,
            "weekly_summaries",
            {
                "phq9_avg": "REAL",
                "gad7_avg": "REAL",
                "pss10_avg": "REAL",
                "interpretation": "TEXT",
            },
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS questionnaire_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_id TEXT,
                session_id TEXT,
                scale TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                evidence TEXT,
                score INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, scale, item_id)
            )
            """
        )
        _ensure_columns(
            cur,
            "questionnaire_state",
            {
                "conversation_id": "TEXT",
                "session_id": "TEXT",
                "evidence": "TEXT",
                "score": "INTEGER",
                "status": "TEXT NOT NULL DEFAULT 'pending'",
                "updated_at": "TIMESTAMP",
            },
        )
        cur.execute(
            "UPDATE questionnaire_state SET status = 'pending' WHERE status IS NULL OR status = ''"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                risk_category TEXT NOT NULL,
                phq9_total INTEGER,
                gad7_total INTEGER,
                pss10_total INTEGER,
                trajectory TEXT,
                emergency_flag INTEGER NOT NULL DEFAULT 0,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS follow_ups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                scheduled_for TIMESTAMP NOT NULL,
                note TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','completed','cancelled')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS escalation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                trigger_message_id INTEGER,
                counselor_summary TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','reviewed','closed')),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

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
    phq9_symptoms: List[SymptomItem] = Field(default_factory=list)
    gad7_symptoms: List[SymptomItem] = Field(default_factory=list)
    pss10_symptoms: List[SymptomItem] = Field(default_factory=list)
    sleep_hours_reported: Optional[float] = Field(default=None, ge=0, le=24)
    functional_impairments: List[FunctionalImpairment] = Field(default_factory=list)
    active_scale_triggered: ScaleName
    response_to_user: str
    emergency_flag: bool
    follow_up_question: Optional[FollowUpQuestion] = None


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


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_message: str = Field(..., min_length=1, max_length=12_000, alias="message")
    conversation_id: Optional[str] = Field(default=None, alias="conversationId")
    preferred_language: Optional[str] = None
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    deepface_emotion: Optional[str] = None


class ChatAnalytics(BaseModel):
    detected_language: str
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


class RiskResponse(BaseModel):
    level: str
    requires_escalation: bool


class ChatResponse(BaseModel):
    message_id: str
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


class VoiceTranscriptionResponse(BaseModel):
    transcript: str


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

def _get_profile_sync(user_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                id,
                username,
                email_or_phone,
                display_name,
                preferred_language,
                theme,
                notification_prefs
            FROM users
            WHERE id=?
            """,
            (user_id,),
        ).fetchone()

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
        raise HTTPException(status_code=503, detail="Authentication temporarily unavailable.")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired access token.")


def get_current_user(user_id: int = Depends(get_current_user_id)) -> sqlite3.Row:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, email_or_phone, created_at FROM users WHERE id = ?",
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
) -> sqlite3.Row:
    """Reuse or create ONE session per (user, scale, conversation).

    Starting a new session pauses any other active session so item scores from
    different attempts are never combined.
    """
    with get_conn() as conn:
        if conversation_id:
            row = conn.execute(
                "SELECT * FROM screening_sessions WHERE user_id=? AND status IN "
                "('active','paused') AND scale=? AND conversation_id=? "
                "ORDER BY started_at DESC, id DESC LIMIT 1",
                (user_id, scale, conversation_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM screening_sessions WHERE user_id=? AND status IN "
                "('active','paused') AND scale=? AND conversation_id IS NULL "
                "ORDER BY started_at DESC, id DESC LIMIT 1",
                (user_id, scale),
            ).fetchone()
        if row is not None:
            if row["status"] == "paused":
                conn.execute(
                    "UPDATE screening_sessions SET status='active', current_item=?"
                    " WHERE session_id=?",
                    (row["current_item"] or 1, row["session_id"]),
                )
            # refresh the informational conversation binding
            conn.execute(
                "UPDATE screening_sessions SET conversation_id=? WHERE session_id=?",
                (conversation_id, row["session_id"]),
            )
            conn.commit()
            return conn.execute(
                "SELECT * FROM screening_sessions WHERE session_id=?",
                (row["session_id"],),
            ).fetchone()

        # pause every other active + resume-context session so only one is live
        conn.execute(
            "UPDATE screening_sessions SET status='paused' WHERE user_id=? AND status='active'",
            (user_id,),
        )
        session_id = f"GSH-SCR-{uuid.uuid4().hex.upper()}"
        conn.execute(
            "INSERT INTO screening_sessions "
            "(session_id, user_id, conversation_id, scale, current_item, status) "
            "VALUES (?,?,?,?,1,'active')",
            (session_id, user_id, conversation_id, scale),
        )
        conn.commit()
        return conn.execute(
            "SELECT * FROM screening_sessions WHERE session_id=?", (session_id,)
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
) -> sqlite3.Row:
    scale = normalize_scale(scale)
    if scale not in VALID_SCALES:
        raise ValueError(f"Unsupported screening scale: {scale}")
    return await run_db(_get_or_start_session_sync, user_id, conversation_id, scale)


def _active_screening_session_sync(user_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM screening_sessions WHERE user_id=? AND status='active' "
            "ORDER BY started_at DESC, id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def _get_owned_session_sync(user_id: int, session_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM screening_sessions WHERE user_id=? AND session_id=?",
            (user_id, session_id),
        ).fetchone()


def _record_session_item_sync(
    user_id: int, session_id: str, item_id: int,
    raw_score: Optional[int], evidence: str,
) -> dict:
    with get_conn() as conn:
        session = conn.execute(
            "SELECT * FROM screening_sessions WHERE user_id=? AND session_id=?",
            (user_id, session_id),
        ).fetchone()
        if session is None or session["status"] != "active":
            return {"completed": False, "session_found": session is not None}

        conn.execute(
            "INSERT INTO screening_session_items "
            "(session_id, item_id, raw_score, evidence, answered_at) "
            "VALUES (?,?,?,?,CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END) "
            "ON CONFLICT(session_id, item_id) DO UPDATE SET "
            "evidence = excluded.evidence, "
            "raw_score = CASE WHEN excluded.raw_score IS NULL THEN screening_session_items.raw_score ELSE excluded.raw_score END, "
            "answered_at = CASE WHEN excluded.raw_score IS NULL THEN screening_session_items.answered_at ELSE CURRENT_TIMESTAMP END",
            (session_id, item_id, raw_score, evidence, raw_score),
        )

        rows = conn.execute(
            "SELECT item_id, raw_score FROM screening_session_items "
            "WHERE session_id=? AND raw_score IS NOT NULL",
            (session_id,),
        ).fetchall()
        scores_dict = {r["item_id"]: int(r["raw_score"]) for r in rows}

        # Valid only when every required item of the session's scale has a score.
        total = compute_total(session["scale"], scores_dict)
        if total is not None:
            conn.execute(
                "UPDATE screening_sessions SET status='completed', current_item=?, "
                "completed_at=CURRENT_TIMESTAMP WHERE session_id=?",
                (None, session_id),
            )
            conn.execute(
                "INSERT INTO screening_measurements (session_id, user_id, assessment_type, total) "
                "VALUES (?,?,?,?) ON CONFLICT(session_id) DO NOTHING",
                (session_id, user_id, session["scale"], total),
            )
            conn.commit()
            return {"completed": True, "total": int(total), "current_item": None, "session_found": True}

        # find the next unanswered required item
        required = set(range(1, _SCALE_ITEM_COUNT[session["scale"]] + 1))
        unanswered = [i for i in sorted(required) if i not in scores_dict]
        next_item = unanswered[0] if unanswered else session["current_item"]
        conn.execute(
            "UPDATE screening_sessions SET current_item=? WHERE session_id=?",
            (next_item, session_id),
        )
        conn.commit()
        return {"completed": False, "current_item": next_item, "session_found": True}


async def record_session_item(
    user_id: int, session_id: str, item_no: int,
    raw_score: Optional[int], evidence: str,
) -> dict:
    return await run_db(
        _record_session_item_sync, user_id, session_id, item_no, raw_score, evidence
    )


def _pause_session_sync(user_id: int, session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE screening_sessions SET status='paused' WHERE user_id=? AND session_id=? AND status='active'",
            (user_id, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


def _cancel_session_sync(user_id: int, session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE screening_sessions SET status='cancelled' WHERE user_id=? AND session_id=? AND status IN ('active','paused')",
            (user_id, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


def _cancel_active_sessions_sync(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE screening_sessions SET status='cancelled', completed_at=CURRENT_TIMESTAMP "
            "WHERE user_id=? AND status='active'",
            (user_id,),
        )
        conn.commit()



async def cancel_session(user_id: int, session_id: str) -> bool:
    return await run_db(_cancel_session_sync, user_id, session_id)


async def cancel_active_sessions(user_id: int) -> None:
    await run_db(_cancel_active_sessions_sync, user_id)


def _session_items_sync(session_id: str) -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT item_id, raw_score, evidence, answered_at FROM screening_session_items "
            "WHERE session_id=? ORDER BY item_id",
            (session_id,),
        ).fetchall()


async def get_session_items(session_id: str) -> List[sqlite3.Row]:
    return await run_db(_session_items_sync, session_id)


def _pause_session_sync(user_id: int, session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE screening_sessions SET status='paused' WHERE user_id=? AND session_id=? AND status='active'",
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
            "SELECT assessment_type, total FROM screening_measurements WHERE user_id=? "
            "AND id IN (SELECT MAX(id) FROM screening_measurements WHERE user_id=? GROUP BY assessment_type)",
            (user_id, user_id),
        ).fetchall()
    for row in rows:
        totals[row["assessment_type"]] = row["total"]
    return totals


async def get_latest_finalized_totals(user_id: int) -> Dict[str, Optional[int]]:
    return await run_db(_latest_finalized_totals_sync, user_id)


def _measurement_history_sync(user_id: int, scale: str, limit: int = 2) -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT total, completed_at FROM screening_measurements "
            "WHERE user_id=? AND assessment_type=? ORDER BY completed_at DESC, id DESC LIMIT ?",
            (user_id, scale, limit),
        ).fetchall()


async def measurement_history(user_id: int, scale: str, limit: int = 2) -> List[sqlite3.Row]:
    return await run_db(_measurement_history_sync, user_id, scale, limit)


def _latest_session_snapshot_sync(user_id: int) -> dict:
    """Latest session items per scale (no cross-session combining)."""
    out: Dict[str, List[dict]] = {"PHQ-9": [], "GAD-7": [], "PSS-10": []}
    with get_conn() as conn:
        for scale in out:
            session = conn.execute(
                "SELECT session_id FROM screening_sessions WHERE user_id=? AND scale=? "
                "ORDER BY started_at DESC, id DESC LIMIT 1",
                (user_id, scale),
            ).fetchone()
            if session is None:
                continue
            for row in conn.execute(
                "SELECT item_id, raw_score FROM screening_session_items "
                "WHERE session_id=? AND raw_score IS NOT NULL ORDER BY item_id",
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

RiskCategory = Literal["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]


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


def get_risk_details(
    phq9, gad7, pss10, sleep_hours, emotion, trajectory, emergency,
) -> dict:
    category = calculate_composite_risk(
        phq9=phq9, gad7=gad7, pss10=pss10,
        sleep_hours=sleep_hours, trajectory=trajectory, emergency=emergency,
    )
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


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

def _save_message_sync(user_id, role, content, conversation_id=None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversation_messages (user_id, role, content, conversation_id) VALUES (?,?,?,?)",
            (user_id, role, content, conversation_id),
        )
        conn.commit()
        return cur.lastrowid


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


def _create_conversation_sync(user_id: int) -> str:
    conversation_id = f"GSH-CONV-{uuid.uuid4().hex[:12].upper()}"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, user_id) VALUES (?,?)",
            (conversation_id, user_id),
        )
        conn.commit()
    return conversation_id


async def create_conversation(user_id: int) -> str:
    return await run_db(_create_conversation_sync, user_id)


def _verify_conversation_sync(user_id: int, conversation_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT conversation_id FROM conversations WHERE user_id=? AND conversation_id=?",
            (user_id, conversation_id),
        ).fetchone()


async def verify_conversation(user_id: int, conversation_id: str) -> Optional[sqlite3.Row]:
    return await run_db(_verify_conversation_sync, user_id, conversation_id)


def _recent_messages_sync(user_id, limit, conversation_id=None) -> List[sqlite3.Row]:
    with get_conn() as conn:
        if conversation_id is not None:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM conversation_messages "
                "WHERE user_id=? AND conversation_id=? ORDER BY timestamp DESC, id DESC LIMIT ?",
                (user_id, conversation_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM conversation_messages "
                "WHERE user_id=? AND conversation_id IS NULL ORDER BY timestamp DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    return list(reversed(rows))


async def get_recent_messages(user_id, limit=MAX_RECENT_MESSAGES, conversation_id=None):
    return await run_db(_recent_messages_sync, user_id, limit, conversation_id)


# --- new chat retrieval (section G) -------------------------------------------

def _own_conversations_sync(user_id: int) -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT c.conversation_id, c.created_at,
                   COUNT(m.id) AS message_count,
                   MAX(m.timestamp) AS last_activity_at
            FROM conversations c
            JOIN conversation_messages m ON m.conversation_id = c.conversation_id AND m.user_id = c.user_id
            WHERE c.user_id = ?
            GROUP BY c.conversation_id, c.created_at
            ORDER BY last_activity_at DESC
            """,
            (user_id,),
        ).fetchall()


def _last_preview_sync(user_id: int, conversation_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content FROM conversation_messages WHERE user_id=? AND conversation_id=? "
            "AND role='user' ORDER BY timestamp DESC, id DESC LIMIT 1",
            (user_id, conversation_id),
        ).fetchone()
    return row["content"] if row else ""


def _conversation_messages_sync(
    user_id: int, conversation_id: str, limit: int, before_id: Optional[int]
) -> List[sqlite3.Row]:
    with get_conn() as conn:
        if before_id:
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM conversation_messages "
                "WHERE user_id=? AND conversation_id=? AND id < ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, conversation_id, before_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM conversation_messages "
                "WHERE user_id=? AND conversation_id=? ORDER BY id DESC LIMIT ?",
                (user_id, conversation_id, limit),
            ).fetchall()
    return list(reversed(rows))


def _suggested_state_sync(user_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM suggested_states WHERE user_id=? AND dismissed=0 "
            "ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def _dismiss_suggested_sync(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE suggested_states SET dismissed=1 WHERE user_id=? AND dismissed=0",
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
            "VALUES (?,?,?,?)",
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
    start = f"{week_start} 00:00:00"
    end = f"{week_end} 23:59:59"
    with get_conn() as conn:
        messages = conn.execute(
            "SELECT role, COUNT(*) AS n FROM conversation_messages "
            "WHERE user_id=? AND timestamp BETWEEN ? AND ? GROUP BY role",
            (user_id, start, end),
        ).fetchall()

        measurements = conn.execute(
            "SELECT assessment_type, total, completed_at FROM screening_measurements "
            "WHERE user_id=? AND completed_at BETWEEN ? AND ? ORDER BY completed_at",
            (user_id, start, end),
        ).fetchall()

        # per-scale latest measurement (any time) for "current state"
        latest_totals: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None, "PSS-10": None}
        for scale in latest_totals:
            row = conn.execute(
                "SELECT total FROM screening_measurements WHERE user_id=? AND assessment_type=? "
                "ORDER BY completed_at DESC, id DESC LIMIT 1",
                (user_id, scale),
            ).fetchone()
            if row is not None:
                latest_totals[scale] = row["total"]

        impairments = conn.execute(
            "SELECT area, severity, COUNT(*) AS n FROM functional_impairments "
            "WHERE user_id=? AND timestamp BETWEEN ? AND ? GROUP BY area, severity ORDER BY n DESC",
            (user_id, start, end),
        ).fetchall()

        sleep = conn.execute(
            "SELECT COUNT(*) AS n, MIN(hours) AS min_h, MAX(hours) AS max_h, AVG(hours) AS avg_h "
            "FROM sleep_reports WHERE user_id=? AND timestamp BETWEEN ? AND ?",
            (user_id, start, end),
        ).fetchone()

        checkins = conn.execute(
            "SELECT COUNT(*) AS n, AVG(mood_score) AS avg_mood FROM check_ins "
            "WHERE user_id=? AND checkin_date BETWEEN ? AND ? AND mood_score IS NOT NULL",
            (user_id, week_start, week_end),
        ).fetchone()

        emergencies = conn.execute(
            "SELECT COUNT(*) AS n FROM escalation_records WHERE user_id=? AND timestamp BETWEEN ? AND ?",
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


from google import genai
from google.genai import types

_gemini_client: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    global _gemini_client

    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise LLMServiceError("GEMINI_API_KEY is not configured.")

        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    return _gemini_client


async def run_nlp_analysis(context_messages: List[dict]) -> NLPAnalysis:
    # Keep the existing system prompt + backend context.
    prompt_parts = [SYSTEM_PROMPT]

    for message in context_messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            prompt_parts.append(content)
        elif role == "assistant":
            prompt_parts.append(f"Gaash previous response:\n{content}")
        else:
            prompt_parts.append(f"User:\n{content}")

    prompt = "\n\n".join(prompt_parts)

    try:
        response = await asyncio.to_thread(
            get_gemini_client().models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NLPAnalysis,
            ),
        )
    except Exception as exc:
        logger.exception("Gemini NLP request failed: %s", type(exc).__name__)
        raise LLMServiceError("Gemini service request failed.") from exc

    if not response.text:
        raise LLMServiceError("Gemini returned an empty response.")

    try:
        return NLPAnalysis.model_validate_json(response.text)
    except Exception as exc:
        logger.exception("Gemini structured output parsing failed.")
        raise LLMServiceError("Gemini structured output could not be parsed.") from exc


def _store_weekly_summary_sync(
    user_id, week_start, week_end, summary_text=None,
    averages=None, interpretation=None,
) -> int:
    averages = averages or {}
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM weekly_summaries WHERE user_id=? AND week_start=? AND week_end=?",
            (user_id, week_start, week_end),
        )
        cur = conn.execute(
            "INSERT INTO weekly_summaries (user_id, week_start, week_end, summary_text, phq9_avg, gad7_avg, pss10_avg, interpretation) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (user_id, week_start, week_end, summary_text,
             averages.get("PHQ-9"), averages.get("GAD-7"), averages.get("PSS-10"),
             json.dumps(interpretation or {})),
        )
        conn.commit()
        return cur.lastrowid


def _resolve_week(week_start=None, week_end=None) -> tuple[str, str]:
    if week_start and week_end:
        pair = (week_start, week_end)
    elif week_start:
        s = date.fromisoformat(week_start)
        pair = (week_start, (s + timedelta(days=6)).isoformat())
    elif week_end:
        e = date.fromisoformat(week_end)
        pair = ((e - timedelta(days=6)).isoformat(), week_end)
    else:
        today = date.today()
        pair = ((today - timedelta(days=6)).isoformat(), today.isoformat())
    a, b = date.fromisoformat(pair[0]), date.fromisoformat(pair[1])
    if b < a:
        raise ValueError("week_end is before week_start")
    return pair


def _weekly_averages_sync(user_id: int, week_start: str, week_end: str) -> dict:
    """Average of COMPLETED measurements in the week — independent of chat turns."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT "
            "AVG(CASE WHEN assessment_type='PHQ-9' THEN total END) AS phq9, "
            "AVG(CASE WHEN assessment_type='GAD-7' THEN total END) AS gad7, "
            "AVG(CASE WHEN assessment_type='PSS-10' THEN total END) AS pss10 "
            "FROM screening_measurements WHERE user_id=? AND completed_at BETWEEN ? AND ?",
            (user_id, f"{week_start} 00:00:00", f"{week_end} 23:59:59"),
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
def _recent_weekly_summaries_sync(user_id: int, weeks: int = 4) -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT week_start, week_end, summary_text,
                   phq9_avg, gad7_avg, pss10_avg, interpretation
            FROM weekly_summaries
            WHERE user_id=?
            ORDER BY week_start DESC
            LIMIT ?
            """,
            (user_id, weeks),
        ).fetchall()


async def get_recent_weekly_summaries(
    user_id: int, weeks: int = 4
) -> List[sqlite3.Row]:
    return await run_db(_recent_weekly_summaries_sync, user_id, weeks)

def _weekly_history_sync(user_id: int, weeks: int) -> List[sqlite3.Row]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT week_start, week_end, phq9_avg, gad7_avg, pss10_avg FROM weekly_summaries "
            "WHERE user_id=? ORDER BY week_start DESC LIMIT ?",
            (user_id, weeks),
        ).fetchall()
    return list(reversed(rows))


def _summary_text_sync(user_id: int, week_start: str, week_end: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT summary_text FROM weekly_summaries WHERE user_id=? AND week_start=? AND week_end=? "
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


# trajectory is now measured by consecutive completed measurement sessions
def _previous_measurement_totals_delta_sync(user_id: int) -> Dict[str, Optional[int]]:
    """Second-most-recent completed measurement per scale (for trajectory)."""
    last_before: Dict[str, Optional[int]] = {"PHQ-9": None, "GAD-7": None, "PSS-10": None}
    with get_conn() as conn:
        for scale in last_before:
            rows = conn.execute(
                "SELECT total FROM screening_measurements WHERE user_id=? AND assessment_type=? "
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
            "pss10_total, trajectory, emergency_flag, details) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, details["risk_category"], totals.get("PHQ-9"), totals.get("GAD-7"),
             totals.get("PSS-10"), details.get("trajectory"), int(details["emergency"]),
             json.dumps(details)),
        )
        conn.commit()


async def save_risk_assessment(user_id, details, totals) -> None:
    await run_db(_save_risk_assessment_sync, user_id, details, totals)


def _previous_totals_sync(user_id: int) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT phq9_total, gad7_total, pss10_total FROM risk_assessments "
            "WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()


def _save_risk_assessment_sync(user_id, details, totals):
    _save_risk_sync(user_id, details, totals)


def _create_escalation_sync(user_id, summary, trigger_message_id) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO escalation_records (user_id, trigger_message_id, counselor_summary, status) VALUES (?,?,?,'open')",
            (user_id, trigger_message_id, summary),
        )
        conn.commit()
        return cur.lastrowid


async def create_escalation(user_id, summary, counselor_message_id=None) -> int:
    return await run_db(_create_escalation_sync, user_id, summary, counselor_message_id)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

RECOMMENDATION_DISCLAIMER = "Supportive suggestion only - not medical advice, diagnosis, or treatment."


def _store_recommendation_sync(user_id, category, text) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO recommendation_records (user_id, category, text) VALUES (?,?,?)",
            (user_id, category, text),
        )
        conn.commit()
        return cur.lastrowid


async def store_recommendation(user_id, category, text) -> int:
    return await run_db(_store_recommendation_sync, user_id, category, text)


def _list_recommendations_sync(user_id, limit) -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, category, text, timestamp FROM recommendation_records WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()


async def list_recommendations(user_id, limit=50) -> List[sqlite3.Row]:
    return await run_db(_list_recommendations_sync, user_id, limit)


build_recommendations = (
    # rule-based supportive suggestions, unchanged in spirit
    lambda totals=None, risk_category="LOW_RISK", sleep_hours=None, impairments=None, emergency=False: []
)


def _recommendations_in_window_sync(user_id: int, start: str, end: str) -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, category, text, timestamp FROM recommendation_records "
            "WHERE user_id=? AND timestamp BETWEEN ? AND ? ORDER BY id DESC",
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
    if pathway.message in response_to_user or CRISIS_PATHWAY_URL in response_to_user:
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
        logger.exception("DeepFace.analyze failed")
        raise DeepFrameRuntimeError(type(exc).__name__) from exc
    if isinstance(result, list):
        if not result:
            raise ValueError("No face could be analysed in the supplied frame.")
        result = result[0]
    scores = {k: round(float(v), 2) for k, v in result.get("emotion", {}).items()}
    return {"dominant_emotion": result.get("dominant_emotion"), "emotion_scores": scores}


async def analyze_frame(image_base64: str) -> AnalyzeFrameResponse:
    try:
        image_bytes = _decode_base64_image(image_base64)
    except ValueError as exc:
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error=str(exc))
    try:
        result = await asyncio.to_thread(_analyze_frame_sync, image_bytes)
    except ImportError as exc:
        logger.warning("DeepFace unavailable: %s", exc)
        return AnalyzeFrameResponse(
            dominant_emotion=None, emotion_scores={}, ok=False,
            error='Visual emotion dependencies are not installed. For Python 3.11 run: pip install deepface tf-keras pillow numpy "opencv-python<5"',
        )
    except ValueError as exc:
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error=str(exc))
    except DeepFrameRuntimeError:
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error="Visual emotion analysis is temporarily unavailable.")
    except Exception:
        logger.exception("Unexpected failure in visual emotion analysis")
        return AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error="Visual emotion analysis is temporarily unavailable.")
    return AnalyzeFrameResponse(dominant_emotion=result["dominant_emotion"], emotion_scores=result["emotion_scores"], ok=True, error=None)

def _pending_score_items_sync(user_id: int) -> List[dict]:
    with get_conn() as conn:
        sessions = conn.execute(
            """
            SELECT session_id, scale, current_item, status
            FROM screening_sessions
            WHERE user_id=?
              AND status IN ('active', 'paused')
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
                WHERE session_id=? AND item_id=?
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


async def get_pending_score_items(user_id: int) -> List[dict]:
    return await run_db(_pending_score_items_sync, user_id)
# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------

async def build_llm_context(
    user_id, current_message, preferred_language, sleep_hours, deepface_emotion,
    active_question=None, conversation_id=None,
) -> List[dict]:
    recent = await get_recent_messages(user_id, conversation_id=conversation_id)
    summaries = await get_recent_weekly_summaries(user_id)
    snapshot = await get_latest_assessment_snapshot(user_id)
    pending = await get_pending_score_items(user_id)
    trends = await compute_four_week_trends(user_id)

    context_lines = ["[BACKEND CONTEXT - not user authored]"]
    if preferred_language:
        context_lines.append(f"preferred_language: {preferred_language}")
    if active_question is not None:
        context_lines.append(
            f"current_follow_up_target: scale={active_question['scale']} "
            f"item={active_question['item_id']} -> previous question: "
            f"\"{active_question['question_text']}\". Interpret the user's reply as the "
            "answer when it provides frequency; otherwise continue naturally."
        )
    if summaries:
        context_lines.append("weekly_summaries (oldest to newest):")
        for s in summaries:
            context_lines.append(f"- {s['week_start']} to {s['week_end']}: {s['summary_text']}")
    if any(snapshot[k] for k in snapshot):
        context_lines.append(f"latest_structured_assessment_snapshot: {snapshot}")
    if any(trends.values()):
        context_lines.append("four_week_screening_trends (backend analysis, not a diagnosis): "
                             f"{ {k: v for k, v in trends.items() if v} }")
    if pending:
        context_lines.append(
            "items_with_evidence_but_no_frequency_yet (ask naturally about the frequency of "
            f"AT MOST one item when it fits; never run the full questionnaire): {pending}"
        )
    if sleep_hours is not None:
        context_lines.append(f"sleep_hours_reported_this_turn: {sleep_hours}")
    if deepface_emotion:
        context_lines.append(
            "optional_visual_emotion_signal (context only, NOT a symptom or diagnosis): "
            + deepface_emotion
        )
    messages: List[dict] = [{"role": "system", "content": "\n".join(context_lines)}]
    for row in recent:
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": current_message})
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


# ---------------------------------------------------------------------------
# Voice transcription (raw audio never retained)

import base64


async def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
) -> str:
    try:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        response = await asyncio.to_thread(
            get_gemini_client().models.generate_content,
            model=GEMINI_MODEL,
            contents=[
                {
                    "text": (
                        "Transcribe this audio accurately. "
                        "Preserve the speaker's original language and mixed-language "
                        "speech such as Hindi, Hinglish, Urdu, Kashmiri, Dogri, or English. "
                        "Return only the transcription, without commentary."
                    )
                },
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": audio_b64,
                    }
                },
            ],
        )

    except Exception as exc:
        logger.exception(
            "Gemini voice transcription failed: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Voice transcription service is temporarily unavailable.",
        ) from exc

    text = (response.text or "").strip()

    if not text:
        raise HTTPException(
            status_code=502,
            detail="Voice transcription returned no text.",
        )

    return text
    
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "gaash-bot"
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

@app.get("/profile")
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
            pass

    return result


@app.put("/profile")
async def update_profile(
    data: ProfileUpdateRequest,
    user_id: int = Depends(get_current_user_id),
):
    updates = []
    values = []

    if data.display_name is not None:
        updates.append("display_name=?")
        values.append(data.display_name)

    if data.preferred_language is not None:
        if data.preferred_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported language."
            )

        updates.append("preferred_language=?")
        values.append(data.preferred_language)

    if data.theme is not None:
        if data.theme not in SUPPORTED_THEMES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported theme."
            )

        updates.append("theme=?")
        values.append(data.theme)

    if data.notification_prefs is not None:
        updates.append("notification_prefs=?")
        values.append(json.dumps(data.notification_prefs))

    if updates:
        values.append(user_id)

        with get_conn() as conn:
            conn.execute(
                f"""
                UPDATE users
                SET {", ".join(updates)}
                WHERE id=?
                """,
                values,
            )
            conn.commit()

    return {
        "status": "updated"
    }

# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: int = Depends(get_current_user_id),
):
    conversation_id = request.conversation_id

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
        current_message=request.user_message,
        conversation_id=conversation_id,
        preferred_language=request.preferred_language,
        sleep_hours=request.sleep_hours,
        deepface_emotion=request.deepface_emotion,
    )

    try:
        analysis = await run_nlp_analysis(context)
    except LLMServiceError as exc:
        logger.exception("LLM analysis failed")
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    await save_message(
        user_id,
        "user",
        request.user_message,
        conversation_id,
    )

    # Save structured screening information.
    for item in validate_symptom_items(
        "PHQ-9",
        analysis.phq9_symptoms,
    ):
        if item.score is not None:
            session = await get_or_start_screening_session(
                user_id,
                "PHQ-9",
                conversation_id,
            )

            await record_session_item(
                user_id,
                session["session_id"],
                item.item_id,
                item.score,
                item.evidence,
            )

    for item in validate_symptom_items(
        "GAD-7",
        analysis.gad7_symptoms,
    ):
        if item.score is not None:
            session = await get_or_start_screening_session(
                user_id,
                "GAD-7",
                conversation_id,
            )

            await record_session_item(
                user_id,
                session["session_id"],
                item.item_id,
                item.score,
                item.evidence,
            )

    for item in validate_symptom_items(
        "PSS-10",
        analysis.pss10_symptoms,
    ):
        if item.score is not None:
            session = await get_or_start_screening_session(
                user_id,
                "PSS-10",
                conversation_id,
            )

            await record_session_item(
                user_id,
                session["session_id"],
                item.item_id,
                item.score,
                item.evidence,
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

    await save_message(
        user_id,
        "assistant",
        reply,
        conversation_id,
    )

    totals = await get_latest_finalized_totals(user_id)

    trajectory = await compute_trajectory(
        user_id,
        totals,
    )

    risk_details = get_risk_details(
        phq9=totals.get("PHQ-9"),
        gad7=totals.get("GAD-7"),
        pss10=totals.get("PSS-10"),
        sleep_hours=request.sleep_hours,
        emotion=request.deepface_emotion,
        trajectory=trajectory,
        emergency=emergency,
    )

    pending_raw = await get_pending_score_items(user_id)

    pending = {
        "PHQ-9": [],
        "GAD-7": [],
        "PSS-10": [],
    }

    for item in pending_raw:
        scale = item.get("scale")
        item_id = item.get("item_id")

        if scale in pending and item_id is not None:
            pending[scale].append(int(item_id))
            
    analytics = ChatAnalytics(
        detected_language=analysis.detected_language,
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

    if request.deepface_emotion:
        emotion = EmotionResponse(
            primary=request.deepface_emotion,
        )

    risk = RiskResponse(
        level=risk_details["risk_category"],
        requires_escalation=emergency,
    )

    return ChatResponse(
        message_id=str(uuid.uuid4()),
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
    )


# ---------------------------------------------------------------------------
# CONVERSATIONS
# ---------------------------------------------------------------------------

@app.post("/conversations")
async def create_new_conversation(
    user_id: int = Depends(get_current_user_id),
):
    conversation_id = await create_conversation(user_id)

    return {
        "conversation_id": conversation_id
    }


@app.get("/conversations")
async def get_conversations(
    user_id: int = Depends(get_current_user_id),
):
    rows = await run_db(
        _own_conversations_sync,
        user_id,
    )

    return {
        "conversations": [
            dict(row)
            for row in rows
        ]
    }


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user_id: int = Depends(get_current_user_id),
):
    conversation = await verify_conversation(
        user_id,
        conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    rows = await run_db(
        _conversation_messages_sync,
        user_id,
        conversation_id,
    )

    return {
        "conversation_id": conversation_id,
        "messages": [
            dict(row)
            for row in rows
        ]
    }


# ---------------------------------------------------------------------------
# ASSESSMENTS
# ---------------------------------------------------------------------------

@app.get("/assessments/latest")
async def latest_assessment(
    user_id: int = Depends(get_current_user_id),
):
    return await get_latest_assessment_snapshot(user_id)


@app.get("/assessments/totals")
async def assessment_totals(
    user_id: int = Depends(get_current_user_id),
):
    return await get_latest_finalized_totals(user_id)


@app.get("/assessments/pending")
async def pending_assessments(
    user_id: int = Depends(get_current_user_id),
):
    return {
        "pending": await get_pending_score_items(user_id)
    }


@app.get("/assessments/{scale}/history")
async def assessment_history(
    scale: str,
    user_id: int = Depends(get_current_user_id),
):
    scale = scale.upper()

    if scale not in _SCALE_ITEM_COUNT:
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment scale."
        )

    rows = await measurement_history(
        user_id,
        scale,
    )

    return {
        "scale": scale,
        "history": [
            dict(row)
            for row in rows
        ]
    }


@app.post("/assessments/{scale}/start")
async def start_assessment(
    scale: str,
    conversation_id: Optional[str] = None,
    user_id: int = Depends(get_current_user_id),
):
    scale = scale.upper()

    if scale not in _SCALE_ITEM_COUNT:
        raise HTTPException(
            status_code=400,
            detail="Invalid assessment scale."
        )

    session = await get_or_start_screening_session(
        user_id,
        scale,
        conversation_id,
    )

    return dict(session)


# ---------------------------------------------------------------------------
# SCREENING SESSION CONTROL
# ---------------------------------------------------------------------------

@app.post("/screening/{session_id}/pause")
async def pause_screening(
    session_id: str,
    user_id: int = Depends(get_current_user_id),
):
    success = await pause_session(
        user_id,
        session_id,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return {
        "status": "paused",
        "session_id": session_id,
    }


@app.post("/screening/{session_id}/cancel")
async def cancel_screening(
    session_id: str,
    user_id: int = Depends(get_current_user_id),
):
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


@app.get("/screening/{session_id}")
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

@app.get("/analytics")
async def get_analytics(
    user_id: int = Depends(get_current_user_id),
):
    totals = await get_latest_finalized_totals(user_id)
    weekly = await get_latest_weekly_aggregate(user_id)
    trends = await compute_four_week_trends(user_id)
    trajectory = await compute_trajectory(
        user_id,
        totals,
    )

    return {
        "screening_totals": totals,
        "weekly_averages": weekly.model_dump(),
        "four_week_trends": trends,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# WELLBEING REPORT
# ---------------------------------------------------------------------------

@app.get("/report")
async def wellbeing_report(
    user_id: int = Depends(get_current_user_id),
):
    totals = await get_latest_finalized_totals(user_id)
    weekly = await get_latest_weekly_aggregate(user_id)
    trends = await compute_four_week_trends(user_id)
    trajectory = await compute_trajectory(
        user_id,
        totals,
    )
    pending = await get_pending_score_items(user_id)

    return {
        "screening_totals": totals,
        "weekly_averages": weekly.model_dump(),
        "four_week_trends": trends,
        "trajectory": trajectory,
        "pending_score_items": pending,
        "safety_status": "screening results are not a diagnosis",
    }


# ---------------------------------------------------------------------------
# WEEKLY SUMMARY
# ---------------------------------------------------------------------------

@app.get("/weekly-summary")
async def weekly_summary(
    week_start: Optional[str] = None,
    week_end: Optional[str] = None,
    user_id: int = Depends(get_current_user_id),
):
    start, end = _resolve_week(
        week_start,
        week_end,
    )

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

@app.get("/recommendations")
async def recommendations(
    user_id: int = Depends(get_current_user_id),
):
    rows = await list_recommendations(user_id)

    return {
        "recommendations": [
            dict(row)
            for row in rows
        ]
    }


# ---------------------------------------------------------------------------
# VOICE
# ---------------------------------------------------------------------------

@app.post("/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
):
    transcript = await transcribe_audio(audio)

    return {
        "transcript": transcript
    }


# ---------------------------------------------------------------------------
# EMOTION
# ---------------------------------------------------------------------------

@app.post("/emotion/analyze")
async def emotion_analysis(
    request: AnalyzeFrameRequest,
    user_id: int = Depends(get_current_user_id),
):
    return await analyze_frame(
        request.image_base64
    )
    
@app.post("/screening/{session_id}/answer")
async def submit_screening_answer(
    session_id: str,
    request: AssessmentAnswerRequest,
    user_id: int = Depends(get_current_user_id),
):
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

    return result

init_gaash_tables()

if __name__ == "__main__":
    uvicorn.run("bot:app", host=HOST, port=PORT, reload=True)