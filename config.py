"""
Gaash intelligence-layer configuration.

Reuses the existing project's DATABASE constant/convention (raw sqlite3,
file "gaash.db") rather than introducing a new database stack. All new
settings are environment-based; nothing is hard-coded.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "20"))

# --- Database (same sqlite file the existing auth API already uses) ---
DATABASE = os.environ.get("DATABASE_URL", "gaash.db")

# --- Conversation memory bounds ---
MAX_RECENT_MESSAGES = int(os.environ.get("MAX_RECENT_MESSAGES", "50"))
MAX_WEEKLY_SUMMARIES = int(os.environ.get("MAX_WEEKLY_SUMMARIES", "4"))

# --- Verified crisis pathway (must be supplied by the deploying team; the
# LLM and this backend never invent helpline numbers/contacts) ---
CRISIS_PATHWAY_LABEL = os.environ.get(
    "CRISIS_PATHWAY_LABEL", "the app's Crisis Support section"
)
CRISIS_PATHWAY_URL = os.environ.get("CRISIS_PATHWAY_URL", "")

# --- DeepFace ---
DEEPFACE_ENABLED = os.environ.get("DEEPFACE_ENABLED", "true").lower() == "true"

# NOTE: RISK_CONFIG lives in services/risk_service.py, clearly labeled as
# an unvalidated prototype — see the warning at the top of that file.
