from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ScaleName = Literal["PHQ-9", "GAD-7", "PSS-10", "NONE"]


# ---------------------------------------------------------------------------
# Structured output returned by the LLM (client.beta.chat.completions.parse)
# ---------------------------------------------------------------------------

class SymptomItem(BaseModel):
    item_id: int
    score: Optional[int] = None
    evidence: str


class FunctionalImpairment(BaseModel):
    area: str
    severity: str
    evidence: str


class NLPAnalysis(BaseModel):
    detected_language: str
    phq9_symptoms: List[SymptomItem] = Field(default_factory=list)
    gad7_symptoms: List[SymptomItem] = Field(default_factory=list)
    pss10_symptoms: List[SymptomItem] = Field(default_factory=list)
    sleep_hours_reported: Optional[float] = None
    functional_impairments: List[FunctionalImpairment] = Field(default_factory=list)
    active_scale_triggered: ScaleName
    response_to_user: str
    emergency_flag: bool


# ---------------------------------------------------------------------------
# Backend-side validation layer (defense in depth on top of the LLM schema —
# section 4 of the build spec: "validate item IDs and score ranges after
# parsing as an additional backend safety layer")
# ---------------------------------------------------------------------------

_ITEM_ID_RANGES = {"PHQ-9": (1, 9), "GAD-7": (1, 7), "PSS-10": (1, 10)}
_SCORE_MAX = {"PHQ-9": 3, "GAD-7": 3, "PSS-10": 4}


def validate_symptom_items(scale: str, items: List[SymptomItem]) -> List[SymptomItem]:
    """Drop/clip anything the LLM returned outside the valid item/score
    range instead of trusting it blindly. Never raises on bad input from
    the model — a malformed item is dropped, not fatal to the request."""
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


# ---------------------------------------------------------------------------
# API request/response bodies
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=4000)
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


class ChatResponse(BaseModel):
    response_to_user: str
    analytics: ChatAnalytics
    risk_category: Optional[str] = None
    escalation_created: bool = False


class AnalyzeFrameRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)


class AnalyzeFrameResponse(BaseModel):
    dominant_emotion: Optional[str]
    emotion_scores: dict
    ok: bool
    error: Optional[str] = None


class FollowUpRequest(BaseModel):
    scheduled_for: str  # ISO-8601 datetime string
    note: Optional[str] = None
