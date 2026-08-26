from __future__ import annotations

from schemas.languages import LanguageRegistryEntry, StateUtLanguageCoverage

from .geography import STATE_UTS


_FRONTEND_UI_CODES = frozenset({"en", "hi", "ur", "ks"})

# Registry entries are capability declarations, not translation claims.  Only
# the existing reviewed interface languages are marked UI-supported.
LANGUAGE_REGISTRY: tuple[LanguageRegistryEntry, ...] = (
    LanguageRegistryEntry(code="en", name="English", native_name="English", script="Latn", direction="ltr", states_ut=[item.code for item in STATE_UTS], ui_supported=True, nlp_supported=True, voice_supported=True, human_review_status="reviewed-existing-interface"),
    LanguageRegistryEntry(code="hi", name="Hindi", native_name="हिन्दी", script="Deva", direction="ltr", states_ut=["IN-BR", "IN-CT", "IN-HR", "IN-HP", "IN-JH", "IN-MP", "IN-RJ", "IN-UP", "IN-UT", "IN-CH", "IN-DL"], ui_supported=True, nlp_supported=True, voice_supported=True, human_review_status="existing-interface-review-required-for-safety-content"),
    LanguageRegistryEntry(code="ur", name="Urdu", native_name="اردو", script="Arab", direction="rtl", states_ut=["IN-JK"], ui_supported=True, nlp_supported=True, voice_supported=False, human_review_status="existing-interface-review-required-for-safety-content"),
    LanguageRegistryEntry(code="ks", name="Kashmiri", native_name="کٲشُر", script="Arab", direction="rtl", states_ut=["IN-JK"], ui_supported=True, nlp_supported=True, voice_supported=False, human_review_status="existing-interface-review-required-for-safety-content"),
    LanguageRegistryEntry(code="doi", name="Dogri", native_name="डोगरी", script="Deva", direction="ltr", states_ut=["IN-JK"], ui_supported=False, nlp_supported=True, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="as", name="Assamese", native_name="অসমীয়া", script="Beng", direction="ltr", states_ut=["IN-AS"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="bn", name="Bengali", native_name="বাংলা", script="Beng", direction="ltr", states_ut=["IN-TR", "IN-WB"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="gu", name="Gujarati", native_name="ગુજરાતી", script="Gujr", direction="ltr", states_ut=["IN-GJ", "IN-DH"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="kn", name="Kannada", native_name="ಕನ್ನಡ", script="Knda", direction="ltr", states_ut=["IN-KA"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="kok", name="Konkani", native_name="कोंकणी", script="Deva", direction="ltr", states_ut=["IN-GA"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="ml", name="Malayalam", native_name="മലയാളം", script="Mlym", direction="ltr", states_ut=["IN-KL", "IN-LD"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="mni", name="Manipuri", native_name="মৈতৈলোন্", script="Mtei", direction="ltr", states_ut=["IN-MN"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="mr", name="Marathi", native_name="मराठी", script="Deva", direction="ltr", states_ut=["IN-MH"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="ne", name="Nepali", native_name="नेपाली", script="Deva", direction="ltr", states_ut=["IN-SK"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="or", name="Odia", native_name="ଓଡ଼ିଆ", script="Orya", direction="ltr", states_ut=["IN-OD"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="pa", name="Punjabi", native_name="ਪੰਜਾਬੀ", script="Guru", direction="ltr", states_ut=["IN-PB"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="ta", name="Tamil", native_name="தமிழ்", script="Taml", direction="ltr", states_ut=["IN-TN", "IN-PY"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
    LanguageRegistryEntry(code="te", name="Telugu", native_name="తెలుగు", script="Telu", direction="ltr", states_ut=["IN-AP", "IN-TS"], ui_supported=False, nlp_supported=False, voice_supported=False, human_review_status="not-yet-reviewed"),
)

_PRIMARY_LANGUAGE_BY_STATE = {
    "IN-AP": "te", "IN-AR": "en", "IN-AS": "as", "IN-BR": "hi", "IN-CT": "hi", "IN-GA": "kok",
    "IN-GJ": "gu", "IN-HR": "hi", "IN-HP": "hi", "IN-JH": "hi", "IN-KA": "kn", "IN-KL": "ml",
    "IN-MP": "hi", "IN-MH": "mr", "IN-MN": "mni", "IN-ML": "en", "IN-MZ": "en", "IN-NL": "en",
    "IN-OD": "or", "IN-PB": "pa", "IN-RJ": "hi", "IN-SK": "ne", "IN-TN": "ta", "IN-TS": "te",
    "IN-TR": "bn", "IN-UP": "hi", "IN-UT": "hi", "IN-WB": "bn", "IN-AN": "en", "IN-CH": "hi",
    "IN-DH": "gu", "IN-DL": "hi", "IN-JK": "ks", "IN-LA": "en", "IN-LD": "ml", "IN-PY": "ta",
}


def language_registry_payload() -> tuple[list[LanguageRegistryEntry], list[StateUtLanguageCoverage]]:
    """Return immutable registry data as API-ready Pydantic models."""

    coverage = [
        StateUtLanguageCoverage(
            code=state.code,
            name=state.name,
            kind=state.kind,
            primary_language_code=_PRIMARY_LANGUAGE_BY_STATE[state.code],
            # English is the existing reviewed UI fallback for every State/UT;
            # this does not imply all primary regional languages are translated.
            available_ui_language_codes=sorted({"en", *([_PRIMARY_LANGUAGE_BY_STATE[state.code]] if _PRIMARY_LANGUAGE_BY_STATE[state.code] in _FRONTEND_UI_CODES else [])}),
        )
        for state in STATE_UTS
    ]
    return list(LANGUAGE_REGISTRY), coverage


def registered_language_codes() -> set[str]:
    return {entry.code for entry in LANGUAGE_REGISTRY}
