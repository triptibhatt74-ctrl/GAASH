from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Internal-only NLP/search representation; visible text is never changed."""

    visible_text: str
    normalized_for_context: str
    code_switching_detected: bool
    romanized_indic_hint: bool
    matched_aliases: tuple[str, ...]


# Conservative aliases are used only for model/search context.  They are not
# emotion labels and must not be used as evidence of distress.
_COMMON_ALIASES = {
    "pls": "please",
    "plz": "please",
    "thik": "theek",
    "theek": "theek",
    "nhi": "nahi",
    "nai": "nahi",
    "mujhe": "mujhe",
    "kyu": "kyun",
}
_ROMANIZED_INDIC_MARKERS = frozenset({"hai", "nahi", "mera", "meri", "mujhe", "kya", "kyun", "aaj", "kal", "bahut"})


def normalize_for_understanding(text: str) -> NormalizationResult:
    """Produce a bounded, non-visible normalization for NLP and retrieval.

    The original message is always retained exactly for display/storage.  This
    routine performs no distress classification and does not rewrite messages.
    """

    visible = text if isinstance(text, str) else ""
    canonical = unicodedata.normalize("NFC", visible)
    compact = re.sub(r"\s+", " ", canonical).strip()
    tokens = re.findall(r"[\w']+", compact.casefold(), flags=re.UNICODE)
    aliases = tuple(sorted({token for token in tokens if token in _COMMON_ALIASES}))
    normalized_tokens = [_COMMON_ALIASES.get(token, token) for token in tokens]
    # The context form deliberately preserves order and only substitutes known
    # aliases; all original user-facing rendering continues to use ``visible``.
    normalized = " ".join(normalized_tokens)
    has_latin = bool(re.search(r"[A-Za-z]", compact))
    has_indic_script = bool(re.search(r"[\u0900-\u0D7F\u0600-\u06FF]", compact))
    return NormalizationResult(
        visible_text=visible,
        normalized_for_context=normalized,
        code_switching_detected=has_latin and has_indic_script,
        romanized_indic_hint=has_latin and bool(set(tokens) & _ROMANIZED_INDIC_MARKERS),
        matched_aliases=aliases,
    )

