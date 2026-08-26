from __future__ import annotations

from pydantic import Field

from .common import ContractModel


class LanguageRegistryEntry(ContractModel):
    code: str = Field(min_length=2, max_length=20)
    name: str = Field(min_length=1, max_length=120)
    native_name: str = Field(min_length=1, max_length=120)
    script: str = Field(min_length=3, max_length=8)
    direction: str = Field(pattern=r"^(ltr|rtl)$")
    states_ut: list[str] = Field(default_factory=list)
    ui_supported: bool = False
    nlp_supported: bool = False
    voice_supported: bool = False
    human_review_status: str = Field(min_length=1, max_length=120)


class StateUtLanguageCoverage(ContractModel):
    code: str = Field(pattern=r"^IN-[A-Z]{2}$")
    name: str = Field(min_length=1, max_length=160)
    kind: str = Field(pattern=r"^(state|union-territory)$")
    primary_language_code: str = Field(min_length=2, max_length=20)
    available_ui_language_codes: list[str] = Field(min_length=1)


class LanguageRegistryResponse(ContractModel):
    languages: list[LanguageRegistryEntry] = Field(default_factory=list)
    states_ut: list[StateUtLanguageCoverage] = Field(default_factory=list)

