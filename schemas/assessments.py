from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import ContractModel


AssessmentScale = Literal["PHQ-9", "GAD-7", "PSS-10"]


class AssessmentMeasurement(ContractModel):
    assessment_type: AssessmentScale
    total: int = Field(ge=0, le=40)
    completed_at: datetime

