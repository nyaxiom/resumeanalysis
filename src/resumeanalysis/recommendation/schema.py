"""Versioned recommendation contracts, separate from legacy qualification records."""

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from resumeanalysis.schemas import Profile, Vacancy

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def fingerprint(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class Salary(StrictModel):
    minimum: float | None = Field(default=None, ge=0)
    maximum: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    period: Literal["hour", "month", "year"] | None = None
    gross: bool | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Salary minimum cannot exceed maximum")
        return self


class HardConstraints(StrictModel):
    remote_only: bool = False
    locations: list[Identifier] = Field(default_factory=list, max_length=100)
    salary: Salary | None = None


class Preferences(StrictModel):
    preferred_roles: list[Identifier] = Field(default_factory=list, max_length=100)
    remote: bool | None = None
    locations: list[Identifier] = Field(default_factory=list, max_length=100)
    salary: Salary | None = None
    hard: HardConstraints = Field(default_factory=HardConstraints)


class SourceDocument(StrictModel):
    source_id: Identifier
    kind: Literal["text", "structured", "txt", "pdf", "docx", "user_edit"]
    text: str = Field(max_length=200_000)
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Character offsets refer to extracted text; documents are never modified by edits.
    pages: list[dict[str, int]] = Field(default_factory=list)


class EvidenceReference(StrictModel):
    source_id: Identifier
    quote: str = Field(min_length=1, max_length=4000)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_interval(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Evidence requires both offsets or neither")
        if self.start is not None and self.end <= self.start:
            raise ValueError("Evidence interval must be nonempty")
        return self


class SkillExperience(StrictModel):
    skill: Identifier
    years: float | None = Field(default=None, ge=0, le=80)


class ProfileData(StrictModel):
    facts: Profile = Field(default_factory=Profile)
    skill_experience: list[SkillExperience] = Field(default_factory=list, max_length=200)
    preferences: Preferences = Field(default_factory=Preferences)

    @field_validator("facts", mode="before")
    @classmethod
    def strict_facts(cls, value):
        return Profile.model_validate(
            value.model_dump() if isinstance(value, Profile) else value, strict=True
        )

    @model_validator(mode="after")
    def separate_preferences(self):
        if self.facts.remote_only or self.facts.allowed_locations:
            raise ValueError("Use preferences.hard for remote/location constraints")
        return self


class StructuredProfile(ProfileData):
    schema_version: Literal["profile-v1"] = "profile-v1"
    user_id: Identifier
    profile_id: Identifier = "primary"
    version: int = Field(default=1, ge=1)
    review_status: Literal["user_supplied", "needs_review", "confirmed"] = "user_supplied"
    sources: list[SourceDocument] = Field(default_factory=list, max_length=100)
    evidence: dict[str, list[EvidenceReference]] = Field(default_factory=dict)


class StructuredVacancy(StrictModel):
    schema_version: Literal["vacancy-v1"] = "vacancy-v1"
    vacancy_id: Identifier
    facts: Vacancy = Field(default_factory=Vacancy)
    salary: Salary | None = None
    source_url: str | None = Field(default=None, max_length=4000)
    source_domain: Identifier | None = None

    @field_validator("facts", mode="before")
    @classmethod
    def strict_facts(cls, value):
        return Vacancy.model_validate(
            value.model_dump() if isinstance(value, Vacancy) else value, strict=True
        )


class ConstraintStatus(StrictModel):
    status: Literal["eligible", "unknown", "blocked"]
    conflicts: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class QualificationEvidence(StrictModel):
    score: float | None = Field(default=None, ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    unknown_skills: list[str] = Field(default_factory=list)
    contradicted_skills: list[str] = Field(default_factory=list)
    experience_gap_years: float | None = None
    explanation: str
    alternative_requirements: list[dict] = Field(default_factory=list)


class FeatureSnapshot(StrictModel):
    schema_version: str
    identity: str
    values: dict[str, float]
    profile_id: Identifier
    profile_version: int = Field(ge=1)
    profile_sha256: str
    vacancy_sha256: str
    content_score: float = Field(ge=0, le=1)


class Recommendation(StrictModel):
    schema_version: Literal["recommendation-v1"] = "recommendation-v1"
    recommendation_id: Identifier
    vacancy_id: Identifier
    qualification: QualificationEvidence
    preference_score: float = Field(ge=0, le=1)
    hard_constraints: ConstraintStatus
    ranking_score: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)
    exploration: bool = False
    explanation: str
    profile_version: int
    feature_schema_version: str
    model_version: str


class FeedbackEvent(StrictModel):
    schema_version: Literal["feedback-v1"] = "feedback-v1"
    sequence: int = Field(default=0, ge=0)
    user_id: Identifier
    vacancy_id: Identifier
    event_id: Identifier
    action: Literal["like", "dislike"]
    reason: str | None = Field(default=None, max_length=2000)
    timestamp: datetime
    recommendation_id: Identifier
    corrects_event_id: Identifier | None = None
    profile_version: int
    feature_schema_version: str
    model_version: str
    feature_snapshot: FeatureSnapshot
    prediction_before_update: float = Field(ge=0, le=1)
    # Kept separately because feedback can arrive after other updates.
    prediction_at_recommendation: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def timezone(self):
        if self.timestamp.tzinfo is None:
            raise ValueError("Feedback timestamps must be timezone-aware")
        return self
