"""Public, JSON-compatible schemas. None means unknown, never zero or false."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Profile(InputModel):
    resume_text: str = ""
    skills: list[str] = Field(default_factory=list)
    denied_skills: list[str] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0, le=80)
    projects: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    seniority: Literal["intern", "junior", "middle", "senior", "lead"] | None = None
    remote_only: bool = False
    allowed_locations: list[str] = Field(default_factory=list)


class Vacancy(InputModel):
    title: str = ""
    description: str = ""
    required_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    min_experience_years: float | None = Field(default=None, ge=0, le=80)
    seniority: Literal["intern", "junior", "middle", "senior", "lead"] | None = None
    remote: bool | None = None
    location: str | None = None


class VacancyAnalysis(BaseModel):
    relevant: bool
    relevance: float = Field(ge=0, le=1)
    matched_skills: list[str]
    missing_skills: list[str]
    unknown_skills: list[str]
    contradicted_skills: list[str]
    experience_gap_years: float | None
    explanation: str
    preference_conflicts: list[str] = Field(default_factory=list)
    preference_unknowns: list[str] = Field(default_factory=list)
    summary: None = None
    resume_id: None = None
    cover_letter: None = None
