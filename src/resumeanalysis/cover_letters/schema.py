"""Transport-independent ApplyPilot inputs; original matching records are unchanged."""

from typing import Literal

from pydantic import Field, model_validator

from resumeanalysis.schemas import InputModel, Profile, Vacancy


class Resume(InputModel):
    id: int = Field(ge=1, strict=True)
    profile: Profile


class AnalysisRequest(InputModel):
    vacancy: Vacancy
    resumes: list[Resume] = Field(default_factory=list, max_length=25)
    language: Literal["en", "ru"] = "en"
    candidate_context: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def distinct_resumes(self):
        if len({resume.id for resume in self.resumes}) != len(self.resumes):
            raise ValueError("Resume IDs must be unique")
        if any(not value.strip() or len(value) > 2000 for value in self.candidate_context):
            raise ValueError("Candidate context must contain nonempty bounded factual statements")
        return self
