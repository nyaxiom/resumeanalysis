"""The three HTTP contracts. Scores are uncalibrated interest scores."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Description = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)
]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class VacancyText(Input):
    title: Title
    description: Description

    @property
    def text(self) -> str:
        # One shared representation for both training and ranking.
        return f"{self.title}\n\n{self.description}"


class Vacancy(VacancyText):
    vacancy_id: Identifier


class RankRequest(Input):
    user_id: Identifier
    vacancies: list[Vacancy] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({v.vacancy_id for v in self.vacancies}) != len(self.vacancies):
            raise ValueError("Vacancy IDs must be unique within a rank request")
        return self


class RankedVacancy(Vacancy):
    score: float = Field(ge=0, le=1, description="Uncalibrated interest score, not a probability")


class RankResponse(BaseModel):
    personalized: bool
    vacancies: list[RankedVacancy]


class FeedbackRequest(Input):
    user_id: Identifier
    vacancy_id: Identifier
    vacancy: VacancyText | None = None
    liked: bool


class FeedbackResponse(BaseModel):
    updated: bool
    revision: int
    feedback_count: int
    personalized: bool


class CoverLetterRequest(Input):
    resume_text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=30000)
    ]
    vacancy_title: Title
    vacancy_description: Description
    instructions: Annotated[str, StringConstraints(max_length=2000)] | None = None
    language: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)] | None
    ) = None


class CoverLetterResponse(BaseModel):
    cover_letter: str
