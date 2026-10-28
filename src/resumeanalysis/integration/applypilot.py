"""One local ApplyPilot client for recommendations, feedback and letter drafts."""

from typing import Protocol
from urllib.parse import urlsplit

from resumeanalysis.cover_letters.service import CoverLetterService
from resumeanalysis.recommendation.adapters import cover_letter_request
from resumeanalysis.recommendation.schema import (
    Identifier,
    Salary,
    StrictModel,
    StructuredProfile,
    StructuredVacancy,
)
from resumeanalysis.schemas import Vacancy


class VacancyLink(StrictModel):
    url: str
    source_domain: Identifier
    vacancy_id: Identifier


class ResolvedVacancy(StrictModel):
    facts: Vacancy
    salary: Salary | None = None


class VacancyContentResolver(Protocol):
    def resolve(self, link: VacancyLink) -> ResolvedVacancy: ...


def adapt_applypilot_link(link, resolver: VacancyContentResolver) -> StructuredVacancy:
    """Accept ApplyPilot's parser dataclass or ORM Link, without importing its code."""
    domain = getattr(link, "source_domain", None) or getattr(link, "domain", None)
    value = VacancyLink(url=link.url, source_domain=domain, vacancy_id=str(link.vacancy_id))
    url = urlsplit(value.url)
    if url.scheme not in {"https", "http"} or url.netloc != value.source_domain:
        raise ValueError("Vacancy link URL and source domain disagree")
    content = resolver.resolve(value)
    if not isinstance(content, ResolvedVacancy):
        content = ResolvedVacancy.model_validate(content, strict=True)
    return StructuredVacancy(
        vacancy_id=f"{value.source_domain}:{value.vacancy_id}",
        facts=content.facts,
        salary=content.salary,
        source_domain=value.source_domain,
        source_url=value.url,
    )


def adapt_applypilot_description(link, description, *, verified_salary=None) -> StructuredVacancy:
    """Map the unmerged ingestion branch's VacancyDescDTO/ORM without trusting enrichment.

    The branch provides salary as free text and no verified remote/experience
    fields. The host must supply parsed/verified salary explicitly. HTML alone
    is not silently passed off as extracted plain text.
    """
    plain = getattr(description, "description_plain", None)
    if not isinstance(plain, str) or not plain.strip():
        raise ValueError("ApplyPilot description requires host-extracted plain text")
    if verified_salary is not None:
        verified_salary = Salary.model_validate(
            verified_salary.model_dump()
            if isinstance(verified_salary, Salary)
            else verified_salary,
            strict=True,
        )
    resolved = ResolvedVacancy(
        facts=Vacancy(
            title=description.title, description=plain, location=description.location or None
        ),
        salary=verified_salary,
    )

    class DescriptionResolver:
        def resolve(self, value):
            return resolved

    return adapt_applypilot_link(link, DescriptionResolver())


class RecommendationProfileProvider(Protocol):
    def get_profile(self, user_id: str) -> StructuredProfile: ...


class RecommendationVacancyProvider(Protocol):
    def list_vacancies(self, user_id: str) -> list[StructuredVacancy]: ...


class ApplyPilotClient:
    def __init__(
        self,
        engine,
        profiles: RecommendationProfileProvider,
        vacancies: RecommendationVacancyProvider,
        *,
        letters: CoverLetterService | None = None,
    ):
        self.engine, self.profiles, self.vacancies = engine, profiles, vacancies
        self.letters = letters

    def recommend(self, user_id, limit=20):
        return self.engine.recommend(
            user_id,
            self.profiles.get_profile(user_id),
            self.vacancies.list_vacancies(user_id),
            limit,
        )

    def record_feedback(self, user_id, vacancy_id, action, reason=None, **kwargs):
        return self.engine.record_feedback(user_id, vacancy_id, action, reason, **kwargs)

    def cover_letter_input(self, user_id, vacancy_id, *, resume_id, language="en"):
        profile, vacancy = self._selected_vacancy(user_id, vacancy_id)
        return cover_letter_request(profile, vacancy, resume_id=resume_id, language=language)

    async def draft_cover_letter(self, user_id, vacancy_id, *, language="en") -> str:
        if self.letters is None:
            raise ValueError("Configure a local cover-letter generator before requesting a draft")
        profile, vacancy = self._selected_vacancy(user_id, vacancy_id)
        return await self.letters.draft(profile, vacancy, language=language)

    def _selected_vacancy(self, user_id, vacancy_id):
        profile = self.profiles.get_profile(user_id)
        if profile.user_id != user_id:
            raise ValueError("Profile belongs to another user")
        vacancy = next(
            (v for v in self.vacancies.list_vacancies(user_id) if v.vacancy_id == vacancy_id), None
        )
        if vacancy is None:
            raise ValueError("Vacancy is unavailable to this user")
        return profile, vacancy
