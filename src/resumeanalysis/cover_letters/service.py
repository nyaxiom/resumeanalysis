"""Explicit letter drafting; qualification and personal interest are separate."""

from resumeanalysis.cover_letters.generator import CoverLetterGenerator
from resumeanalysis.recommendation.adapters import cover_letter_request
from resumeanalysis.recommendation.schema import StructuredProfile, StructuredVacancy


class CoverLetterService:
    def __init__(self, generator: CoverLetterGenerator):
        self.generator = generator

    async def draft(
        self, profile: StructuredProfile, vacancy: StructuredVacancy, *, language="en"
    ) -> str:
        profile = StructuredProfile.model_validate(profile.model_dump(), strict=True)
        vacancy = StructuredVacancy.model_validate(vacancy.model_dump(), strict=True)
        request = cover_letter_request(profile, vacancy, resume_id=1, language=language)
        return await self.generator.generate(
            request.vacancy,
            request.resumes[0],
            language=request.language,
            context=request.candidate_context,
        )
