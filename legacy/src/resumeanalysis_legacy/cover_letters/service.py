"""Relevance → deterministic resume selection → one optional local letter."""

import asyncio

from resumeanalysis.cover_letters.generator import CoverLetterError, CoverLetterGenerator
from resumeanalysis.cover_letters.schema import AnalysisRequest
from resumeanalysis.schemas import VacancyAnalysis
from resumeanalysis_legacy.inference.backends import MatchingBackend


class AnalysisService:
    def __init__(self, relevance: MatchingBackend, generator: CoverLetterGenerator):
        self.relevance, self.generator = relevance, generator

    async def analyze(self, request: AnalysisRequest) -> VacancyAnalysis:
        request = AnalysisRequest.model_validate(request.model_dump(), strict=True)
        if not request.resumes:
            return VacancyAnalysis(
                relevant=False,
                relevance=0,
                matched_skills=[],
                missing_skills=[],
                unknown_skills=[],
                contradicted_skills=[],
                experience_gap_years=None,
                explanation="No resumes were supplied.",
            )
        evaluated = []
        for resume in request.resumes:
            analysis = await asyncio.to_thread(
                self.relevance.predict, resume.profile, request.vacancy
            )
            analysis = VacancyAnalysis.model_validate(analysis.model_dump(), strict=True)
            evaluated.append((resume, analysis))
        # max preserves original input order for exact ties. Positive decisions
        # precede a higher raw score whose hard policy rejected the match.
        resume, analysis = max(evaluated, key=lambda pair: (pair[1].relevant, pair[1].relevance))
        result = analysis.model_copy(
            deep=True, update={"resume_id": resume.id, "cover_letter": None}
        )
        if result.relevant:
            try:
                letter = await self.generator.generate(
                    request.vacancy,
                    resume,
                    language=request.language,
                    context=request.candidate_context,
                )
                if not isinstance(letter, str) or not letter.strip():
                    raise CoverLetterError("Generator returned no plain-text draft")
                result.cover_letter = letter
                result.explanation += (
                    " The cover letter is a draft and requires factual review before use."
                )
            except CoverLetterError:
                result.explanation += " A cover-letter draft was not produced; generation failed or required verification."
        return result
