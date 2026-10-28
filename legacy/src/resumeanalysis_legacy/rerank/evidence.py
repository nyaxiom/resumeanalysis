"""Legacy cross-encoder serialization; shared evidence lives in the runtime."""

from resumeanalysis.matching.evidence import (
    EvidenceSummary,
    build_evidence,
    has_candidate_evidence,
    preference_conflicts,
)
from resumeanalysis.matching.text import profile_text, vacancy_text
from resumeanalysis.schemas import Profile, Vacancy

SERIALIZATION_VERSION = "resume-vacancy-evidence-v1"
__all__ = [
    "EvidenceSummary",
    "build_evidence",
    "has_candidate_evidence",
    "preference_conflicts",
    "cross_encoder_texts",
    "SERIALIZATION_VERSION",
]


def cross_encoder_texts(profile: Profile, vacancy: Vacancy) -> tuple[str, str]:
    """Serialize both sides consistently for standalone and pipeline evaluation.

    Job titles and preferred roles do not establish competencies. Explicit
    numeric/level fields supplement the existing content serializers. Preferences
    are enforced independently instead of being left to semantic similarity.
    """
    resume, job = [profile_text(profile)], [vacancy_text(vacancy)]
    if profile.experience_years is not None:
        resume.append(f"Stated experience: {profile.experience_years:g} years")
    if profile.seniority is not None:
        resume.append(f"Stated seniority: {profile.seniority}")
    if vacancy.min_experience_years is not None:
        job.append(f"Minimum experience: {vacancy.min_experience_years:g} years")
    if vacancy.seniority is not None:
        job.append(f"Required seniority: {vacancy.seniority}")
    return "\n".join(resume), "\n".join(job)
