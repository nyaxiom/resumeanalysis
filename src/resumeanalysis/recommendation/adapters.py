"""Explicit migrations and a factual bridge to the existing cover-letter contract."""

from resumeanalysis.cover_letters.schema import AnalysisRequest, Resume
from resumeanalysis.recommendation.schema import (
    HardConstraints,
    Preferences,
    StructuredProfile,
    StructuredVacancy,
)


def matching_facts(profile):
    """Skill durations do not imply total career duration; named skills remain explicit."""
    facts = profile.facts.model_copy(deep=True)
    facts.skills = list(
        dict.fromkeys([*facts.skills, *(s.skill for s in profile.skill_experience)])
    )
    return facts


def from_legacy_profile(user_id, profile, *, profile_id="primary"):
    preferred_roles = list(getattr(profile, "preferred_roles", []))
    facts = profile.model_dump(exclude={"preferred_roles"})
    hard = HardConstraints(
        remote_only=facts.pop("remote_only"), locations=facts.pop("allowed_locations")
    )
    return StructuredProfile(
        user_id=user_id,
        profile_id=profile_id,
        facts=facts,
        preferences=Preferences(preferred_roles=preferred_roles, hard=hard),
    )


def from_legacy_vacancy(vacancy_id, vacancy):
    return StructuredVacancy(vacancy_id=vacancy_id, facts=vacancy)


def cover_letter_request(profile, vacancy, *, resume_id, language="en"):
    if profile.review_status == "needs_review":
        raise ValueError("Confirm extracted profile facts before generating a cover letter")
    facts = matching_facts(profile)
    facts.remote_only = profile.preferences.hard.remote_only
    facts.allowed_locations = list(profile.preferences.hard.locations)
    context = [
        f"Explicitly stated experience with {item.skill}: {item.years:g} years."
        for item in profile.skill_experience
        if item.years is not None
    ]
    return AnalysisRequest(
        vacancy=vacancy.facts,
        resumes=[Resume(id=resume_id, profile=facts)],
        language=language,
        candidate_context=context,
    )
