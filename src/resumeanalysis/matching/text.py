"""Shared evidence text; job titles do not establish competencies."""

from resumeanalysis.schemas import Profile, Vacancy

LEVELS = {k: n for n, k in enumerate(["intern", "junior", "middle", "senior", "lead"])}


def profile_text(profile: Profile) -> str:
    return "\n".join(
        [
            profile.resume_text,
            "Skills: " + ", ".join(profile.skills),
            *profile.projects,
            *profile.education,
            "Explicitly not known: " + ", ".join(profile.denied_skills),
        ]
    )


def vacancy_text(vacancy: Vacancy) -> str:
    # Job titles are intentionally excluded: compare evidence of competencies.
    return "\n".join(
        [
            vacancy.description,
            "Required: " + ", ".join(vacancy.required_skills),
            "Optional: " + ", ".join(vacancy.optional_skills),
        ]
    )
