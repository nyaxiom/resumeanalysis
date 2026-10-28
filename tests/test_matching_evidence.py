"""Authored evidence-policy cases; these are not hiring relevance benchmarks."""

import pytest

from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis.matching.evidence import (
    build_evidence,
    has_candidate_evidence,
    preference_conflicts,
)
from resumeanalysis.schemas import Profile, Vacancy


def evidence(profile, vacancy, extractor=None):
    return build_evidence(
        Profile.model_validate(profile),
        Vacancy.model_validate(vacancy),
        extractor or create_extractor("v2"),
    )


@pytest.mark.parametrize(
    "resume,job",
    [
        ("I use Python and Docker.", "Required: Python. Optional: Java."),
        ("Использую Python и Docker.", "Требования: Python. Желательно: Java."),
        ("Опыт разработки: Python and Docker.", "Required: Python. Желательно: Java."),
    ],
)
def test_multilingual_confirmation_and_optional_unknown_do_not_block(resume, job):
    result = evidence({"resume_text": resume}, {"description": job})
    assert result.status == "eligible"
    assert result.detail.matched == {"python"}
    assert result.detail.unknown == {"java"}
    assert result.metadata["optional_skills"] == ["java"]


def test_unknown_mandatory_is_review_and_never_a_denial():
    result = evidence({"skills": ["Python"]}, {"required_skills": ["Docker"]})
    assert result.status == "needs_review"
    assert result.detail.unknown == {"docker"}
    assert not result.detail.contradicted
    assert "not proof of absence" in " ".join(result.review_reasons)


@pytest.mark.parametrize("text", ["I do not know Python.", "Не знаю Python."])
def test_explicit_negative_mandatory_evidence_blocks(text):
    result = evidence({"resume_text": text}, {"required_skills": ["Python"]})
    assert result.status == "blocked"
    assert result.detail.contradicted == {"python"}


@pytest.mark.parametrize(
    "profile",
    [
        {"resume_text": "Python. I do not know Python."},
        {"resume_text": "Использую Python. Не знаю Python."},
        {"resume_text": "I do not know Python.", "skills": ["Python"]},
        {"resume_text": "I use Python.", "denied_skills": ["Python"]},
        {"skills": ["Python"], "denied_skills": ["Python"]},
    ],
)
def test_conflicting_evidence_cannot_be_hidden_by_normalization_or_fields(profile):
    result = evidence(profile, {"required_skills": ["Python"]})
    assert result.status == "blocked"
    assert result.metadata["conflicted_skills"] == ["python"]
    assert not result.detail.matched
    assert not result.detail.contradicted  # Conflict does not establish a pure denial.
    assert result.detail.unknown == {"python"}
    assert "Conflicting positive and negative" in " ".join(result.blocked_reasons)


def test_satisfied_alternative_does_not_require_denied_other_option():
    result = evidence(
        {"skills": ["Python"], "denied_skills": ["Java"]},
        {"description": "Required: Python or Java."},
    )
    assert result.status == "eligible"
    assert result.detail.matched == {"python"}
    assert not result.detail.unknown
    assert not result.detail.required
    assert result.detail.alternative_groups[0]["status"] == "matched"


@pytest.mark.parametrize(
    "profile,status",
    [
        ({"skills": ["Docker"], "denied_skills": ["Python", "Java"]}, "blocked"),
        ({"skills": ["Docker"], "denied_skills": ["Python"]}, "needs_review"),
        ({"skills": ["Docker"]}, "needs_review"),
        ({"skills": ["Python"], "denied_skills": ["Python", "Java"]}, "blocked"),
    ],
)
def test_alternative_denied_unknown_and_conflicted_states(profile, status):
    result = evidence(profile, {"description": "Требования: Python или Java."})
    assert result.status == status
    assert len(result.detail.alternative_groups) == 1


def test_optional_and_negated_alternatives_do_not_gate():
    result = evidence(
        {"skills": ["Docker"], "denied_skills": ["Python", "Java"]},
        {"description": "Optional: Python or Java. Docker is not required."},
    )
    assert result.status == "eligible"
    assert not result.detail.required


@pytest.mark.parametrize(
    "profile,job,status",
    [
        ({"skills": ["Python"]}, {"min_experience_years": 3}, "needs_review"),
        ({"experience_years": 1}, {"min_experience_years": 3}, "blocked"),
        ({"experience_years": 3}, {"min_experience_years": 3}, "eligible"),
        ({"skills": ["Python"]}, {"seniority": "senior"}, "needs_review"),
        ({"seniority": "junior"}, {"seniority": "senior"}, "blocked"),
        ({"seniority": "lead"}, {"seniority": "senior"}, "eligible"),
    ],
)
def test_structured_experience_and_seniority(profile, job, status):
    result = evidence(profile, job)
    assert result.status == status
    assert result.metadata["experience"]["source"] == "explicit_structured_fields_only"


def test_bare_text_years_and_role_terms_are_diagnostics_not_hard_qualifications():
    result = evidence(
        {"resume_text": "I led a 10 year community project."},
        {"description": "Our company is 20 years old. Work with senior developers."},
    )
    assert result.status == "eligible"
    assert result.detail.experience_gap is None
    assert result.metadata["experience"]["candidate_years"] is None
    assert result.metadata["experience"]["unverified_text_vacancy_years"] == 20
    assert result.metadata["seniority"]["required"] is None


def test_remote_location_known_conflicts_and_unknowns():
    profile = {"skills": ["Python"], "remote_only": True, "allowed_locations": ["London"]}
    blocked = evidence(
        profile, {"required_skills": ["Python"], "remote": False, "location": "Paris"}
    )
    assert blocked.status == "blocked"
    assert len(blocked.blocked_reasons) == 2
    assert preference_conflicts(Profile(**profile), Vacancy(remote=False, location="Paris"))
    unknown = evidence(profile, {"required_skills": ["Python"]})
    assert unknown.status == "needs_review"
    assert len(unknown.review_reasons) == 2
    eligible = evidence(
        profile, {"required_skills": ["Python"], "remote": True, "location": "LONDON"}
    )
    assert eligible.status == "eligible"


@pytest.mark.parametrize(
    "profile,job",
    [
        ({}, {"description": "Required: Python."}),
        ({"skills": ["Python"]}, {}),
        ({"skills": ["Python"]}, {"title": "Senior Python developer"}),
        ({"skills": [" "]}, {"required_skills": ["Python"]}),
    ],
)
def test_empty_content_and_title_only_never_establish_usable_pair(profile, job):
    result = evidence(profile, job)
    assert result.status == "needs_review"
    assert result.metadata["usable_input"] is False


def test_extractor_called_once_per_side_and_remains_replaceable():
    class RecordingProvider:
        def __init__(self):
            self.provider = create_extractor("v2")
            self.normalizer = self.provider.normalizer
            self.calls = []

        def extract(self, text, *, vacancy=False):
            self.calls.append((text, vacancy))
            return self.provider.extract(text, vacancy=vacancy)

    provider = RecordingProvider()
    result = evidence(
        {"resume_text": "Python", "projects": ["Docker"]},
        {"description": "Required: Python"},
        provider,
    )
    assert result.status == "eligible"
    assert provider.calls == [("Python\nDocker", False), ("Required: Python", True)]


@pytest.mark.parametrize(
    "description",
    [
        "Python developer required. At least 5 years of experience required.",
        "Requirements:\nExperience: at least 5 years.",
        "Требования: опыт работы не менее 5 лет.",
        "Обязательно: 5 лет опыта.",
        "Required seniority: senior.",
        "Senior Python engineer required.",
        "Требуется старший разработчик Python.",
    ],
)
def test_clear_text_requirements_need_verification_when_fields_are_missing(description):
    result = evidence(
        {"resume_text": "Python developer with 1 year of experience.", "seniority": "junior"},
        {"description": description},
    )
    assert result.status == "needs_review"
    assert not result.blocked_reasons
    assert "verification" in " ".join(result.review_reasons)


@pytest.mark.parametrize(
    "job",
    [
        {"description": "At least 5 years of experience required.", "min_experience_years": 2},
        {"description": "Senior Python engineer required.", "seniority": "junior"},
    ],
)
def test_clear_text_requirement_cannot_silently_be_overridden_by_conflicting_fields(job):
    result = evidence({"skills": ["Python"], "experience_years": 3, "seniority": "middle"}, job)
    assert result.status == "needs_review"
    assert not result.blocked_reasons


@pytest.mark.parametrize(
    "job",
    [
        {"description": "At least 5 years of experience required.", "min_experience_years": 5},
        {"description": "Required seniority: senior.", "seniority": "senior"},
    ],
)
def test_verified_fields_matching_clear_text_requirements_remain_eligible(job):
    result = evidence({"skills": ["Python"], "experience_years": 6, "seniority": "senior"}, job)
    assert result.status == "eligible"


@pytest.mark.parametrize(
    "description",
    [
        "Our company is 20 years old. Work with senior developers.",
        "Optional: 5 years of experience. Seniority: senior preferred.",
        "5 years of experience is not required. Senior level is not required.",
        "Желательно: 5 лет опыта. Старший уровень не требуется.",
        "Requirements:\nPython\nAbout the company:\nOur company has 20 years of experience.",
        "Required: work with senior developers.",
    ],
)
def test_optional_qualifications_company_age_and_colleagues_never_become_requirement_cues(
    description,
):
    result = evidence({"skills": ["Python"]}, {"description": description})
    assert result.status == "eligible"
    assert result.metadata["unverified_text_requirements"]["experience_years"] == []
    assert result.metadata["unverified_text_requirements"]["seniority"] == []


@pytest.mark.parametrize(
    "description,remote",
    [
        ("Office only. Required: Python.", True),
        ("On-site work only.", True),
        ("No remote work.", True),
        ("Remote work is not available.", True),
        ("Только работа в офисе.", True),
        ("Удалённая работа не предусмотрена.", True),
        ("Fully remote role.", False),
        ("Полностью удалённая работа.", False),
        ("Fully remote. Office only.", None),
    ],
)
def test_conflicting_remote_statements_need_review(description, remote):
    result = evidence({"skills": ["Python"]}, {"description": description, "remote": remote})
    assert result.status == "needs_review"
    assert not result.blocked_reasons
    assert "Remote-work statements conflict" in " ".join(result.review_reasons)


@pytest.mark.parametrize(
    "description,remote",
    [
        ("Not an office only role.", True),
        ("Работа не только в офисе.", True),
        ("Not fully remote.", False),
        ("Не полностью удалённая работа.", False),
    ],
)
def test_negated_work_arrangement_phrase_is_not_an_affirmative_conflict(description, remote):
    result = evidence({"skills": ["Python"]}, {"description": description, "remote": remote})
    assert result.status == "eligible"


@pytest.mark.parametrize(
    "profile,expected",
    [
        ({}, False),
        ({"denied_skills": ["Python"]}, False),
        ({"resume_text": "  ", "skills": [" "]}, False),
        ({"skills": ["Python"]}, True),
        ({"experience_years": 0}, True),
        ({"seniority": "intern"}, True),
        ({"projects": ["Built an API"]}, True),
    ],
)
def test_cheap_positive_candidate_check(profile, expected):
    assert has_candidate_evidence(Profile.model_validate(profile)) is expected
