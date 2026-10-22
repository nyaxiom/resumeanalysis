import pytest
from pydantic import ValidationError

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.features.feature_builder import FEATURE_NAMES, FeatureEngineering
from resumeanalysis.postprocessing.rules import Postprocessor
from resumeanalysis.schemas import Profile, Vacancy


def detail(profile=None, vacancy=None):
    return FeatureEngineering(LexicalSmokeEncoder()).build(
        profile or Profile(), vacancy or Vacancy()
    )


def test_three_skill_states():
    d = detail(
        Profile(skills=["Python"], denied_skills=["SQL"]),
        Vacancy(required_skills=["Python", "SQL", "Docker"]),
    )
    assert d.matched == {"python"}
    assert d.contradicted == {"sql"}
    assert d.unknown == {"docker"}
    features = dict(zip(FEATURE_NAMES, d.vector))
    assert features["required_match"] == pytest.approx(1 / 3)
    assert features["required_unknown"] == pytest.approx(1 / 3)
    assert features["required_denied"] == pytest.approx(1 / 3)


def test_no_title_skills_and_no_related_equivalence():
    d = detail(
        Profile(skills=["Postgres"]), Vacancy(title="Python SQL Engineer", required_skills=["SQL"])
    )
    assert d.matched == set()
    assert d.unknown == {"sql"}
    assert dict(zip(FEATURE_NAMES, d.vector))["related_support"] > 0


def test_unknown_experience_is_not_zero_experience():
    assert detail(Profile(), Vacancy(min_experience_years=2)).experience_gap is None
    assert detail(Profile(experience_years=0), Vacancy(min_experience_years=2)).experience_gap == 2


def test_conflict_between_structured_and_text_is_unknown():
    d = detail(
        Profile(resume_text="No experience with SQL", skills=["SQL"]),
        Vacancy(required_skills=["SQL"]),
    )
    assert d.unknown == {"sql"}


def test_remote_hard_preferences_and_unknown():
    p = Profile(remote_only=True)
    for remote in [False, None]:
        v = Vacancy(remote=remote)
        result = Postprocessor().process(0.99, detail(p, v), p, v)
        assert not result.relevant
        assert result.relevance == 0.99
    v = Vacancy(remote=True)
    assert Postprocessor().process(0.99, detail(p, v), p, v).relevant


def test_explanation_is_factual_and_fields_reserved():
    p, v = Profile(skills=["Python"]), Vacancy(required_skills=["SQL"])
    result = Postprocessor().process(0.7, detail(p, v), p, v)
    assert result.unknown_skills == ["sql"]
    assert "not proof" in result.explanation
    assert result.summary is result.resume_id is result.cover_letter is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_experience(value):
    with pytest.raises(ValidationError):
        Profile(experience_years=value)
