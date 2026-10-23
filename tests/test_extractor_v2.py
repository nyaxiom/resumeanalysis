"""Manually authored regressions; these are not independent language-quality claims."""

import json
from pathlib import Path

import numpy as np
import pytest

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.features.feature_builder import FEATURE_NAMES, FeatureEngineering
from resumeanalysis.schemas import Profile, Vacancy

FIXTURE_PATH = Path(__file__).parent / "fixtures/skill_extraction_v2.json"
FIXTURES = json.loads(FIXTURE_PATH.read_text())


def categories(result):
    return {
        "required": set(result.required),
        "optional": set(result.optional),
        "mentioned": {m.skill for m in result.mentions if m.requirement == "mentioned"},
        "negated_requirements": set(result.negated_requirements),
        "alternative_groups": {frozenset(g.skills) for g in result.alternative_groups},
    }


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda row: row["id"])
def test_manually_authored_requirement_fixtures(fixture):
    result = create_extractor("v2").extract(fixture["text"], vacancy=fixture["vacancy"])
    actual = categories(result)
    for name, values in fixture["expected"].items():
        expected = {frozenset(v) for v in values} if name == "alternative_groups" else set(values)
        assert actual[name] == expected, name
    for mention in result.mentions:
        assert 0 <= mention.start < mention.end <= len(fixture["text"])
        assert fixture["text"][mention.start : mention.end] == mention.text
        assert mention.provenance
    for group in result.alternative_groups:
        assert fixture["text"][group.start : group.end] == group.text
        assert group.requirement in {"required", "optional", "mentioned", "negated"}
        assert len(set(group.skills)) >= 2


def test_fixture_scope_and_language_accounting_are_explicit():
    assert len({f["id"] for f in FIXTURES}) == len(FIXTURES)
    assert {f["language"] for f in FIXTURES} == {"en", "ru"}
    assert all("regression fixture" in f["label_note"] for f in FIXTURES)


def test_essential_skills_regression_preserves_the_legacy_baseline():
    text = FIXTURES[0]["text"]
    legacy = create_extractor("legacy")
    before_identity = legacy.normalizer.taxonomy.identity
    assert type(legacy) is SkillExtractor
    assert legacy.extract(text, vacancy=True).required == set()
    improved = create_extractor("v2").extract(text, vacancy=True)
    assert len(improved.required) == 15
    assert legacy.normalizer.taxonomy.identity == before_identity
    assert create_extractor("legacy").extract(text, vacancy=True) == legacy.extract(
        text, vacancy=True
    )


@pytest.mark.parametrize(
    ("text", "confirmed", "denied", "uncertain"),
    [
        ("No experience with Docker or SQL, but Python.", {"python"}, {"docker", "sql"}, set()),
        (
            "Нет опыта с Docker, но знаю Python. Не владею SQL.",
            {"python"},
            {"docker", "sql"},
            set(),
        ),
        ("Python. No experience with Python.", set(), set(), {"python"}),
        ("Want to learn Airflow and Docker.", set(), set(), {"airflow", "docker"}),
        ("Хочу изучить Docker и PostgreSQL.", set(), set(), {"docker", "postgresql"}),
    ],
)
def test_candidate_evidence_negation_and_uncertainty(text, confirmed, denied, uncertain):
    result = create_extractor("v2").extract(text)
    assert result.confirmed == confirmed
    assert result.denied == denied
    assert result.uncertain == uncertain


def test_negated_vacancy_requirements_never_imply_candidate_absence():
    result = create_extractor("v2").extract("No Kubernetes experience required.", vacancy=True)
    assert result.negated_requirements == {"kubernetes"}
    assert not result.required and not result.optional and not result.denied
    assert "kubernetes" not in result.confirmed


@pytest.mark.parametrize("text", ["", "  \n\t\n"])
def test_empty_input(text):
    result = create_extractor("v2").extract(text, vacancy=True)
    assert result.mentions == [] and result.alternative_groups == []
    assert result.required == result.optional == result.negated_requirements == set()


@pytest.mark.parametrize("text", [None, {}, [], b"Required: Python", 123])
def test_malformed_input_rejected(text):
    with pytest.raises(TypeError, match="text"):
        create_extractor("v2").extract(text, vacancy=True)


def test_unicode_offsets_and_punctuation_are_exact():
    text = "🚀 Работа\nОбязательно: PyTorch, Node . js и scikit - learn.\nЖелательно: Docker."
    result = create_extractor("v2").extract(text, vacancy=True)
    assert result.required == {"pytorch", "node.js", "scikit-learn"}
    assert result.optional == {"docker"}
    expected = {
        "PyTorch": "pytorch",
        "Node . js": "node.js",
        "scikit - learn": "scikit-learn",
        "Docker": "docker",
    }
    assert {m.text: m.skill for m in result.mentions} == expected
    for mention in result.mentions:
        assert mention.start == text.index(mention.text)
        assert mention.end == mention.start + len(mention.text)


def test_long_description_keeps_evidence_after_large_background_section():
    background = "Background information about the employer.\n" * 2000
    text = background + "Essential Skills\nPython and Docker"
    result = create_extractor("v2").extract(text, vacancy=True)
    assert result.required == {"python", "docker"}
    assert all(m.start >= len(background) for m in result.mentions)


def test_unknown_explicit_structured_skills_are_authoritative():
    builder = FeatureEngineering(LexicalSmokeEncoder(), create_extractor("v2"))
    detail = builder.build(
        Profile(skills=["Python", "NebulaDBX"]),
        Vacancy(
            description="A role whose wording omits the technologies.",
            required_skills=["Python", "NebulaDBX"],
        ),
    )
    assert detail.required == detail.matched == {"python", "nebuladbx"}
    assert detail.unknown == detail.contradicted == set()
    assert detail.vector[FEATURE_NAMES.index("required_match")] == 1
    assert np.isfinite(detail.vector).all()


def test_job_title_does_not_create_requirements_or_qualifications():
    builder = FeatureEngineering(LexicalSmokeEncoder(), create_extractor("v2"))
    detail = builder.build(
        Profile(), Vacancy(title="Python engineer", description="Contact our hiring team.")
    )
    assert not detail.required and not detail.matched
    assert detail.vector[FEATURE_NAMES.index("requirements_known")] == 0


def test_or_group_is_one_requirement_unit_and_preserves_evidence():
    builder = FeatureEngineering(LexicalSmokeEncoder(), create_extractor("v2"))
    detail = builder.build(
        Profile(skills=["Python"]), Vacancy(description="Required: Python or Node.js, Docker.")
    )
    assert detail.required == {"docker"}
    assert detail.unknown == {"docker"}
    assert detail.matched == {"python"}
    assert detail.vector[FEATURE_NAMES.index("required_match")] == 0.5
    assert detail.vector[FEATURE_NAMES.index("required_unknown")] == 0.5
    assert detail.vector[FEATURE_NAMES.index("required_count_scaled")] == pytest.approx(2 / 20)
    assert len(detail.alternative_groups) == 1
    assert set(detail.alternative_groups[0]["skills"]) == {"python", "node.js"}
    assert detail.alternative_groups[0]["status"] == "matched"


def test_an_unknown_alternative_is_not_automatic_negative_evidence():
    builder = FeatureEngineering(LexicalSmokeEncoder(), create_extractor("v2"))
    detail = builder.build(
        Profile(denied_skills=["Python"]), Vacancy(description="Required: Python or Node.js.")
    )
    assert not detail.contradicted
    assert detail.alternative_groups[0]["status"] == "unknown"
    assert detail.vector[FEATURE_NAMES.index("required_denied")] == 0
    assert detail.vector[FEATURE_NAMES.index("required_unknown")] == 1


def test_preferred_alternative_does_not_become_mandatory():
    builder = FeatureEngineering(LexicalSmokeEncoder(), create_extractor("v2"))
    detail = builder.build(
        Profile(skills=["Python"]), Vacancy(description="Preferred: Python or Node.js.")
    )
    assert not detail.required
    assert detail.vector[FEATURE_NAMES.index("requirements_known")] == 0
    assert detail.vector[FEATURE_NAMES.index("required_count_scaled")] == 0
    assert detail.vector[FEATURE_NAMES.index("optional_match")] == 1


def test_short_aliases_require_context_and_do_not_invent_skills():
    extractor = create_extractor("v2")
    assert extractor.extract("A node in the graph.").confirmed == set()
    assert extractor.extract("We go to meetings. A grade of C.").confirmed == set()
    assert extractor.extract("Backend JavaScript: Node.").confirmed == {"node.js", "javascript"}
    assert extractor.extract("Programming languages: C, Go, R.").confirmed == {"c", "golang", "r"}
    assert extractor.extract("Required: NebulaDBX.", vacancy=True).required == set()


def test_unrecognized_backend_name_is_an_explicit_error():
    with pytest.raises(ValueError, match="legacy or v2"):
        create_extractor("unknown")
