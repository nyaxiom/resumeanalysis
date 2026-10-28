import numpy as np
import pytest
from resumeanalysis_legacy.features.feature_builder import FEATURE_NAMES, FeatureEngineering

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis.schemas import Profile, Vacancy


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
