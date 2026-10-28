"""Refactor boundaries and saved-feature compatibility, not model quality tests."""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from resumeanalysis import cli
from resumeanalysis.cover_letters.service import CoverLetterService
from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.extraction.compatibility import identity_source_hashes
from resumeanalysis.integration import ApplyPilotClient
from resumeanalysis.recommendation.engine import RecommendationEngine
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis.recommendation.repository import SQLiteFeedbackRepository
from resumeanalysis.recommendation.schema import StructuredProfile, StructuredVacancy

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/runtime_refactor_baseline.json").read_text()
)


@pytest.mark.parametrize("row", FIXTURE["cases"])
def test_features_and_identities_match_the_captured_pre_refactor_outputs(row):
    result = PreferenceFeatureBuilder(
        LexicalSmokeEncoder(), embedding_components=row["components"]
    ).build(
        StructuredProfile.model_validate(row["profile"]),
        StructuredVacancy.model_validate(row["vacancy"]),
    )
    snapshot = result.snapshot.model_dump(mode="json")
    expected_snapshot = dict(row["snapshot"])
    # Python/NumPy builds can differ by one floating-point rounding unit. Keep
    # identities and categorical evidence exact; tolerate only numeric roundoff.
    assert snapshot.pop("values") == pytest.approx(
        expected_snapshot.pop("values"), rel=0, abs=1e-12
    )
    assert snapshot.pop("content_score") == pytest.approx(
        expected_snapshot.pop("content_score"), rel=0, abs=1e-12
    )
    assert snapshot == expected_snapshot
    qualification = result.qualification.model_dump(mode="json")
    expected_qualification = dict(row["qualification"])
    score, expected_score = qualification.pop("score"), expected_qualification.pop("score")
    if expected_score is None:
        assert score is None
    else:
        assert score == pytest.approx(expected_score, rel=0, abs=1e-12)
    assert qualification == expected_qualification
    assert result.constraints.model_dump(mode="json") == row["constraints"]


def test_all_runtime_modules_import_without_legacy_available():
    program = """
import importlib, importlib.abc, pkgutil, sys
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('resumeanalysis_legacy'):
            raise AssertionError('Runtime attempted to import optional legacy code: ' + fullname)
sys.meta_path.insert(0, NoLegacy())
import resumeanalysis
for module in pkgutil.walk_packages(resumeanalysis.__path__, resumeanalysis.__name__ + '.'):
    if not module.name.endswith('.__main__'):
        importlib.import_module(module.name)
assert not hasattr(resumeanalysis, 'Predictor')
assert not any(name.startswith('resumeanalysis_legacy') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_runtime_has_only_the_eight_product_packages():
    root = Path(__file__).parents[1] / "src/resumeanalysis"
    packages = {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert packages == {
        "recommendation",
        "extraction",
        "normalization",
        "embeddings",
        "profiles",
        "matching",
        "cover_letters",
        "integration",
    }


def test_source_compatibility_is_limited_to_the_exact_reviewed_refactor(tmp_path):
    source = Path(__file__).parents[1] / "src/resumeanalysis/extraction"
    manifest = json.loads((source / "identity_compatibility.json").read_text())
    for name in [*manifest["refactored_sources"], "identity_compatibility.json"]:
        shutil.copyfile(source / name, tmp_path / name)
    assert identity_source_hashes(tmp_path) == manifest["previous_sources"]
    target = tmp_path / "phrase_lexicon.py"
    target.write_text(target.read_text() + "\n# An unreviewed source change\n")
    assert identity_source_hashes(tmp_path) != manifest["previous_sources"]


@pytest.mark.parametrize("command", ["train", "evaluate", "predict", "rank", "datasets", "analyze"])
def test_historical_commands_are_not_runtime_entry_points(command, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([command])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_one_applypilot_client_handles_recommendations_feedback_and_explicit_letters():
    class Letters:
        def __init__(self):
            self.calls = []

        async def generate(self, vacancy, resume, **kwargs):
            self.calls.append((vacancy, resume, kwargs))
            return "Authored draft from supplied facts."

    profile = StructuredProfile(user_id="alice", facts={"skills": ["Python"]})
    vacancy = StructuredVacancy(
        vacancy_id="host:1", facts={"description": "Python work", "required_skills": ["Python"]}
    )
    letters = Letters()
    repository = SQLiteFeedbackRepository(":memory:")
    client = ApplyPilotClient(
        RecommendationEngine(PreferenceFeatureBuilder(LexicalSmokeEncoder()), repository),
        SimpleNamespace(get_profile=lambda user: profile),
        SimpleNamespace(list_vacancies=lambda user: [vacancy]),
        letters=CoverLetterService(letters),
    )
    recommendation = client.recommend("alice")[0]
    assert not letters.calls  # Showing jobs never generates letters implicitly.
    client.record_feedback("alice", vacancy.vacancy_id, "like", event_id="like-1")
    assert asyncio.run(client.draft_cover_letter("alice", vacancy.vacancy_id)).startswith(
        "Authored"
    )
    assert recommendation.qualification.matched_skills == ["python"]
    assert letters.calls[0][1].profile.skills == ["Python"]
    for user, job in [("bob", vacancy.vacancy_id), ("alice", "unavailable")]:
        with pytest.raises(ValueError):
            asyncio.run(client.draft_cover_letter(user, job))
    profile.review_status = "needs_review"
    with pytest.raises(ValueError, match="Confirm"):
        asyncio.run(client.draft_cover_letter("alice", vacancy.vacancy_id))
    assert len(letters.calls) == 1
    repository.close()


def test_cover_letter_cli_uses_selected_profile_without_a_matching_backend(
    tmp_path, monkeypatch, capsys
):
    from resumeanalysis.cover_letters import cli as letters_cli

    class Generator:
        async def generate(self, vacancy, resume, **kwargs):
            assert resume.profile.skills == ["Python"]
            return "A test draft."

    monkeypatch.setattr(letters_cli, "LocalCoverLetterGenerator", lambda *a, **kw: Generator())
    profile = tmp_path / "profile.json"
    vacancy = tmp_path / "vacancy.json"
    profile.write_text(
        StructuredProfile(user_id="alice", facts={"skills": ["Python"]}).model_dump_json()
    )
    vacancy.write_text(
        StructuredVacancy(vacancy_id="host:1", facts={"description": "Python"}).model_dump_json()
    )
    cli.main(["cover-letter", "--profile", str(profile), "--vacancy", str(vacancy)])
    assert json.loads(capsys.readouterr().out) == {
        "vacancy_id": "host:1",
        "cover_letter": "A test draft.",
        "needs_review": True,
    }
