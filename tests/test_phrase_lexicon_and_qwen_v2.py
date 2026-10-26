import hashlib
import json

import pytest

from resumeanalysis.extraction import phrase_lexicon
from resumeanalysis.extraction.registry import create_extractor, extractor_identity, is_legacy
from resumeanalysis.qwen.predictor import QwenPredictor


def synthetic_training(tmp_path, monkeypatch):
    rows = [
        {
            "tokens": ["mentoring", "Python", "or", "Java"],
            "tags_skill": ["B", "B", "I", "I"],
            "tags_knowledge": ["O"] * 4,
        }
    ] * 2
    source = tmp_path / "train.json"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        phrase_lexicon, "TRAIN_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    return source


def test_train_only_phrase_artifact_counts_source_license_and_offsets(tmp_path, monkeypatch):
    source = synthetic_training(tmp_path, monkeypatch)
    target = tmp_path / "phrases.json"
    metadata = phrase_lexicon.build_lexicon(source, target)
    assert metadata["training_rows"] == 2 and metadata["phrase_count"] == 1
    loaded = phrase_lexicon.load_lexicon(target)
    assert loaded["phrases"] == ["mentoring"]
    assert "Copyright (c) 2023 Mike Zhang" in loaded["license_notice"]
    model = create_extractor("v2", lexicon_path=target)
    result = model.extract("Required: mentoring.", vacancy=True)
    assert result.required == {"mentoring"}
    assert result.mentions[0].provenance == "skillspan-train-phrase"
    assert model.identity != create_extractor("v2").identity
    with pytest.raises(FileExistsError):
        phrase_lexicon.build_lexicon(source, target)
    with pytest.raises(ValueError):
        create_extractor("legacy", lexicon_path=target)


def test_lexicon_rejects_other_splits_and_modified_artifacts(tmp_path, monkeypatch):
    source = synthetic_training(tmp_path, monkeypatch)
    target = tmp_path / "phrases.json"
    phrase_lexicon.build_lexicon(source, target)
    other = tmp_path / "renamed-test-as-train.json"
    other.write_text(source.read_text() + "{}\n")
    with pytest.raises(ValueError, match="TRAIN"):
        phrase_lexicon.build_lexicon(other, tmp_path / "bad.json")
    contents = json.loads(target.read_text())
    contents["phrases"].append("invented")
    target.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="provenance"):
        phrase_lexicon.load_lexicon(target)


def test_mutated_legacy_patterns_cannot_masquerade_as_original():
    import re

    model = create_extractor("legacy")
    model.pattern = re.compile("invented")
    assert not is_legacy(model)
    with pytest.raises(ValueError, match="identity"):
        extractor_identity(model)


@pytest.mark.parametrize(
    "text", ["We go to programming meetups.", "A C rating for the development plan."]
)
def test_ordinary_words_are_not_programming_qualifications(text):
    assert not create_extractor("v2").extract(text).confirmed


def test_qwen_v2_uses_same_catalog_and_retains_alternative_metadata():
    class Generator:
        def generate(self, messages, schema):
            data, _ = json.JSONDecoder().raw_decode(messages[1]["content"])
            assert "do not flatten alternatives" in messages[1]["content"]
            return json.dumps(
                {"relevant": True, "relevance": 0.8, **data["skill_catalog"], "evidence": []}
            )

    model = QwenPredictor(generator=Generator(), extractor=create_extractor("v2"))
    result = model.predict(
        {"skills": ["Node.js"]}, {"description": "Essential: Python or Node.js, Docker."}
    )
    assert result.matched_skills == ["node.js"] and result.unknown_skills == ["docker"]
    assert "one option suffices" in result.explanation
    assert "python" not in result.missing_skills


def test_console_extractor_dispatch_exits_successfully_without_duplicate_stderr():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from resumeanalysis.cli import main; sys.exit(main())",
            "extractors",
            "extract",
            "--extractor",
            "v2",
            "--vacancy",
            "--text",
            "Required: Python",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["required"] == ["python"]
    assert completed.stderr == ""


def test_role_radar_optional_model_uses_supplied_profile_and_audits_failure(tmp_path):
    from resumeanalysis.experiments.extractor_evaluation import evaluate_role_radar
    from resumeanalysis.schemas import VacancyAnalysis

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "profile": {"skills": ["Python"]},
                "vacancy": {"description": "Essential: Python"},
                "score": 0.91,
            }
        )
        + "\n"
    )

    class Backend:
        def predict(self, profile, vacancy):
            assert profile.skills == ["Python"] and vacancy.description == "Essential: Python"
            return VacancyAnalysis(
                relevant=False,
                relevance=0.4,
                matched_skills=["python"],
                missing_skills=[],
                unknown_skills=[],
                contradicted_skills=[],
                experience_gap_years=None,
                explanation="Authored test output only",
            )

    report = evaluate_role_radar(path, create_extractor("v2"), backend=Backend())
    assert report["record_outcomes"][0]["analysis"]["relevance"] == 0.4
    assert report["record_outcomes"][0]["source_scores_metadata"] == {"score": 0.91}
    assert report["accuracy_metrics"] is None

    class FailingBackend:
        def predict(self, profile, vacancy):
            raise RuntimeError("private failure text must not leak")

    failed = evaluate_role_radar(path, create_extractor("v2"), backend=FailingBackend())
    assert (
        failed["outcome_counts"] == {"diagnosed_with_model_error": 1}
        and failed["count_reconciliation"]
    )
    assert failed["record_outcomes"][0]["model_status"] == "failed"
    assert "private failure text" not in json.dumps(failed)


def test_whitespace_only_extractor_span_is_not_silently_discarded(tmp_path):
    from types import SimpleNamespace

    from resumeanalysis.experiments.extractor_evaluation import evaluate_skillspan
    from resumeanalysis.extraction.skill_extractor import Extraction, SkillMention

    path = tmp_path / "spans.jsonl"
    path.write_text(
        json.dumps({"tokens": ["a", "b"], "tags_skill": ["O", "O"], "tags_knowledge": ["O", "O"]})
        + "\n"
    )
    model = SimpleNamespace(
        extract=lambda text: Extraction(
            mentions=[SkillMention("invented", " ", 1, 2, "confirmed", "mentioned")]
        )
    )
    with pytest.raises(ValueError, match="overlaps no source token"):
        evaluate_skillspan(path, model)
