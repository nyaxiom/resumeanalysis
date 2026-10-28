import json

import pytest
from resumeanalysis_legacy.data.dataset import PairRecord, save_splits
from resumeanalysis_legacy.experiments.extractor_ablation import CONDITIONS, run_ablation
from resumeanalysis_legacy.training.train import TrainingConfig, train

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.extraction.registry import create_extractor


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("extractor-ablation")
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = [
            PairRecord(
                candidate_id=f"{split}-candidate-{index}",
                vacancy_id=f"{split}-vacancy-{index}",
                profile={
                    "resume_text": f"{split} candidate record {index}",
                    "skills": ["Python"] if index % 2 else ["Excel"],
                },
                vacancy={
                    "title": f"Unchanged title {index}",
                    "description": f"{split} vacancy record {index}\nEssential Skills\nPython or Node.js\nNice to have\nDocker",
                    "required_skills": ["Python", "UnknownCaseTool"],
                    "optional_skills": ["Docker"],
                },
                label=index % 2,
                provenance="authored test fixture; source labels are not real hiring judgments",
            )
            for index in range(6)
        ]
    save_splits(splits, root / "data", {})
    encoder = LexicalSmokeEncoder()
    configuration = TrainingConfig(epochs=2, patience=1, hidden_dims=(8,), dropout=0)
    for name in ("legacy", "v2"):
        train(root / "data", root / name, encoder, configuration, create_extractor(name))
    return root, encoder


def ablate(experiment, output, **kwargs):
    root, encoder = experiment
    return run_ablation(
        root / "data", root / "legacy/model.pt", root / "v2/model.pt", encoder, output, **kwargs
    )


def test_four_conditions_share_original_examples_and_only_declared_lists_change(
    experiment, tmp_path
):
    root, _ = experiment
    originals = {
        split: (root / "data" / (split + ".jsonl")).read_bytes()
        for split in ("train", "validation", "test")
    }
    checkpoints = {name: (root / name / "model.pt").read_bytes() for name in ("legacy", "v2")}
    output = tmp_path / "report.json"
    report = ablate(experiment, output)
    assert set(report["conditions"]) == {condition for condition, _, _ in CONDITIONS}
    assert report["count"] == 6 and report["source_split"] == "test"
    assert report["checkpoints"]["legacy"]["format_version"] == 1
    assert report["checkpoints"]["v2"]["format_version"] == 2
    for condition, _, explicit in CONDITIONS:
        result = report["conditions"][condition]
        assert result["source_record_sha256"] == report["source_record_sha256"]
        assert result["source_label_diagnostic_metrics"]["count"] == 6
        assert "not revalidated" in result["label_warning"]
        for index, row in enumerate(result["records"]):
            assert row["candidate_id"] == f"test-candidate-{index}"
            assert row["vacancy_id"] == f"test-vacancy-{index}"
            assert row["skill_span_or_requirement_accuracy"] is None
            assert 0 <= row["analysis"]["relevance"] <= 1
            if explicit:
                assert row["supplied_required_skills"] == ["Python", "UnknownCaseTool"]
                assert row["supplied_optional_skills"] == ["Docker"]
                assert row["intervention"]["removed_values"] == {}
                assert "unknowncasetool" in row["effective_independent_required_skills"]
            else:
                assert row["supplied_required_skills"] == row["supplied_optional_skills"] == []
                assert row["intervention"]["removed_values"] == {
                    "/vacancy/required_skills": ["Python", "UnknownCaseTool"],
                    "/vacancy/optional_skills": ["Docker"],
                }
            assert row["inference_wall_seconds"] >= 0
            assert row["inference_process_cpu_seconds"] >= 0
            assert row["vacancy_extraction"]["mentions"]
    assert json.loads(output.read_text())["source_record_sha256"] == report["source_record_sha256"]
    for split, contents in originals.items():
        assert (root / "data" / (split + ".jsonl")).read_bytes() == contents
    for name, contents in checkpoints.items():
        assert (root / name / "model.pt").read_bytes() == contents


def test_ablation_uses_compatible_v2_checkpoint_not_original_weights(experiment, tmp_path):
    root, encoder = experiment
    output = tmp_path / "incompatible.json"
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        run_ablation(
            root / "data", root / "legacy/model.pt", root / "legacy/model.pt", encoder, output
        )
    assert not output.exists()


def test_training_and_evaluation_partitions_must_match(experiment, tmp_path):
    root, encoder = experiment
    changed = tmp_path / "different-data"
    changed.mkdir()
    for split in ("train", "validation", "test"):
        (changed / (split + ".jsonl")).write_bytes(
            (root / "data" / (split + ".jsonl")).read_bytes()
        )
    rows = [json.loads(line) for line in (changed / "train.jsonl").read_text().splitlines()]
    rows[0]["label"] = 1 - rows[0]["label"]
    (changed / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="same original train/validation/test"):
        run_ablation(
            changed,
            root / "legacy/model.pt",
            root / "v2/model.pt",
            encoder,
            tmp_path / "wrong-data.json",
        )


def test_report_paths_are_fresh_and_outside_source(experiment, tmp_path):
    root, _ = experiment
    path = tmp_path / "existing.json"
    path.write_text("preserve previous result")
    with pytest.raises(FileExistsError, match="fresh"):
        ablate(experiment, path)
    assert path.read_text() == "preserve previous result"
    with pytest.raises(ValueError, match="outside"):
        ablate(experiment, root / "data/new-report.json")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_subset_limit_rejected(experiment, tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        ablate(experiment, tmp_path / "invalid.json", limit=limit)


def test_explicit_subset_is_reported_and_keeps_all_four_conditions(experiment, tmp_path):
    report = ablate(experiment, tmp_path / "subset.json", limit=1)
    assert report["limit"] == report["count"] == 1
    assert all(condition["count"] == 1 for condition in report["conditions"].values())
