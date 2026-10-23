import json

import pytest

from resumeanalysis.data.dataset import PairRecord, save_splits
from resumeanalysis.experiments.comparison import (
    combine_reports,
    decision_metrics,
    evaluate_backend,
)
from resumeanalysis.experiments.partial_profiles import augment_dataset, partial_profile
from resumeanalysis.qwen.predictor import QwenPredictor


def record(split="train", index=0):
    return PairRecord(
        candidate_id=f"{split}-p{index}",
        vacancy_id=f"{split}-v{index}",
        profile={
            "skills": ["Python", "SQL", "Docker", "PyTorch"],
            "resume_text": f"{split} candidate {index} knows Python SQL Docker PyTorch",
            "experience_years": 2,
            "education": ["Degree"],
            "projects": ["Python app"],
            "remote_only": True,
        },
        vacancy={
            "required_skills": ["Python", "Docker"],
            "description": f"{split} vacancy {index}",
            "remote": True,
        },
        label=1,
        provenance="authored fixture; not hiring ground truth",
    )


def test_partial_profiles_keep_lineage_not_labels_and_prevent_text_recovery():
    source = record()
    a = partial_profile(source, "train", 0.5, 42)
    b = partial_profile(source, "train", 0.5, 42)
    assert a == b and a.label is None and a.source_label_metadata == 1
    assert a.source_split == "train" and a.candidate_id == source.candidate_id
    assert len(a.profile.skills) == 2 and a.retained_skill_fraction == 0.5
    assert not a.profile.resume_text and not a.profile.projects and not a.profile.education
    assert a.source_provenance == source.provenance
    assert a.retained_information_fraction < 1
    assert source.profile.resume_text  # No source mutation.
    assert partial_profile(source, "train", 1).profile == source.profile
    assert set(partial_profile(source, "train", 0.25).profile.skills) <= set(a.profile.skills)
    same_candidate_other_vacancy = source.model_copy(update={"vacancy_id": "another"})
    assert partial_profile(same_candidate_other_vacancy, "train", 0.5).profile == a.profile


def test_absent_skills_are_not_reextracted_or_marked_denied():
    source = record().model_copy(deep=True)
    source.profile.skills = []
    view = partial_profile(source, "train", 0.5)
    assert view.profile.skills == [] and view.profile.denied_skills == []
    assert view.retained_skill_fraction is None
    with pytest.raises(ValueError):
        partial_profile(source, "unknown", 0.5)


def test_zero_years_remains_known_until_explicit_dropout():
    source = record().model_copy(deep=True)
    source.profile = source.profile.model_validate({"skills": ["Python"], "experience_years": 0})
    full = partial_profile(source, "train", 1)
    partial = partial_profile(source, "train", 0.25)
    assert full.profile.experience_years == 0 and full.retained_information_fraction == 1
    assert partial.profile.experience_years is None
    assert "experience_years" in partial.augmentation["masked_fields"]


def test_augmentation_partitioned_before_dropout_and_official_test_unchanged(tmp_path):
    splits = {s: [record(s)] for s in ("train", "validation", "test")}
    save_splits(splits, tmp_path / "official", {})
    original = (tmp_path / "official/test.jsonl").read_bytes()
    result = augment_dataset(tmp_path / "official", tmp_path / "partial")
    assert result["counts"] == {"train": 4, "validation": 4}
    assert not (tmp_path / "partial/test.observations.jsonl").exists()
    assert (tmp_path / "official/test.jsonl").read_bytes() == original
    for line in (tmp_path / "partial/train.observations.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["label"] is None and row["source_split"] == "train"
    with pytest.raises(ValueError):
        augment_dataset(tmp_path / "official", tmp_path / "official/partial")
    splits["test"] = splits["train"]
    (tmp_path / "official/test.jsonl").write_text(splits["test"][0].model_dump_json() + "\n")
    with pytest.raises(ValueError, match="leakage"):
        augment_dataset(tmp_path / "official", tmp_path / "leaky")


class CatalogGenerator:
    """Produces mechanical schema-valid decisions; not a Qwen quality simulation."""

    def generate(self, messages, schema):
        data = json.loads(messages[1]["content"])
        skills = data["input"]["profile"]["skills"]
        positive = bool(skills)
        return json.dumps(
            {
                "relevant": positive,
                "relevance": 0.8 if positive else 0.0,
                **data["skill_catalog"],
                "evidence": [],
            }
        )


def test_shared_evaluation_same_views_and_partial_labels_never_scored():
    model = QwenPredictor(generator=CatalogGenerator())
    reports = {
        name: evaluate_backend(model, [record("test")], warmup=0) for name in ("mock-a", "mock-b")
    }
    combined = combine_reports(reports)
    assert "Same ordered" in combined["comparison"]
    full = reports["mock-a"]["views"]["1.0"]
    assert full["metrics"]["accuracy"] == 1 and full["metrics"]["roc_auc"] is None
    for ratio in ("0.75", "0.5", "0.25"):
        assert reports["mock-a"]["views"][ratio]["metrics"] is None
        assert "diagnostics" in reports["mock-a"]["views"][ratio]["label_status"]
    reports["mock-b"]["source_record_sha256"] = ["different"]
    with pytest.raises(ValueError, match="different"):
        combine_reports(reports)


def test_metrics_use_final_boolean_after_preferences():
    model = QwenPredictor(generator=CatalogGenerator())
    analysis = model.predict(
        {"skills": ["Python"], "remote_only": True},
        {"required_skills": ["Python"], "remote": False},
    )
    assert not analysis.relevant and analysis.relevance == 0.8
    result = decision_metrics([1], [analysis])
    assert result["accuracy"] == 0 and result["confusion_matrix"] == [[0, 0], [1, 0]]


def test_normalized_conversion_consumed_by_existing_mlp_and_qwen(tmp_path):
    from resumeanalysis.data.adapters import canonical_spec
    from resumeanalysis.data.conversion import convert_dataset
    from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
    from resumeanalysis.experiments.partial_profiles import load_splits
    from resumeanalysis.inference.predictor import Predictor
    from resumeanalysis.training.train import TrainingConfig, train

    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "validation", "test"):
        records = [record(split, i).model_copy(update={"label": i % 2}) for i in range(6)]
        (source / (split + ".jsonl")).write_text(
            "".join(r.model_dump_json() + "\n" for r in records)
        )
    normalized = tmp_path / "normalized"
    convert_dataset(
        source,
        normalized,
        canonical_spec(),
        dataset="test-fixture",
        revision="v1",
        provenance="Authored fixture; tests mechanics only",
    )
    encoder = LexicalSmokeEncoder()
    train(
        normalized,
        tmp_path / "run",
        encoder,
        TrainingConfig(epochs=2, patience=1, hidden_dims=(8,)),
    )
    records = load_splits(normalized)["test"]
    models = {
        "mlp": Predictor(tmp_path / "run/model.pt", encoder),
        "mock-qwen": QwenPredictor(generator=CatalogGenerator()),
    }
    reports = {name: evaluate_backend(model, records, warmup=0) for name, model in models.items()}
    assert combine_reports(reports)["backends"]["mlp"]["views"]["1.0"]["metrics"]["count"] == 6
