"""Metric correctness and held-out threshold guards; fixtures are authored, not real qrels."""

import json
from types import SimpleNamespace

import pytest

from resumeanalysis.data.dataset import PairRecord
from resumeanalysis.data.download import sha256_file
from resumeanalysis.experiments.rerank_evaluation import (
    benchmark_worker,
    candidate_groups,
    freeze_threshold,
    grouped_ranking_report,
    pair_metrics,
    query_ranking_metrics,
    validation_threshold,
)
from resumeanalysis.schemas import Profile, Vacancy


def pair(candidate, vacancy, label, *, text=None):
    return PairRecord(
        candidate_id=candidate,
        vacancy_id=vacancy,
        profile=Profile(resume_text=text or f"Resume for {candidate}"),
        vacancy=Vacancy(description=f"Job {vacancy}"),
        label=label,
        provenance="authored software fixture, not independent hiring relevance",
    )


def test_rank_metrics_retain_missing_relevant_candidates_in_denominator():
    row = query_ranking_metrics([2, 1, 0], [1], 3)
    assert row["precision_at_k"] == pytest.approx(1 / 3)
    assert row["recall_at_k"] == 0.5
    assert row["ndcg_at_k"] == pytest.approx(1 / (3 + 1 / 1.584962500721156))
    assert row["mrr_at_k"] == 1
    assert row["returned"] == 1


def test_rank_metrics_all_negative_query_has_explicit_undefined_metrics():
    row = query_ranking_metrics([0, 0], [0, 1], 10)
    assert row["precision_at_k"] == 0
    assert row["recall_at_k"] is None
    assert row["ndcg_at_k"] is None
    assert row["mrr_at_k"] is None


@pytest.mark.parametrize(
    "grades,order,k",
    [
        ([1, 0], [0, 0], 1),
        ([1, 0], [2], 1),
        ([None, 0], [0], 1),
        ([True, 0], [0], 1),
        ([3, 0], [0], 1),
        ([1, 0], [0], True),
        ([], [], 1),
        ([1], [0], 0),
    ],
)
def test_invalid_ranking_inputs_rejected(grades, order, k):
    with pytest.raises(ValueError):
        query_ranking_metrics(grades, order, k)


def test_true_pools_only_with_stable_input_ties():
    records = [pair("a", "a1", 0), pair("a", "a2", 1), pair("b", "b1", 1)]
    report = grouped_ranking_report(records, [0.5, 0.5, 0.9], k_values=(1, 2))
    assert report["queries"] == 1
    assert report["singleton_candidates_excluded"] == 1
    assert report["means"]["1"]["mrr_at_k"] == 0
    assert report["means"]["2"]["mrr_at_k"] == 0.5
    assert len(candidate_groups(records, minimum_size=1)) == 2


def test_preselection_recall_and_policy_use_complete_judged_pool():
    records = [pair("a", "a1", 1), pair("a", "a2", 1), pair("a", "a3", 0)]
    report = grouped_ranking_report(
        records,
        [0.9, 0.8, 0.7],
        retrieval_scores=[0.3, 0.1, 0.2],
        preselect_k=1,
        k_values=(2,),
    )
    assert report["preselect_recall"] == 0.5
    assert report["means"]["2"]["recall_at_k"] == 0.5
    assert report["means"]["2"]["precision_at_k"] == 0.5
    policy = grouped_ranking_report(records, [0.9, 0.8, 0.7], explicit_orders={"a": []})
    assert policy["means"]["1"]["recall_at_k"] == 0
    assert policy["means"]["1"]["precision_at_k"] == 0


def test_group_identity_and_duplicates_are_checked():
    with pytest.raises(ValueError, match="inconsistent profiles"):
        candidate_groups([pair("a", "1", 1), pair("a", "2", 0, text="Changed candidate")])
    with pytest.raises(ValueError, match="Duplicate judged"):
        candidate_groups([pair("a", "1", 1), pair("a", "1", 1)])
    changed = pair("b", "1", 0)
    changed.vacancy.description = "Changed vacancy"
    with pytest.raises(ValueError, match="inconsistent vacancy"):
        candidate_groups([pair("a", "1", 1), changed])
    with pytest.raises(ValueError, match="foreign"):
        grouped_ranking_report(
            [pair("a", "1", 1), pair("a", "2", 0)], [0.2, 0.3], explicit_orders={"a": [9]}
        )


def test_binary_metrics_and_threshold_selection():
    selected = validation_threshold([0, 0, 1, 1], [-9.0, -1.0, 2.0, 4.0])
    assert -1 < selected <= 2
    report = pair_metrics([0, 0, 1, 1], [-9.0, -1.0, 2.0, 4.0], selected)
    assert report["macro_f1"] == report["roc_auc"] == 1
    gated = pair_metrics([0, 1], [-1.0, 2.0], 0.0, decisions=[False, False])
    assert gated["recall"] == 0
    assert gated["roc_auc"] == 1  # AUC remains the explicitly separate semantic score.
    assert pair_metrics([0, 0], [0.1, 0.3], 0.5)["roc_auc"] is None
    with pytest.raises(ValueError):
        validation_threshold([1, 1], [1.0, 2.0])
    with pytest.raises(ValueError):
        pair_metrics([0, 1], [0.0, float("nan")], 0.5)
    with pytest.raises(ValueError):
        pair_metrics([0, 1], [0.0, 1.0], 0.5, decisions=[0, 1])


def test_frozen_threshold_is_immutable(tmp_path):
    path = tmp_path / "threshold.json"
    kwargs = dict(
        method="cosine",
        threshold=0.1,
        source_hash="validation-sha",
        validation_scores=[0.1, 0.3],
        identity="fixed-model",
        selection="validation",
    )
    artifact = freeze_threshold(path, **kwargs)
    assert artifact["test_labels_used"] is False
    assert artifact["source_split"] == "validation"
    with pytest.raises(FileExistsError):
        freeze_threshold(path, **kwargs)


def test_worker_persists_threshold_before_test_inference(tmp_path, monkeypatch):
    import torch

    import resumeanalysis.embeddings.encoder as encoder_module
    import resumeanalysis.inference.predictor as predictor_module

    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation", "test"):
        records = [pair(f"{split}{i}", f"{split}{i}", i) for i in (0, 1)]
        (data / f"{split}.jsonl").write_text(
            "".join(record.model_dump_json() + "\n" for record in records)
        )
    hashes = {name: sha256_file(data / f"{name}.jsonl") for name in ("train", "validation", "test")}
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"inert test checkpoint")
    output = tmp_path / "measurement"
    test_calls = []

    class Encoder:
        identity = "authored-encoder"

        def __init__(self, path, cache_dir=None):
            pass

    class Predictor:
        def __init__(self, checkpoint, encoder):
            pass

        def predict(self, profile, vacancy):
            if "test" in profile.resume_text:
                frozen = json.loads((output / "threshold.json").read_text())
                assert frozen["threshold"] == 0.6
                assert frozen["source_sha256"] == hashes["validation"]
                test_calls.append(profile.resume_text)
            score = 0.9 if profile.resume_text.endswith("1") else 0.1
            return SimpleNamespace(relevant=score >= 0.6, relevance=score)

    monkeypatch.setattr(encoder_module, "TextEncoder", Encoder)
    monkeypatch.setattr(predictor_module, "Predictor", Predictor)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"threshold": 0.6, "data_sha256": hashes})
    report = benchmark_worker(
        "mlp",
        data=data,
        output=output,
        e5_path="unused",
        model_path="unused",
        checkpoint=checkpoint,
    )
    assert len(test_calls) == 2
    assert report["test_pair_metrics"]["macro_f1"] == 1
    assert report["source"]["source_hashes"] == hashes
    assert {name: sha256_file(data / f"{name}.jsonl") for name in hashes} == hashes
