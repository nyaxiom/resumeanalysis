"""Authored validator/metric fixtures only; these are not human evaluation results."""

import copy
import json
from pathlib import Path

import pytest

from resumeanalysis.experiments.rerank_human_evaluation import (
    evaluate_test,
    freeze_validation,
    load_judged_groups,
)


def source(identity):
    return {
        "data_origin": "consented_real",
        "source_url": "urn:test-only:" + identity,
        "license": "Test fixture declaration, no real personal data",
        "permission_basis": "Authored software test, not an actual permission record",
        "source_document_id": identity,
    }


def group(split, language="en"):
    return {
        "schema_version": 1,
        "query_id": split + "-query",
        "profile_id": split + "-profile",
        "split": split,
        "language": language,
        "profile": {"resume_text": split + " authored candidate with Python"},
        "provenance": source(split + "-profile"),
        "label_origin": "independent_human",
        "judged_pool_complete": True,
        "candidates": [
            {
                "vacancy_id": f"{split}-job-{i}",
                "vacancy": {"description": f"{split} distinct authored vacancy {i}"},
                "judgment": {
                    "grade": grade,
                    "abstain_reason": None,
                    "assessor_ids": ["fixture-a", "fixture-b"],
                    "adjudicated": True,
                },
                "provenance": source(f"{split}-job-{i}"),
            }
            for i, grade in enumerate((2, 1, 0))
        ],
    }


def prediction(split):
    return {
        "schema_version": 1,
        "query_id": split + "-query",
        "model_identity": "test-only-model-v1",
        "config_identity": "fixed-test-config",
        "candidates": [
            {
                "vacancy_id": f"{split}-job-{i}",
                "score": score,
                "retrieval_score": score,
                "decision": i != 2,
                "evidence_status": "eligible",
            }
            for i, score in enumerate((0.9, 0.8, 0.1))
        ],
    }


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def corpus(tmp_path):
    annotations = tmp_path / "judgments.jsonl"
    val = tmp_path / "validation_predictions.jsonl"
    test = tmp_path / "test_predictions.jsonl"
    frozen = tmp_path / "frozen.json"
    report = tmp_path / "report.json"
    write_lines(annotations, [group("validation"), group("test", "ru")])
    write_lines(val, [prediction("validation")])
    write_lines(test, [prediction("test")])
    return annotations, val, test, frozen, report


def test_freeze_then_human_test_metrics(corpus):
    annotations, val, test, frozen, report = corpus
    artifact = freeze_validation(annotations, val, frozen, k_values=(1, 2, 3))
    assert artifact["test_labels_used"] is False
    assert artifact["threshold"] == pytest.approx(0.45)
    result = evaluate_test(annotations, test, frozen, val, report)
    assert result["overall"]["pair_metrics"]["macro_f1"] == 1.0
    assert result["overall"]["ranking"]["1"]["precision_at_k"] == 1.0
    assert result["overall"]["ranking"]["1"]["recall_at_k"] == 0.5
    assert result["overall"]["ranking"]["3"]["ndcg_at_k"] == 1.0
    assert result["overall"]["grade_2_only_pair_metrics"]["precision"] == 0.5
    assert set(result["languages"]) == {"ru"}
    assert result["overall"]["retrieval_preselect_recall"]["1"]["mean"] == 0.5


def test_filtering_keeps_full_pool_denominators(corpus):
    annotations, val, test, frozen, report = corpus
    scores = prediction("test")
    scores["retrieved_vacancy_ids"] = ["test-job-1"]
    scores["ranked_vacancy_ids"] = ["test-job-1"]
    write_lines(test, [scores])
    freeze_validation(annotations, val, frozen, k_values=(2,))
    result = evaluate_test(annotations, test, frozen, val, report)
    metric = result["overall"]["ranking"]["2"]
    assert metric["precision_at_k"] == 0.5
    assert metric["recall_at_k"] == 0.5
    assert metric["ndcg_at_k"] < 0.5
    assert result["overall"]["per_query"][0]["actual_retrieved_pool_recall"] == 0.5


def test_empty_rank_is_zero_credit(corpus):
    annotations, val, test, frozen, report = corpus
    scores = prediction("test")
    scores["ranked_vacancy_ids"] = []
    write_lines(test, [scores])
    freeze_validation(annotations, val, frozen)
    result = evaluate_test(annotations, test, frozen, val, report)
    assert result["overall"]["ranking"]["1"]["precision_at_k"] == 0
    assert result["overall"]["ranking"]["1"]["recall_at_k"] == 0


def test_all_negative_query_null_ranking_denominators(corpus):
    annotations, val, test, frozen, report = corpus
    negative = group("test")
    for candidate in negative["candidates"]:
        candidate["judgment"]["grade"] = 0
    write_lines(annotations, [group("validation"), negative])
    freeze_validation(annotations, val, frozen)
    result = evaluate_test(annotations, test, frozen, val, report)
    metric = result["overall"]["ranking"]["1"]
    assert metric["precision_at_k"] == 0
    assert metric["precision_at_k_queries"] == 1
    assert metric["recall_at_k"] is None
    assert metric["recall_at_k_queries"] == 0
    assert result["overall"]["pair_metrics"]["roc_auc"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda g: g.update(label_origin="synthetic_diagnostic"),
        lambda g: g.update(judged_pool_complete=False),
        lambda g: g["provenance"].update(data_origin="authored_fixture"),
        lambda g: g["provenance"].update(source_url=None),
        lambda g: g["candidates"][0]["provenance"].update(permission_basis=""),
        lambda g: g["candidates"][0]["judgment"].update(grade=None),
        lambda g: g["candidates"][0]["judgment"].update(grade=True),
        lambda g: g["candidates"][0]["judgment"].update(adjudicated=False),
        lambda g: g["candidates"][0]["judgment"].update(abstain_reason="unclear"),
        lambda g: g["candidates"][0]["judgment"].update(assessor_ids=["same", "same"]),
        lambda g: g["candidates"][0]["judgment"].update(assessor_ids=["one"]),
        lambda g: g["profile"].update(extra="unsupported"),
        lambda g: g["candidates"][1].update(vacancy_id=g["candidates"][0]["vacancy_id"]),
    ],
)
def test_rejects_non_gold_or_invalid_schema(tmp_path, mutation):
    data = group("validation")
    mutation(data)
    path = tmp_path / "data.jsonl"
    write_lines(path, [data])
    with pytest.raises(ValueError):
        load_judged_groups(path)


def test_checked_in_template_is_not_ground_truth():
    path = Path(__file__).resolve().parents[1] / "evaluation/rerank/annotation_template.jsonl"
    with pytest.raises(ValueError):
        load_judged_groups(path)


@pytest.mark.parametrize(
    "entity", ["profile_id", "profile_text", "vacancy_id", "vacancy_text", "source", "source_hash"]
)
def test_detects_cross_split_leakage(tmp_path, entity):
    val, test = group("validation"), group("test")
    if entity == "profile_id":
        test["profile_id"] = val["profile_id"]
    elif entity == "profile_text":
        test["profile"]["resume_text"] = val["profile"]["resume_text"].upper() + "!!!"
    elif entity == "vacancy_id":
        test["candidates"][0]["vacancy_id"] = val["candidates"][0]["vacancy_id"]
    elif entity == "vacancy_text":
        test["candidates"][0]["vacancy"]["description"] = val["candidates"][0]["vacancy"][
            "description"
        ]
    elif entity == "source":
        test["provenance"] = copy.deepcopy(val["provenance"])
    else:
        test["provenance"]["source_sha256"] = val["provenance"]["source_sha256"] = "a" * 64
    path = tmp_path / "data.jsonl"
    write_lines(path, [val, test])
    with pytest.raises(ValueError, match="leak|identity"):
        load_judged_groups(path, split="validation")


def test_frozen_artifact_and_results_cannot_be_overwritten(corpus):
    annotations, val, test, frozen, report = corpus
    freeze_validation(annotations, val, frozen)
    with pytest.raises(FileExistsError):
        freeze_validation(annotations, val, frozen)
    evaluate_test(annotations, test, frozen, val, report)
    with pytest.raises(FileExistsError):
        evaluate_test(annotations, test, frozen, val, report)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(model_identity="changed"),
        lambda p: p.update(config_identity="changed"),
        lambda p: p.update(query_id="foreign"),
        lambda p: p["candidates"].reverse(),
        lambda p: p["candidates"].pop(),
        lambda p: p["candidates"][0].update(score=float("nan")),
        lambda p: p["candidates"][0].update(score=True),
        lambda p: p["candidates"][0].update(score="0.9"),
        lambda p: p["candidates"][0].pop("decision"),
        lambda p: p["candidates"][0].pop("retrieval_score"),
        lambda p: p.update(ranked_vacancy_ids=["foreign"]),
        lambda p: p.update(ranked_vacancy_ids=["test-job-0", "test-job-0"]),
        lambda p: p.update(ranked_vacancy_ids=["test-job-0"], retrieved_vacancy_ids=[]),
        lambda p: p.update(retrieved_vacancy_ids=["test-job-0"]),
        lambda p: p["candidates"].__setitem__(0, ["not an object"]),
    ],
)
def test_rejects_misaligned_changed_or_invalid_predictions(corpus, mutation):
    annotations, val, test, frozen, report = corpus
    freeze_validation(annotations, val, frozen)
    changed = prediction("test")
    mutation(changed)
    write_lines(test, [changed])
    with pytest.raises(ValueError):
        evaluate_test(annotations, test, frozen, val, report)
    assert not report.exists()


def test_validation_predictions_immutable_after_freeze(corpus):
    annotations, val, test, frozen, report = corpus
    freeze_validation(annotations, val, frozen)
    changed = prediction("validation")
    changed["candidates"][0]["score"] = 0.95
    write_lines(val, [changed])
    with pytest.raises(ValueError, match="changed after"):
        evaluate_test(annotations, test, frozen, val, report)


def test_threshold_cannot_be_replaced_with_test_selected_value(corpus):
    annotations, val, test, frozen, report = corpus
    freeze_validation(annotations, val, frozen)
    changed = json.loads(frozen.read_text())
    changed["threshold"] = 0.7
    frozen.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs"):
        evaluate_test(annotations, test, frozen, val, report)


def test_test_grades_do_not_select_threshold(corpus):
    annotations, val, test, frozen, report = corpus
    before = freeze_validation(annotations, val, frozen)
    changed = group("test")
    for candidate in changed["candidates"]:
        candidate["judgment"]["grade"] = 2 - candidate["judgment"]["grade"]
    write_lines(annotations, [group("validation"), changed])
    after = freeze_validation(annotations, val, frozen.with_name("second.json"))
    assert before == after


def test_rank_ties_preserve_original_pool_order(corpus):
    annotations, val, test, frozen, report = corpus
    scores = prediction("test")
    for candidate in scores["candidates"]:
        candidate["score"] = 0.5
    write_lines(test, [scores])
    freeze_validation(annotations, val, frozen)
    result = evaluate_test(annotations, test, frozen, val, report)
    assert result["overall"]["ranking"]["1"]["ndcg_at_k"] == 1


def test_rejects_repeated_vacancy_text_even_under_new_id(tmp_path):
    data = group("validation")
    data["candidates"][1]["vacancy"] = copy.deepcopy(data["candidates"][0]["vacancy"])
    path = tmp_path / "data.jsonl"
    write_lines(path, [data])
    with pytest.raises(ValueError, match="Duplicate vacancy"):
        load_judged_groups(path)


def test_rejects_blank_permission_declaration(tmp_path):
    data = group("validation")
    data["provenance"]["permission_basis"] = "   "
    path = tmp_path / "data.jsonl"
    write_lines(path, [data])
    with pytest.raises(ValueError, match="blank"):
        load_judged_groups(path)


def test_rejects_repeated_profile_under_new_query_and_profile_id(tmp_path):
    original = group("validation")
    duplicate = copy.deepcopy(original)
    duplicate.update(query_id="duplicate-query", profile_id="duplicate-profile")
    duplicate["provenance"] = source("duplicate-source")
    path = tmp_path / "data.jsonl"
    write_lines(path, [original, duplicate])
    with pytest.raises(ValueError, match="Duplicate profile query"):
        load_judged_groups(path)


def test_schema_errors_do_not_echo_private_input(tmp_path):
    data = group("validation")
    data["profile"]["skills"] = "PRIVATE_CANDIDATE_DETAIL"
    path = tmp_path / "data.jsonl"
    write_lines(path, [data])
    with pytest.raises(ValueError) as error:
        load_judged_groups(path)
    assert "PRIVATE_CANDIDATE_DETAIL" not in str(error.value)


def test_non_object_frozen_configuration_is_rejected(corpus):
    annotations, val, test, frozen, report = corpus
    frozen.write_text("null")
    with pytest.raises(ValueError, match="validation-frozen"):
        evaluate_test(annotations, test, frozen, val, report)


def test_policy_violations_reported_without_hiding_predictions(corpus):
    annotations, val, test, frozen, report = corpus
    data = group("test")
    data["candidates"][0]["judgment"]["preference_status"] = "conflict"
    write_lines(annotations, [group("validation"), data])
    scores = prediction("test")
    scores["candidates"][0]["evidence_status"] = "blocked"
    write_lines(test, [scores])
    freeze_validation(annotations, val, frozen)
    result = evaluate_test(annotations, test, frozen, val, report)
    assert result["overall"]["policy_audit_counts"]["approved_with_human_preference_conflict"] == 1
    assert result["overall"]["policy_audit_counts"]["approved_with_reported_block"] == 1


def test_cli_end_to_end(corpus):
    import subprocess
    import sys

    annotations, val, test, frozen, report = corpus
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_rerank_human.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "freeze-validation",
            "--annotations",
            str(annotations),
            "--predictions",
            str(val),
            "--output",
            str(frozen),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "evaluate-test",
            "--annotations",
            str(annotations),
            "--predictions",
            str(test),
            "--validation-predictions",
            str(val),
            "--frozen",
            str(frozen),
            "--output",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(report.read_text())["overall"]["pairs"] == 3
