"""Prospective human relevance evaluation from immutable, fully judged pools.

This module never creates ground truth. Source/permission declarations still need
manual audit; passing validation is not evidence that declarations are true.
"""

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, ValidationError, model_validator

from resumeanalysis.jsonio import loads
from resumeanalysis.schemas import InputModel, Profile, Vacancy
from resumeanalysis_legacy.data.adapters import digest
from resumeanalysis_legacy.data.dataset import fingerprint
from resumeanalysis_legacy.data.download import sha256_file
from resumeanalysis_legacy.experiments.rerank_evaluation import (
    pair_metrics,
    query_ranking_metrics,
    validation_threshold,
)

SPLITS = {"train", "validation", "test"}


class Source(InputModel):
    data_origin: Literal["consented_real", "licensed_public_real"]
    source_url: StrictStr = Field(min_length=1)
    license: StrictStr = Field(min_length=1)
    permission_basis: StrictStr = Field(min_length=1)
    source_document_id: StrictStr = Field(min_length=1)
    source_sha256: StrictStr | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_meaningful_provenance(self):
        if any(
            not getattr(self, name).strip()
            for name in ("source_url", "license", "permission_basis", "source_document_id")
        ):
            raise ValueError("Source and permission declarations cannot be blank")
        return self


class Judgment(InputModel):
    grade: StrictInt = Field(ge=0, le=2)
    abstain_reason: None
    assessor_ids: list[StrictStr] = Field(min_length=2)
    adjudicated: StrictBool
    evidence_notes: StrictStr = ""
    preference_status: Literal["compatible", "conflict", "unknown"] = "unknown"

    @model_validator(mode="after")
    def require_adjudication(self):
        if not self.adjudicated or len(set(self.assessor_ids)) != len(self.assessor_ids):
            raise ValueError("Judgments must be adjudicated by at least two distinct assessors")
        if any(not assessor.strip() for assessor in self.assessor_ids):
            raise ValueError("Assessor IDs must be nonempty")
        return self


class JudgedCandidate(InputModel):
    vacancy_id: StrictStr = Field(min_length=1)
    vacancy: Vacancy
    judgment: Judgment
    provenance: Source


class JudgedGroup(InputModel):
    schema_version: StrictInt
    query_id: StrictStr = Field(min_length=1)
    profile_id: StrictStr = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    language: Literal["en", "ru", "mixed", "other"]
    profile: Profile
    provenance: Source
    label_origin: Literal["independent_human"]
    judged_pool_complete: StrictBool
    candidates: list[JudgedCandidate] = Field(min_length=2)

    @model_validator(mode="after")
    def require_complete_pool(self):
        if self.schema_version != 1 or not self.judged_pool_complete:
            raise ValueError("Version 1 and fully judged pools are required")
        ids = [candidate.vacancy_id for candidate in self.candidates]
        text_keys = [
            fingerprint(
                candidate.vacancy.description.strip()
                or json.dumps(candidate.vacancy.model_dump(), sort_keys=True)
            )
            for candidate in self.candidates
        ]
        source_keys = [
            (candidate.provenance.source_url, candidate.provenance.source_document_id)
            for candidate in self.candidates
        ]
        if (
            len(set(ids)) != len(ids)
            or len(set(text_keys)) != len(text_keys)
            or len(set(source_keys)) != len(source_keys)
        ):
            raise ValueError("Duplicate vacancy ID, text or source in judged pool")
        return self


class PredictionCandidate(InputModel):
    vacancy_id: StrictStr = Field(min_length=1)
    score: float
    retrieval_score: float | None = None
    decision: StrictBool | None = None
    evidence_status: Literal["eligible", "needs_review", "blocked"] | None = None

    @model_validator(mode="before")
    @classmethod
    def numeric_scores(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Each scored candidate must be a JSON object")
        for name in ("score", "retrieval_score"):
            number = value.get(name)
            if number is not None and (
                type(number) not in {int, float} or not math.isfinite(number)
            ):
                raise ValueError("Scores must be finite JSON numbers, not strings or booleans")
        return value


class PredictionGroup(InputModel):
    schema_version: StrictInt
    query_id: StrictStr = Field(min_length=1)
    model_identity: StrictStr = Field(min_length=1)
    config_identity: StrictStr = Field(min_length=1)
    candidates: list[PredictionCandidate] = Field(min_length=2)
    ranked_vacancy_ids: list[StrictStr] | None = None
    retrieved_vacancy_ids: list[StrictStr] | None = None

    @model_validator(mode="after")
    def known_unique_ids(self):
        if self.schema_version != 1:
            raise ValueError("Unknown prediction schema")
        ids = [candidate.vacancy_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate scored vacancy")
        for selected in (self.ranked_vacancy_ids, self.retrieved_vacancy_ids):
            if selected is not None and (
                len(set(selected)) != len(selected) or not set(selected) <= set(ids)
            ):
                raise ValueError(
                    "Ranked/retrieved IDs must be a unique subset of the complete pool"
                )
        if self.ranked_vacancy_ids is not None and self.retrieved_vacancy_ids is not None:
            if not set(self.ranked_vacancy_ids) <= set(self.retrieved_vacancy_ids):
                raise ValueError("Ranked IDs must belong to the retrieved subset")
        if self.retrieved_vacancy_ids is not None and self.ranked_vacancy_ids is None:
            raise ValueError(
                "Logged retrieval requires the actual ranked order, including empty results"
            )
        if any(c.retrieval_score is None for c in self.candidates) and any(
            c.retrieval_score is not None for c in self.candidates
        ):
            raise ValueError("Retrieval scores must cover the entire declared pool or be absent")
        if any(c.decision is None for c in self.candidates) and any(
            c.decision is not None for c in self.candidates
        ):
            raise ValueError("Policy decisions must cover the entire declared pool or be absent")
        return self


def _jsonl(path, model):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(loads(line), strict=True))
        except ValidationError as exc:
            # Annotation files can contain private resumes. Preserve actionable
            # schema messages without putting their input values into CLI logs.
            details = "; ".join(error["msg"] for error in exc.errors(include_input=False))
            raise ValueError(f"Invalid {model.__name__} at line {number}: {details}") from None
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid {model.__name__} at line {number}: {exc}") from exc
    if not rows:
        raise ValueError("Empty evaluation file")
    return rows


def load_judged_groups(path, *, split=None):
    """Validate all splits together, including IDs, source IDs and normalized text.

    Near-duplicate/source-family auditing remains a collection-time manual step.
    """
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown requested split")
    groups = _jsonl(path, JudgedGroup)
    seen_queries, seen_entities, seen_text, seen_sources = set(), {}, {}, {}
    seen_profile_queries = set()

    def check_entity(kind, identity, content, source, partition):
        content_hash = digest(content.model_dump())
        previous = seen_entities.setdefault((kind, identity), (partition, content_hash))
        if previous != (partition, content_hash):
            raise ValueError("Human entity identity/split leakage or inconsistent content")
        text = content.resume_text if kind == "profile" else content.description
        if not text.strip():
            text = json.dumps(content.model_dump(), sort_keys=True, ensure_ascii=False)
        key = (kind, fingerprint(text))
        if seen_text.setdefault(key, partition) != partition:
            raise ValueError("Human normalized text leaks across splits")
        source_key = (kind, source.source_url, source.source_document_id)
        if seen_sources.setdefault(source_key, (partition, content_hash)) != (
            partition,
            content_hash,
        ):
            raise ValueError("Human source document identity leaks or changes content")
        if source.source_sha256 is not None:
            # The source hash may describe original pre-redaction bytes. Its
            # declared identity must still not cross a cold-entity split.
            if (
                seen_sources.setdefault((kind, "sha256", source.source_sha256), partition)
                != partition
            ):
                raise ValueError("Human source hash leaks across splits")

    for group in groups:
        if group.query_id in seen_queries:
            raise ValueError("Duplicate human query ID")
        seen_queries.add(group.query_id)
        check_entity("profile", group.profile_id, group.profile, group.provenance, group.split)
        # One canonical fixed pool per profile: repeated query IDs are not the
        # only way to accidentally double-weight a candidate in macro averages.
        profile_text = group.profile.resume_text.strip() or json.dumps(
            group.profile.model_dump(), sort_keys=True, ensure_ascii=False
        )
        profile_key = fingerprint(profile_text)
        if profile_key in seen_profile_queries:
            raise ValueError(
                "Duplicate profile query content; combine its judged vacancies into one pool"
            )
        seen_profile_queries.add(profile_key)
        for candidate in group.candidates:
            check_entity(
                "vacancy",
                candidate.vacancy_id,
                candidate.vacancy,
                candidate.provenance,
                group.split,
            )
    selected = [group for group in groups if split is None or group.split == split]
    if not selected:
        raise ValueError("No fully judged pools for requested split")
    return selected


def _aligned_predictions(path, groups):
    predictions = _jsonl(path, PredictionGroup)
    if len({p.query_id for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction query ID")
    if {p.query_id for p in predictions} != {g.query_id for g in groups}:
        raise ValueError("Prediction query IDs must exactly cover the requested split")
    indexed = {p.query_id: p for p in predictions}
    aligned = []
    for group in groups:
        prediction = indexed[group.query_id]
        if [c.vacancy_id for c in prediction.candidates] != [
            c.vacancy_id for c in group.candidates
        ]:
            raise ValueError("Predictions must preserve the complete original vacancy order")
        aligned.append(prediction)
    identities = {(p.model_identity, p.config_identity) for p in aligned}
    if len(identities) != 1:
        raise ValueError("One immutable model/config identity is required per evaluation")
    return aligned


def _write_new(path, value):
    # Exclusive creation makes freezing/reports fail on accidental overwrites.
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _positive_integers(values):
    if (
        not values
        or any(type(k) is not int or k < 1 for k in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Distinct positive integer cutoffs are required")
    return list(values)


def freeze_validation(
    annotations, predictions, output, *, k_values=(1, 5, 10), preselect_values=(1, 5, 10, 20)
):
    """Select a scalar cutoff using validation only and freeze evaluation choices."""
    groups = load_judged_groups(annotations, split="validation")
    scores = _aligned_predictions(predictions, groups)
    labels = [int(c.judgment.grade > 0) for g in groups for c in g.candidates]
    values = [c.score for p in scores for c in p.candidates]
    threshold = validation_threshold(labels, values)
    artifact = {
        "schema_version": 1,
        "source_split": "validation",
        "validation_groups_sha256": digest([g.model_dump() for g in groups]),
        "validation_predictions_sha256": sha256_file(predictions),
        "model_identity": scores[0].model_identity,
        "config_identity": scores[0].config_identity,
        "threshold": threshold,
        "selection": "Validation grade>=1 macro-F1, then precision, then higher cutoff",
        "k_values": _positive_integers(k_values),
        "preselect_values": _positive_integers(preselect_values),
        "test_labels_used": False,
        "validation_metrics": pair_metrics(labels, values, threshold),
        "limitations": "Artifact records supplied identities, not cryptographic proof of prior model choice or truthful human/source declarations.",
    }
    _write_new(output, artifact)
    return artifact


def _mean_ranking(rows, k_values):
    result = {}
    for k in k_values:
        result[str(k)] = {}
        for name in ("precision_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k"):
            values = [row[str(k)][name] for row in rows if row[str(k)][name] is not None]
            result[str(k)][name] = statistics.mean(values) if values else None
            result[str(k)][name + "_queries"] = len(values)
    return result


def _summary(groups, predictions, threshold, k_values, preselect_values):
    labels, grades, scores, decisions, rows, grade2_rows, sensitivity = [], [], [], [], [], [], {}
    policy_available = all(c.decision is not None for p in predictions for c in p.candidates)
    per_query = []
    policy_audit = Counter()
    for group, prediction in zip(groups, predictions, strict=True):
        local_grades = [c.judgment.grade for c in group.candidates]
        local_scores = [c.score for c in prediction.candidates]
        local_ids = {c.vacancy_id: i for i, c in enumerate(group.candidates)}
        raw_order = sorted(range(len(local_scores)), key=lambda i: (-local_scores[i], i))
        order = (
            raw_order
            if prediction.ranked_vacancy_ids is None
            else [local_ids[i] for i in prediction.ranked_vacancy_ids]
        )
        metrics = {str(k): query_ranking_metrics(local_grades, order, k) for k in k_values}
        rows.append(metrics)
        grade2_rows.append(
            {
                str(k): query_ranking_metrics([int(g == 2) for g in local_grades], order, k)
                for k in k_values
            }
        )
        retrieved = prediction.retrieved_vacancy_ids
        relevant = sum(g > 0 for g in local_grades)
        actual_recall = (
            None
            if retrieved is None or not relevant
            else sum(local_grades[local_ids[i]] > 0 for i in retrieved) / relevant
        )
        per_query.append(
            {
                "query_id": group.query_id,
                "language": group.language,
                "pool_size": len(local_grades),
                "metrics": metrics,
                "ranking_source": "actual_logged_order"
                if prediction.ranked_vacancy_ids is not None
                else "complete_pool_score_order",
                "actual_retrieved_pool_recall": actual_recall,
            }
        )
        if prediction.candidates[0].retrieval_score is not None:
            retrieval_order = sorted(
                range(len(local_scores)),
                key=lambda i: (-prediction.candidates[i].retrieval_score, i),
            )
            for k in preselect_values:
                recall = (
                    sum(local_grades[i] > 0 for i in retrieval_order[:k]) / relevant
                    if relevant
                    else None
                )
                sensitivity.setdefault(str(k), []).append(recall)
        labels.extend(int(g > 0) for g in local_grades)
        grades.extend(local_grades)
        scores.extend(local_scores)
        if policy_available:
            decisions.extend(c.decision for c in prediction.candidates)
            for candidate, predicted in zip(group.candidates, prediction.candidates, strict=True):
                policy_audit["judged_pairs"] += 1
                policy_audit["approved"] += int(predicted.decision)
                policy_audit["approved_with_reported_block"] += int(
                    predicted.decision and predicted.evidence_status == "blocked"
                )
                policy_audit["approved_with_reported_unknown_evidence"] += int(
                    predicted.decision and predicted.evidence_status == "needs_review"
                )
                policy_audit["approved_with_human_preference_conflict"] += int(
                    predicted.decision and candidate.judgment.preference_status == "conflict"
                )
                policy_audit["approved_with_human_preference_unknown"] += int(
                    predicted.decision and candidate.judgment.preference_status == "unknown"
                )
    return {
        "queries": len(groups),
        "pairs": len(labels),
        "grade_counts": dict(Counter(grades)),
        "pair_metrics": pair_metrics(labels, scores, threshold),
        "grade_2_only_pair_metrics": pair_metrics([int(g == 2) for g in grades], scores, threshold),
        "policy_pair_metrics": pair_metrics(labels, scores, threshold, decisions=decisions)
        if policy_available
        else None,
        "policy_audit_counts": dict(policy_audit) if policy_available else None,
        "ranking": _mean_ranking(rows, k_values),
        "grade_2_only_ranking": _mean_ranking(grade2_rows, k_values),
        "retrieval_preselect_recall": {
            key: {
                "mean": statistics.mean([v for v in values if v is not None])
                if any(v is not None for v in values)
                else None,
                "positive_queries": sum(v is not None for v in values),
                "measured_queries": len(values),
            }
            for key, values in sensitivity.items()
        },
        "evidence_status_counts": dict(
            Counter(c.evidence_status or "not_reported" for p in predictions for c in p.candidates)
        ),
        "per_query": per_query,
    }


def evaluate_test(annotations, predictions, frozen, validation_predictions, output):
    """Score held-out human pools with an unchanged validation-frozen artifact."""
    artifact = loads(Path(frozen).read_text())
    expected = {
        "schema_version",
        "source_split",
        "validation_groups_sha256",
        "validation_predictions_sha256",
        "model_identity",
        "config_identity",
        "threshold",
        "selection",
        "k_values",
        "preselect_values",
        "test_labels_used",
        "validation_metrics",
        "limitations",
    }
    if (
        not isinstance(artifact, dict)
        or set(artifact) != expected
        or artifact["schema_version"] != 1
        or artifact["source_split"] != "validation"
        or artifact["test_labels_used"] is not False
    ):
        raise ValueError("A validation-frozen configuration artifact is required")
    validation = load_judged_groups(annotations, split="validation")
    val_predictions = _aligned_predictions(validation_predictions, validation)
    if artifact["validation_groups_sha256"] != digest(
        [g.model_dump() for g in validation]
    ) or artifact["validation_predictions_sha256"] != sha256_file(validation_predictions):
        raise ValueError("Validation data/predictions changed after configuration freeze")
    threshold = artifact["threshold"]
    if type(threshold) not in {int, float} or not math.isfinite(threshold):
        raise ValueError("Finite frozen threshold required")
    val_labels = [int(c.judgment.grade > 0) for g in validation for c in g.candidates]
    val_scores = [c.score for p in val_predictions for c in p.candidates]
    if threshold != validation_threshold(val_labels, val_scores):
        raise ValueError("Frozen threshold differs from recorded validation selection")
    groups = load_judged_groups(annotations, split="test")
    scores = _aligned_predictions(predictions, groups)
    for prediction in [*val_predictions, *scores]:
        if (prediction.model_identity, prediction.config_identity) != (
            artifact["model_identity"],
            artifact["config_identity"],
        ):
            raise ValueError("Model/config identity differs from validation freeze")
    k_values = _positive_integers(artifact["k_values"])
    preselect_values = _positive_integers(artifact["preselect_values"])
    report = {
        "schema_version": 1,
        "split": "test",
        "model_identity": artifact["model_identity"],
        "config_identity": artifact["config_identity"],
        "threshold": threshold,
        "annotations_sha256": sha256_file(annotations),
        "predictions_sha256": sha256_file(predictions),
        "frozen_configuration_sha256": sha256_file(frozen),
        "overall": _summary(groups, scores, threshold, k_values, preselect_values),
        "languages": {},
        "limitations": "All metrics are conditional on complete supplied judged pools. Null all-negative-query recall/NDCG/MRR are excluded from those macro averages; precision includes them. Denominators retain blocked and unretrieved relevant jobs. Grade-2 sensitivity reuses the grade>=1 validation threshold. Source permissions, assessor independence, near-duplicates, source-family boundaries, annotation agreement and configuration chronology require manual audit. No calibrated hiring-fit or population generalization claim.",
    }
    for language in sorted({g.language for g in groups}):
        selected = [(g, p) for g, p in zip(groups, scores, strict=True) if g.language == language]
        report["languages"][language] = _summary(
            [g for g, _ in selected],
            [p for _, p in selected],
            threshold,
            k_values,
            preselect_values,
        )
    _write_new(output, report)
    return report
