"""Auditable pair and judged-pool ranking evaluation for retrieval experiments.

No cross-product labels are invented. Thresholds are fitted on validation only;
test data may be scored only after a separate threshold artifact is written.
"""

import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from resumeanalysis.data.adapters import digest
from resumeanalysis.data.download import sha256_file
from resumeanalysis.experiments.partial_profiles import load_splits
from resumeanalysis.jsonio import write_json

METHODS = ("mlp", "rule", "cosine", "cross_encoder", "retrieve_rerank")
STATUS_ORDER = {"eligible": 0, "needs_review": 1, "blocked": 2}


@dataclass(frozen=True)
class CandidateGroup:
    query_id: str
    indices: tuple[int, ...]


def candidate_groups(records, *, minimum_size=2):
    """Group existing judgments by candidate, checking profile and vacancy identity."""
    if type(minimum_size) is not int or minimum_size < 1:
        raise ValueError("minimum_size must be a positive integer")
    grouped, profiles, vacancies, pairs = defaultdict(list), {}, {}, set()
    for index, record in enumerate(records):
        if not record.candidate_id or not record.vacancy_id:
            raise ValueError("Ranking requires nonempty candidate and vacancy IDs")
        profile = digest(record.profile.model_dump())
        vacancy = digest(record.vacancy.model_dump())
        if profiles.setdefault(record.candidate_id, profile) != profile:
            raise ValueError("A candidate ID refers to inconsistent profiles")
        if vacancies.setdefault(record.vacancy_id, vacancy) != vacancy:
            raise ValueError("A vacancy ID refers to inconsistent vacancy content")
        pair = record.candidate_id, record.vacancy_id
        if pair in pairs:
            raise ValueError("Duplicate judged candidate-vacancy pair")
        pairs.add(pair)
        grouped[record.candidate_id].append(index)
    return [
        CandidateGroup(key, tuple(indices))
        for key, indices in grouped.items()
        if len(indices) >= minimum_size
    ]


def _scores(labels, scores):
    if not labels or len(labels) != len(scores):
        raise ValueError("Nonempty aligned labels and scores required")
    if any(type(label) is not int or label not in {0, 1} for label in labels):
        raise ValueError("Binary integer labels required")
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(labels),) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite scalars")
    return values


def pair_metrics(labels, scores, threshold, *, decisions=None):
    scores = _scores(labels, scores)
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    predictions = scores >= threshold if decisions is None else np.asarray(decisions)
    if predictions.shape != (len(labels),) or any(
        type(v) not in {bool, np.bool_} for v in predictions
    ):
        raise ValueError("One boolean decision is required per pair")
    return {
        "count": len(labels),
        "positive_count": sum(labels),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "score_semantics": "Ranking/uncalibrated relevance score, never a validated job-fit probability",
    }


def validation_threshold(labels, scores):
    """Validation-only scalar fit; ties favor precision, then a higher cutoff."""
    values = _scores(labels, scores)
    if len(set(labels)) != 2:
        raise ValueError("Validation threshold selection requires both classes")
    unique = np.unique(values)
    candidates = [float(unique[0]), float(np.nextafter(unique[-1], math.inf))]
    candidates.extend(float(a / 2 + b / 2) for a, b in zip(unique, unique[1:]))
    return max(
        candidates,
        key=lambda cutoff: (
            pair_metrics(labels, scores, cutoff)["macro_f1"],
            pair_metrics(labels, scores, cutoff)["precision"],
            cutoff,
        ),
    )


def query_ranking_metrics(grades, order, k):
    """Grade>0 is relevant. Missing retrieved slots receive no metric credit.

    Precision denominator is min(k, complete judged pool size), not returned size.
    Recall/NDCG/MRR are undefined for all-negative queries and reported as null.
    """
    if type(k) is not int or k < 1 or not grades:
        raise ValueError("A positive integer k and nonempty judged pool are required")
    if any(type(g) is not int or g < 0 or g > 2 for g in grades):
        raise ValueError("Relevance grades must be integers in [0, 2]")
    if len(set(order)) != len(order) or any(
        type(i) is not int or not 0 <= i < len(grades) for i in order
    ):
        raise ValueError("Rank order must contain unique pool indices")
    cutoff = min(k, len(grades))
    selected = list(order[:cutoff])
    relevant = sum(g > 0 for g in grades)
    hits = sum(grades[i] > 0 for i in selected)

    def dcg(values):
        return sum(
            (2**grade - 1) / math.log2(position + 2) for position, grade in enumerate(values)
        )

    ideal = dcg(sorted(grades, reverse=True)[:cutoff])
    reciprocal = next((1 / (i + 1) for i, item in enumerate(selected) if grades[item] > 0), 0.0)
    return {
        "pool_size": len(grades),
        "returned": len(selected),
        "relevant_in_pool": relevant,
        "precision_at_k": hits / cutoff,
        "recall_at_k": hits / relevant if relevant else None,
        "ndcg_at_k": dcg([grades[i] for i in selected]) / ideal if ideal else None,
        "mrr_at_k": reciprocal if relevant else None,
    }


def grouped_ranking_report(
    records,
    scores,
    *,
    k_values=(1, 2, 5, 10),
    retrieval_scores=None,
    preselect_k=None,
    statuses=None,
    explicit_orders=None,
):
    _scores([r.label for r in records], scores)
    if retrieval_scores is not None:
        _scores([r.label for r in records], retrieval_scores)
        if type(preselect_k) is not int or preselect_k < 1:
            raise ValueError("Explicit positive preselect_k required")
    if statuses is not None and (
        len(statuses) != len(records) or any(s not in STATUS_ORDER for s in statuses)
    ):
        raise ValueError("One declared policy evidence status is required per pair")
    groups = candidate_groups(records)
    if not groups:
        return {
            "status": "unavailable",
            "reason": "No candidate has at least two judged vacancies",
            "queries": 0,
        }
    rows, preselect_recalls = [], []
    for group in groups:
        indices = list(group.indices)
        candidates = indices
        if retrieval_scores is not None:
            candidates = sorted(indices, key=lambda i: (-retrieval_scores[i], i))[:preselect_k]
            positives = sum(records[i].label for i in indices)
            if positives:
                preselect_recalls.append(sum(records[i].label for i in candidates) / positives)
        order = (
            sorted(
                candidates,
                key=lambda i: (
                    STATUS_ORDER[statuses[i]] if statuses is not None else 0,
                    -scores[i],
                    i,
                ),
            )
            if explicit_orders is None
            else explicit_orders[group.query_id]
        )
        if len(set(order)) != len(order) or not set(order) <= set(indices):
            raise ValueError("Explicit ranking contains duplicate or foreign judged candidates")
        local = {index: position for position, index in enumerate(indices)}
        grades = [records[i].label for i in indices]
        rows.append(
            {
                "query_id": group.query_id,
                "metrics": {
                    str(k): query_ranking_metrics(grades, [local[i] for i in order], k)
                    for k in k_values
                },
            }
        )
    means = {}
    for k in k_values:
        means[str(k)] = {}
        for metric in ("precision_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k"):
            values = [
                row["metrics"][str(k)][metric]
                for row in rows
                if row["metrics"][str(k)][metric] is not None
            ]
            means[str(k)][metric] = statistics.mean(values) if values else None
            means[str(k)][metric + "_queries"] = len(values)
    return {
        "status": "measured_on_existing_judged_pools",
        "queries": len(groups),
        "pool_size_distribution": dict(Counter(len(g.indices) for g in groups)),
        "singleton_candidates_excluded": len(candidate_groups(records, minimum_size=1))
        - len(groups),
        "means": means,
        "preselect_k": preselect_k,
        "preselect_recall": statistics.mean(preselect_recalls) if preselect_recalls else None,
        "positive_queries": len(preselect_recalls)
        if retrieval_scores is not None
        else sum(any(records[i].label for i in g.indices) for g in groups),
        "limitations": "Metrics cover only the supplied judged vacancies for each candidate, not the full vacancy catalog. Unjudged cross-product pairs are never assumed negative. Singleton pools are excluded. Null all-negative-query recall/NDCG/MRR do not enter those macro averages; precision includes them.",
        "per_query": rows,
    }


def evaluation_inputs(data):
    root = Path(data)
    splits = load_splits(root)
    manifest = {
        "source_hashes": {name: sha256_file(root / (name + ".jsonl")) for name in splits},
        "record_hashes": {
            name: [digest(r.model_dump()) for r in rows] for name, rows in splits.items()
        },
        "partition_integrity": "All original train/validation/test IDs and normalized text checked for cross-split leakage",
        "label_warning": "Current repository pairs are synthetic resumes/vacancies with heuristic/source labels; metrics do not establish actual hiring relevance.",
    }
    return splits, manifest


def _latency(values):
    return {
        "total_seconds": sum(values),
        "mean_seconds": statistics.mean(values) if values else None,
        "p50_seconds": float(np.quantile(values, 0.5)) if values else None,
        "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
        "measurement_batches": len(values),
    }


def _rss():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


def freeze_threshold(
    path, *, method, threshold, source_hash, validation_scores, identity, selection
):
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            "Threshold artifacts are immutable; choose a fresh experiment directory"
        )
    value = {
        "format_version": 1,
        "method": method,
        "threshold": float(threshold),
        "source_split": "validation",
        "source_sha256": source_hash,
        "validation_scores_sha256": digest(validation_scores),
        "runtime_identity": identity,
        "selection": selection,
        "test_labels_used": False,
    }
    write_json(path, value)
    return value


def benchmark_worker(
    method,
    *,
    data,
    output,
    e5_path,
    model_path,
    checkpoint,
    batch_size=8,
    max_length=512,
    threads=2,
):
    """Run exactly one method in a fresh process; no learning or checkpoint writes."""
    if method not in METHODS:
        raise ValueError("Unknown benchmark method")
    import torch

    torch.set_num_threads(threads)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Worker output directory must be new")
    output.mkdir(parents=True)
    splits, source = evaluation_inputs(data)
    initialized = time.perf_counter()
    details = {}
    encoder = cross_encoder = predictor = builder = retriever = None
    if method in {"mlp", "rule", "cosine", "retrieve_rerank"}:
        from resumeanalysis.embeddings.encoder import TextEncoder

        encoder = TextEncoder(e5_path, cache_dir=output / "legacy-e5-cache")
    if method == "mlp":
        from resumeanalysis.inference.predictor import Predictor

        metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if metadata.get("data_sha256") != source["source_hashes"]:
            raise ValueError("MLP checkpoint was fitted with different dataset partitions")
        predictor = Predictor(checkpoint, encoder)
        identity = "mlp:" + sha256_file(checkpoint) + ":" + encoder.identity
    elif method == "rule":
        from resumeanalysis.features.feature_builder import FeatureEngineering

        builder = FeatureEngineering(encoder)
        identity = "legacy_rule:" + encoder.identity + ":" + builder.extractor_identity
    elif method == "cosine":
        from resumeanalysis.rerank.retrieval import ExactRetriever

        retriever = ExactRetriever(encoder)
        identity = retriever.identity
    if method in {"cross_encoder", "retrieve_rerank"}:
        from resumeanalysis.rerank.config import RerankConfig
        from resumeanalysis.rerank.evidence import SERIALIZATION_VERSION
        from resumeanalysis.rerank.model import CrossEncoder

        cross_encoder = CrossEncoder(
            RerankConfig(model_path=str(model_path), batch_size=batch_size, max_length=max_length)
        )
        identity = "cross-encoder-pairs:" + digest(
            {
                "cross_encoder": cross_encoder.identity,
                "serialization": SERIALIZATION_VERSION,
            }
        )
        if method == "retrieve_rerank":
            from resumeanalysis.rerank.predictor import RerankPredictor

            predictor = RerankPredictor(
                encoder=encoder, reranker=cross_encoder, decision_logit_threshold=0.0
            )
            identity = "retrieval-cross-encoder:" + digest(
                {
                    "cross_encoder": cross_encoder.identity,
                    "retrieval": predictor.retriever.identity,
                    "serialization": SERIALIZATION_VERSION,
                }
            )
    initialization_seconds = time.perf_counter() - initialized

    def score_records(records):
        scores, decisions, statuses, latencies, diagnostic = [None] * len(records), [], [], [], []
        if method == "cross_encoder":
            from resumeanalysis.rerank.evidence import cross_encoder_texts

            started = time.perf_counter()
            scores = cross_encoder.score_pairs(
                [cross_encoder_texts(r.profile, r.vacancy) for r in records]
            )
            latencies.append(time.perf_counter() - started)
            diagnostic = list(cross_encoder.last_diagnostics)
        elif method == "retrieve_rerank":
            for index, record in enumerate(records):
                started = time.perf_counter()
                result = predictor.analyze_pair(record.profile, record.vacancy)
                latencies.append(time.perf_counter() - started)
                scores[index] = result.rerank_score
                decisions.append(result.analysis.relevant)
                statuses.append(result.evidence_status)
                diagnostic.append(result.diagnostics)
        elif method == "cosine":
            for group in candidate_groups(records, minimum_size=1):
                selected = [records[i] for i in group.indices]
                started = time.perf_counter()
                ranked = retriever.retrieve(
                    selected[0].profile, [r.vacancy for r in selected], preselect_k=len(selected)
                )
                for item in ranked:
                    scores[group.indices[item.index]] = item.cosine_score
                latencies.append(time.perf_counter() - started)
        else:
            for index, record in enumerate(records):
                started = time.perf_counter()
                if method == "mlp":
                    analysis = predictor.predict(record.profile, record.vacancy)
                    scores[index] = analysis.relevance
                    decisions.append(analysis.relevant)
                else:
                    from resumeanalysis.training.evaluate import rule_scores

                    detail = builder.build(record.profile, record.vacancy)
                    scores[index] = float(rule_scores(detail.vector[None, :])[0])
                latencies.append(time.perf_counter() - started)
        if any(s is None or not math.isfinite(s) for s in scores):
            raise ValueError(
                "Missing/nonfinite benchmark score; empty pairs need a separate cohort"
            )
        return list(map(float, scores)), decisions, statuses, latencies, diagnostic

    # Warmup sees a validation example only. Test has not been passed to any model.
    score_records(splits["validation"][:1])
    val_scores, _, val_statuses, val_latencies, val_diagnostics = score_records(
        splits["validation"]
    )
    if method == "mlp":
        threshold = float(metadata["threshold"])
        selection = "Existing checkpoint validation-selected threshold retained unchanged"
    elif method == "rule":
        from resumeanalysis.training.evaluate import choose_threshold

        threshold = choose_threshold([r.label for r in splits["validation"]], val_scores)
        selection = "Existing rule-baseline 0.05..0.95 validation grid, same formula and tie-break"
    else:
        threshold = validation_threshold([r.label for r in splits["validation"]], val_scores)
        selection = "Observed validation cutoffs; maximize macro-F1, then precision, then cutoff"
    frozen = freeze_threshold(
        output / "threshold.json",
        method=method,
        threshold=threshold,
        source_hash=source["source_hashes"]["validation"],
        validation_scores=val_scores,
        identity=identity,
        selection=selection,
    )
    # Persisted threshold evidence exists before any test model inference.
    threshold_sha = sha256_file(output / "threshold.json")
    if method == "retrieve_rerank":
        predictor = RerankPredictor(
            encoder=encoder, reranker=cross_encoder, decision_logit_threshold=threshold
        )
        details["selected_policy_identity"] = predictor.identity
    test_scores, test_decisions, test_statuses, latencies, diagnostics = score_records(
        splits["test"]
    )
    production_orders, production_rank_latency = {}, {}
    if method == "retrieve_rerank":
        for preselect in (1, 2, 5, 10, 20):
            orders, times = {}, []
            for group in candidate_groups(splits["test"]):
                pool = [splits["test"][i] for i in group.indices]
                started = time.perf_counter()
                ranked = predictor.rank(
                    pool[0].profile,
                    [r.vacancy for r in pool],
                    preselect_k=preselect,
                    top_k=min(preselect, 10),
                )
                times.append(time.perf_counter() - started)
                orders[group.query_id] = [group.indices[item.index] for item in ranked]
            production_orders[str(preselect)] = orders
            production_rank_latency[str(preselect)] = _latency(times)
    if sha256_file(output / "threshold.json") != threshold_sha:
        raise ValueError("Frozen threshold changed during test evaluation")
    for split in source["source_hashes"]:
        if sha256_file(Path(data) / (split + ".jsonl")) != source["source_hashes"][split]:
            raise ValueError("Dataset changed during evaluation")
    report = {
        "format_version": 1,
        "method": method,
        "source": source,
        "identity": identity,
        "threshold": frozen,
        "threshold_sha256": threshold_sha,
        "initialization_seconds": initialization_seconds,
        "validation_pair_metrics": pair_metrics(
            [r.label for r in splits["validation"]], val_scores, threshold
        ),
        "test_pair_metrics": pair_metrics(
            [r.label for r in splits["test"]], test_scores, threshold
        ),
        "test_policy_metrics": pair_metrics(
            [r.label for r in splits["test"]], test_scores, threshold, decisions=test_decisions
        )
        if method in {"mlp", "retrieve_rerank"}
        else None,
        "validation_scores": val_scores,
        "test_scores": test_scores,
        "validation_statuses": val_statuses if method == "retrieve_rerank" else None,
        "test_statuses": test_statuses if method == "retrieve_rerank" else None,
        "policy_status_counts": dict(Counter(test_statuses)) if test_statuses else None,
        "production_ranking_orders": production_orders,
        "production_ranking_latency": production_rank_latency,
        "latency": {
            "validation": _latency(val_latencies),
            "test": _latency(latencies),
            "test_seconds_per_pair": sum(latencies) / len(splits["test"]),
        },
        "process_peak_rss_bytes": _rss(),
        "measurement_scope": "Fresh process per method; initialization excludes source loading and initial imports. One validation-only warmup. Inference latency includes feature/evidence/policy/caching work of this method; Each method starts with a fresh private E5 disk cache and private retrieval memory cache; the validation warmup primes only one validation pair. Original entity-disjoint test inputs are cold on first inference; repeated profiles within the split may reuse embeddings. Group-ranking timing runs afterward and is warm. RSS is process-lifetime high-water including Python/imports/models/source, not standalone model allocation. Cross-encoder-only scores all pairs in one batched call; rerank scores real candidate pools, mostly size one/two; latency protocols therefore differ and do not establish an architecture speedup.",
        "diagnostics": {"validation": val_diagnostics, "test": diagnostics},
        "limitations": "Synthetic/heuristic pair labels, not independent employer relevance. Policy metrics intentionally include conservative evidence/preference gates; raw semantic classifier metrics are separate. No downstream model retraining.",
        "details": details,
        "runtime": {
            "torch_threads": threads,
            "batch_size": batch_size,
            "max_length": max_length,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
    }
    write_json(output / "report.json", report)
    return report


def combine_benchmarks(
    data, worker_reports, *, k_values=(1, 2, 5, 10), preselect_values=(1, 2, 5, 10, 20)
):
    splits, source = evaluation_inputs(data)
    if set(worker_reports) != set(METHODS):
        raise ValueError("Exactly the five declared baseline methods are required")
    for method, report in worker_reports.items():
        if report["method"] != method or report["source"] != source:
            raise ValueError("Methods used different ordered evaluation records or partitions")
    records = splits["test"]
    ranking = {
        method: grouped_ranking_report(records, report["test_scores"], k_values=k_values)
        for method, report in worker_reports.items()
        if method != "retrieve_rerank"
    }
    sensitivity = {}
    for preselect in preselect_values:
        sensitivity[str(preselect)] = {
            "semantic": grouped_ranking_report(
                records,
                worker_reports["retrieve_rerank"]["test_scores"],
                k_values=k_values,
                retrieval_scores=worker_reports["cosine"]["test_scores"],
                preselect_k=preselect,
            ),
            "with_structured_policy": grouped_ranking_report(
                records,
                worker_reports["retrieve_rerank"]["test_scores"],
                k_values=k_values,
                explicit_orders=worker_reports["retrieve_rerank"]["production_ranking_orders"][
                    str(preselect)
                ],
            ),
        }
    return {
        "format_version": 1,
        "source": source,
        "workers": worker_reports,
        "ranking": ranking,
        "preselect_sensitivity": sensitivity,
        "claims": "Exploratory synthetic-label measurements only. Natural judged pools are small and cannot validate catalog-scale retrieval or hiring relevance. No unknown pair was invented as a negative, and no final test label selected a threshold or model.",
    }


def benchmark_summary(report):
    """Compact publishable measurements; original texts and pair scores stay outside Git."""
    workers = report["workers"]
    return {
        "format_version": 1,
        "source_hashes": report["source"]["source_hashes"],
        "record_counts": {
            split: len(records) for split, records in report["source"]["record_hashes"].items()
        },
        "label_warning": report["source"]["label_warning"],
        "methods": {
            name: {
                key: worker[key]
                for key in (
                    "identity",
                    "threshold",
                    "initialization_seconds",
                    "validation_pair_metrics",
                    "test_pair_metrics",
                    "test_policy_metrics",
                    "policy_status_counts",
                    "latency",
                    "process_peak_rss_bytes",
                    "runtime",
                    "measurement_scope",
                    "limitations",
                    "production_ranking_latency",
                    "details",
                )
            }
            for name, worker in workers.items()
        },
        "ranking": {
            name: {key: value for key, value in ranking.items() if key != "per_query"}
            for name, ranking in report["ranking"].items()
        },
        "preselect_sensitivity": {
            k: {
                name: {key: value for key, value in ranking.items() if key != "per_query"}
                for name, ranking in variants.items()
            }
            for k, variants in report["preselect_sensitivity"].items()
        },
        "claims": report["claims"],
    }
