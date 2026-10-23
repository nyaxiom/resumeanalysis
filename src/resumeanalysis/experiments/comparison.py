"""Shared examples and decision metrics; partial views remain unlabeled diagnostics."""

import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from resumeanalysis.data.adapters import digest
from resumeanalysis.data.download import sha256_file
from resumeanalysis.experiments.partial_profiles import load_splits, partial_profile


def decision_metrics(labels, analyses) -> dict:
    if not labels or len(labels) != len(analyses):
        raise ValueError("Nonempty, aligned labels and decisions required")
    decisions = [a.relevant for a in analyses]
    scores = [a.relevance for a in analyses]
    return {
        "count": len(labels),
        "accuracy": accuracy_score(labels, decisions),
        "precision": precision_score(labels, decisions, zero_division=0),
        "recall": recall_score(labels, decisions, zero_division=0),
        "macro_f1": f1_score(labels, decisions, labels=[0, 1], average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, decisions, labels=[0, 1]).tolist(),
        "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else None,
        "score_interpretation": "Uncalibrated model scores; ROC-AUC evaluates ranking, not probability calibration. Decisions include preference gating.",
    }


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_backend(backend, records, *, split="test", seed=42, warmup=1) -> dict:
    if not records:
        raise ValueError("Cannot evaluate an empty split")
    for _ in range(warmup):
        backend.predict(records[0].profile, records[0].vacancy)
    results, original_decisions = {}, None
    for ratio in (1.0, 0.75, 0.5, 0.25):
        views = [partial_profile(r, split, ratio, seed) for r in records]
        analyses, latencies = [], []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for view in views:
            _sync()
            start = time.perf_counter()
            analysis = backend.predict(view.profile, view.vacancy)
            _sync()
            latencies.append(time.perf_counter() - start)
            analyses.append(analysis)
        decisions = [a.relevant for a in analyses]
        if original_decisions is None:
            original_decisions = decisions
        retention = [
            v.retained_information_fraction
            for v in views
            if v.retained_information_fraction is not None
        ]
        result = {
            "requested_ratio": ratio,
            "observation_ids": [v.observation_id for v in views],
            "actual_information_fraction_mean": float(np.mean(retention)) if retention else None,
            "retention_definition": "Count of retained explicit skills and populated profile fields; not semantic completeness",
            "latency_seconds": {
                "mean": float(np.mean(latencies)),
                "p50": float(np.median(latencies)),
                "p95": float(np.quantile(latencies, 0.95)),
                "warmup_calls": warmup,
            },
            "positive_decisions": sum(decisions),
            "decision_changes_from_original": sum(
                a != b for a, b in zip(decisions, original_decisions)
            ),
            "scores": [a.relevance for a in analyses],
            "decisions": decisions,
            "metrics": decision_metrics([r.label for r in records], analyses)
            if ratio == 1
            else None,
            "label_status": "Original source labels (possibly synthetic/heuristic)"
            if ratio == 1
            else "No independently validated labels; diagnostics only",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if torch.cuda.is_available()
            else None,
        }
        results[str(ratio)] = result
    # Includes interpreter, initialization and warmup. Compare isolated subprocesses only.
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = rss if sys.platform == "darwin" else rss * 1024
    except ImportError:
        peak_rss = None
    return {
        "views": results,
        "process_peak_rss_bytes": peak_rss,
        "memory_scope": "Process lifetime high-water RSS; CUDA allocator per view where available. Includes runtime/model; not system-wide device memory.",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "seed": seed,
        "source_record_sha256": [digest(r.model_dump()) for r in records],
        "limitations": "Full means all source-provided information, not a verified complete resume. Synthetic labels do not establish hiring accuracy. Partial labels are never reused.",
    }


def comparison_manifest(data, split="test") -> tuple[list, dict]:
    splits = load_splits(data)
    return splits[split], {
        "source_split": split,
        "source_sha256": sha256_file(Path(data) / (split + ".jsonl")),
        "partition_check": "passed on all original splits",
    }


def combine_reports(reports: dict[str, dict]) -> dict:
    if len(reports) < 2:
        raise ValueError("Comparison requires at least two completed backend reports")
    baseline = next(iter(reports.values()))
    for report in reports.values():
        if report["source_record_sha256"] != baseline["source_record_sha256"]:
            raise ValueError("Backends were evaluated on different source records")
        for ratio in baseline["views"]:
            if (
                report["views"][ratio]["observation_ids"]
                != baseline["views"][ratio]["observation_ids"]
            ):
                raise ValueError("Backend views differ; cannot compare")
    return {
        "backends": reports,
        "comparison": "Same ordered records and partial observations verified",
        "claim": "No superiority claim; assess label validity and uncertainty before interpreting metrics",
    }
