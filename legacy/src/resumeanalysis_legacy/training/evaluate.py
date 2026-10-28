import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from resumeanalysis_legacy.features.feature_builder import FEATURE_NAMES


def metrics(labels, scores, threshold=0.5) -> dict:
    y = np.asarray(labels)
    scores = np.asarray(scores)
    if y.size == 0 or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid evaluation arrays")
    pred = scores >= threshold
    return {
        "count": len(y),
        "positive_count": int(y.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None,
    }


def choose_threshold(labels, scores) -> float:
    # Select using validation ONLY. Tie-break toward 0.5, then lower threshold.
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 37), 0.5])
    return float(
        max(candidates, key=lambda t: (metrics(labels, scores, t)["macro_f1"], -abs(t - 0.5), -t))
    )


def rule_scores(features: np.ndarray) -> np.ndarray:
    """Transparent baseline; not the learned model exposed by predict()."""
    ix = {n: i for i, n in enumerate(FEATURE_NAMES)}
    return np.clip(
        0.65 * features[:, ix["required_match"]]
        + 0.10 * features[:, ix["optional_match"]]
        + 0.15 * features[:, ix["semantic_similarity"]]
        + 0.10 * features[:, ix["experience_match"]]
        - 0.35 * features[:, ix["required_denied"]],
        0,
        1,
    )


def write_error_analysis(records, labels, scores, features, threshold, output: str | Path) -> None:
    errors = []
    for i, (record, label, score) in enumerate(zip(records, labels, scores)):
        if int(score >= threshold) != int(label):
            errors.append(
                {
                    "candidate_id": record.candidate_id,
                    "vacancy_id": record.vacancy_id,
                    "label": int(label),
                    "score": float(score),
                    "type": "false_positive" if label == 0 else "false_negative",
                    "hard_negative": record.hard_negative,
                    "provenance": record.provenance,
                    "features": dict(zip(FEATURE_NAMES, map(float, features[i]))),
                }
            )
    Path(output).write_text(json.dumps(errors, ensure_ascii=False, indent=2))
