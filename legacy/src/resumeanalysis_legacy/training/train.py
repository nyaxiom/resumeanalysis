"""CPU head training with cached frozen embeddings, early stopping and held-out test."""

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resumeanalysis.embeddings.encoder import Encoder
from resumeanalysis.extraction.registry import Extractor, is_legacy
from resumeanalysis_legacy.data.dataset import SPLITS, assert_no_leakage, load_records
from resumeanalysis_legacy.data.download import sha256_file
from resumeanalysis_legacy.features.feature_builder import (
    FEATURE_NAMES,
    FeatureEngineering,
)
from resumeanalysis_legacy.models.resume_analysis_net import ResumeAnalysisNet

from .evaluate import choose_threshold, metrics, rule_scores, write_error_analysis


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    hidden_dims: tuple[int, ...] = (128, 32)
    dropout: float = 0.2
    patience: int = 10
    min_delta: float = 0.0001
    threads: int = 2

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience, self.threads) < 1:
            raise ValueError("Epochs, batch size, patience and threads must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("Invalid optimizer/early stopping configuration")


def precompute(records, builder: FeatureEngineering, cache_path: Path, source_digest: str):
    components = [
        source_digest,
        builder.encoder.identity,
        builder.pipeline_version,
        builder.extractor.normalizer.taxonomy.identity,
        FEATURE_NAMES,
    ]
    # The legacy list is intentionally unchanged, preserving original cache keys.
    # Changed semantics require their own identity even if feature names are equal.
    if not is_legacy(builder.extractor):
        components.append(builder.extractor_identity)
    identity = json.dumps(components)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    cache_path.mkdir(parents=True, exist_ok=True)
    path = cache_path / f"{digest}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as a:
            arrays = [a[k].copy() for k in ["resumes", "vacancies", "features", "labels"]]
        if len(arrays[0]) != len(records):
            raise ValueError("Feature cache row count mismatch")
        return arrays
    details = []
    for i, record in enumerate(records):
        details.append(builder.build(record.profile, record.vacancy))
        if (i + 1) % 100 == 0:
            print(f"Encoded {i + 1}/{len(records)} pairs", flush=True)
    arrays = [
        np.stack([d.resume_embedding for d in details]),
        np.stack([d.vacancy_embedding for d in details]),
        np.stack([d.vector for d in details]),
        np.asarray([r.label for r in records], dtype=np.float32),
    ]
    np.savez_compressed(path, **dict(zip(["resumes", "vacancies", "features", "labels"], arrays)))
    return arrays


def combined(arrays):
    r, j, f, _ = arrays
    return np.concatenate([r, j, np.abs(r - j), r * j, f], axis=1)


def score_model(model, arrays) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), 256):
            inputs = [torch.from_numpy(a[start : start + 256]) for a in arrays[:3]]
            result.extend(model(*inputs).sigmoid().tolist())
    return np.asarray(result)


def train(
    data_dir: str | Path,
    output: str | Path,
    encoder: Encoder,
    config: TrainingConfig | None = None,
    extractor: Extractor | None = None,
) -> dict:
    config = config or TrainingConfig()
    data_dir, output = Path(data_dir), Path(output)
    builder = FeatureEngineering(encoder, extractor)
    legacy = is_legacy(builder.extractor)
    if (
        not legacy
        and output.exists()
        and (
            any(output.glob("*.pt"))
            or any(output.glob("*.pth"))
            or any(output.glob("*.safetensors"))
            or (output / "metrics.json").exists()
        )
    ):
        raise FileExistsError(
            "Changed extraction requires a new training output directory; existing checkpoints/metrics are protected"
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    records = {s: load_records(data_dir / f"{s}.jsonl") for s in SPLITS}
    assert_no_leakage(records)
    if any(len({r.label for r in rows}) != 2 for rows in records.values()):
        raise ValueError("Each train/validation/test split must contain both classes")
    digests = {s: sha256_file(data_dir / f"{s}.jsonl") for s in SPLITS}
    arrays = {
        s: precompute(records[s], builder, output.parent / "feature_cache", digests[s])
        for s in SPLITS
    }
    tr, va, te = [arrays[s] for s in SPLITS]
    model = ResumeAnalysisNet(
        encoder.dimension, len(FEATURE_NAMES), config.hidden_dims, config.dropout
    )
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    positives = float(tr[3].sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr[3]) - positives) / positives))
    loader = DataLoader(
        TensorDataset(*[torch.from_numpy(a) for a in tr]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for r, j, f, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(r, j, f), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        model.eval()
        with torch.inference_mode():
            validation_loss = criterion(
                model(*[torch.from_numpy(a) for a in va[:3]]), torch.from_numpy(va[3])
            ).item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(tr[3]),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise ValueError("No valid checkpoint was trained")
    model.load_state_dict(best_state)
    learned = any(not torch.equal(initial_state[k], best_state[k]) for k in best_state)
    if not learned:
        raise ValueError("Training did not update parameters")
    val_neural = score_model(model, va)
    threshold = choose_threshold(va[3], val_neural)
    checkpoint = {
        "format_version": 1 if legacy else 2,
        "state_dict": best_state,
        "network_config": model.config,
        "feature_names": FEATURE_NAMES,
        "pipeline_version": builder.pipeline_version,
        "taxonomy_identity": builder.extractor.normalizer.taxonomy.identity,
        "encoder_identity": encoder.identity,
        "threshold": threshold,
        "training_config": asdict(config),
        "best_epoch": best_epoch,
        "data_sha256": digests,
    }
    if not legacy:
        checkpoint["extractor_identity"] = builder.extractor_identity
    torch.save(checkpoint, output / "model.pt")
    # Both learned methods receive exactly the same input representation.
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=config.seed),
    )
    logistic.fit(combined(tr), tr[3])
    val_logistic = logistic.predict_proba(combined(va))[:, 1]
    logistic_threshold = choose_threshold(va[3], val_logistic)
    rule_threshold = choose_threshold(va[3], rule_scores(va[2]))
    scaler, classifier = logistic.steps[0][1], logistic.steps[1][1]
    # Store only numeric arrays, never executable pickle from third-party sources.
    np.savez(
        output / "logistic.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        threshold=logistic_threshold,
    )
    scores = {
        "rule": rule_scores(te[2]),
        "logistic": logistic.predict_proba(combined(te))[:, 1],
        "neural": score_model(model, te),
    }
    thresholds = {"rule": rule_threshold, "logistic": logistic_threshold, "neural": threshold}
    report = {
        "encoder": encoder.identity,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "epochs_executed": len(history),
        "parameters_updated": learned,
        "test": {name: metrics(te[3], scores[name], thresholds[name]) for name in scores},
        "validation": {
            "neural": metrics(va[3], val_neural, threshold),
            "logistic": metrics(va[3], val_logistic, logistic_threshold),
            "rule": metrics(va[3], rule_scores(va[2]), rule_threshold),
        },
        "split_counts": {s: len(records[s]) for s in SPLITS},
        "data_sha256": digests,
        "label_warning": "See processed manifest. Synthetic labels are not employer decisions.",
        "score_warning": "Sigmoid score is not a validated calibrated probability.",
        "history": history,
    }
    if not legacy:
        report.update(
            {
                "pipeline_version": builder.pipeline_version,
                "extractor_identity": builder.extractor_identity,
            }
        )
    report["neural_improves_over_rule"] = (
        report["test"]["neural"]["macro_f1"] > report["test"]["rule"]["macro_f1"]
    )
    report["neural_improves_over_logistic"] = (
        report["test"]["neural"]["macro_f1"] > report["test"]["logistic"]["macro_f1"]
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    for name in scores:
        write_error_analysis(
            records["test"],
            te[3],
            scores[name],
            te[2],
            thresholds[name],
            output / f"errors_{name}.json",
        )
    np.savez(output / "test_predictions.npz", labels=te[3], **scores)
    return report


def evaluate_saved(
    data_dir: str | Path,
    output: str | Path,
    encoder: Encoder,
    extractor: Extractor | None = None,
) -> dict:
    """Re-evaluate saved weights on the exact held-out test, without refitting anything."""
    from resumeanalysis_legacy.inference.predictor import Predictor

    output, data_dir = Path(output), Path(data_dir)
    saved = json.loads((output / "metrics.json").read_text())
    if saved["data_sha256"]["test"] != sha256_file(data_dir / "test.jsonl"):
        raise ValueError("Held-out test data changed")
    predictor = Predictor(output / "model.pt", encoder, extractor=extractor)
    records = load_records(data_dir / "test.jsonl")
    arrays = precompute(
        records, predictor.builder, output.parent / "feature_cache", saved["data_sha256"]["test"]
    )
    with np.load(output / "logistic.npz", allow_pickle=False) as weights:
        z = ((combined(arrays) - weights["mean"]) / weights["scale"]) @ weights["coef"].T + weights[
            "intercept"
        ]
        logistic_scores = 1 / (1 + np.exp(-np.clip(z[:, 0], -60, 60)))
        logistic_threshold = float(weights["threshold"])
    return {
        "neural": metrics(
            arrays[3], score_model(predictor.model, arrays), predictor.postprocessor.threshold
        ),
        "logistic": metrics(arrays[3], logistic_scores, logistic_threshold),
        "rule": metrics(
            arrays[3], rule_scores(arrays[2]), saved["validation"]["rule"]["threshold"]
        ),
    }
