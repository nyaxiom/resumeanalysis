import json

import numpy as np
import pytest
import torch

from resumeanalysis.data.dataset import PairRecord, save_splits
from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.inference.predictor import Predictor
from resumeanalysis.models.resume_analysis_net import ResumeAnalysisNet
from resumeanalysis.schemas import Profile, Vacancy
from resumeanalysis.training.evaluate import metrics
from resumeanalysis.training.train import TrainingConfig, evaluate_saved, train


def test_network_backprop_uses_embeddings_and_features():
    model = ResumeAnalysisNet(4, 3, hidden_dims=(8,), dropout=0)
    resume, vacancy, features = [torch.ones(2, d, requires_grad=True) for d in [4, 4, 3]]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(resume, vacancy, features), torch.tensor([0.0, 1.0])
    )
    loss.backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in [resume, vacancy, features])


def test_training_save_reload_and_baseline_evaluation(tmp_path):
    # Hand-authored text fixtures exercise training mechanics, not reported model quality.
    splits = {}
    for name in ["train", "validation", "test"]:
        rows = []
        for i in range(12):
            positive = bool(i % 2)
            rows.append(
                PairRecord(
                    candidate_id=f"{name}-p{i}",
                    vacancy_id=f"{name}-v{i}",
                    profile=Profile(
                        resume_text=f"{name} candidate {i}",
                        skills=["Python"] if positive else ["Excel"],
                    ),
                    vacancy=Vacancy(description=f"{name} vacancy {i}", required_skills=["Python"]),
                    label=int(positive),
                    provenance="authored integration fixture",
                )
            )
        splits[name] = rows
    save_splits(splits, tmp_path / "data", {})
    encoder = LexicalSmokeEncoder()
    report = train(
        tmp_path / "data",
        tmp_path / "run",
        encoder,
        TrainingConfig(epochs=8, patience=3, hidden_dims=(16, 8), dropout=0),
    )
    assert report["parameters_updated"]
    assert report["test"]["neural"]["count"] == 12
    predictor = Predictor(tmp_path / "run/model.pt", encoder)
    pair = splits["test"][0]
    a = predictor.predict(pair.profile, pair.vacancy)
    b = Predictor(tmp_path / "run/model.pt", encoder).predict(pair.profile, pair.vacancy)
    assert a.relevance == b.relevance
    repeated = evaluate_saved(tmp_path / "data", tmp_path / "run", encoder)
    for name in repeated:
        assert repeated[name]["macro_f1"] == report["test"][name]["macro_f1"]
        assert repeated[name]["roc_auc"] == report["test"][name]["roc_auc"]

    class OtherEncoder(LexicalSmokeEncoder):
        identity = "different-encoder"

    with pytest.raises(ValueError, match="encoder"):
        Predictor(tmp_path / "run/model.pt", OtherEncoder())
    assert json.loads((tmp_path / "run/metrics.json").read_text())["best_epoch"] > 0


def test_metrics_single_class_and_empty_input():
    assert metrics(np.array([1, 1]), np.array([0.8, 0.9]))["roc_auc"] is None
    with pytest.raises(ValueError):
        metrics([], [])
