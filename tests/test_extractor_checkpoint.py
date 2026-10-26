"""Changing extraction semantics must not silently reuse old learned weights/caches."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from resumeanalysis.data.dataset import PairRecord, save_splits
from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.features.feature_builder import FEATURE_NAMES, FeatureEngineering
from resumeanalysis.inference.predictor import Predictor
from resumeanalysis.normalization.skill_normalizer import SkillNormalizer
from resumeanalysis.training.train import TrainingConfig, evaluate_saved, precompute, train


def _splits():
    result = {}
    for split in ("train", "validation", "test"):
        result[split] = [
            PairRecord(
                candidate_id=f"{split}-person-{index}",
                vacancy_id=f"{split}-job-{index}",
                profile={
                    "resume_text": f"{split} candidate record {index}",
                    "skills": ["Python"] if index % 2 else ["Excel"],
                },
                vacancy={
                    "description": f"{split} vacancy record {index}. Essential Skills: Python",
                    "required_skills": ["Python"],
                },
                label=index % 2,
                provenance="authored compatibility fixture; no quality claim",
            )
            for index in range(6)
        ]
    return result


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("extractor-checkpoints")
    save_splits(_splits(), root / "data", {})
    encoder = LexicalSmokeEncoder()
    config = TrainingConfig(epochs=2, patience=1, hidden_dims=(8,), dropout=0)
    for name in ("legacy", "v2"):
        train(root / "data", root / name, encoder, config, create_extractor(name))
    return root, encoder


def test_original_format_one_and_v2_round_trip(trained):
    root, encoder = trained
    old = torch.load(root / "legacy/model.pt", weights_only=True)
    new = torch.load(root / "v2/model.pt", weights_only=True)
    assert old["format_version"] == old["pipeline_version"] == 1
    assert "extractor_identity" not in old
    assert new["format_version"] == new["pipeline_version"] == 2
    assert new["extractor_identity"]
    pair = _splits()["test"][1]
    default = Predictor(root / "legacy/model.pt", encoder)
    explicit = Predictor(root / "legacy/model.pt", encoder, extractor=create_extractor("legacy"))
    assert default.predict(pair.profile, pair.vacancy) == explicit.predict(
        pair.profile, pair.vacancy
    )
    model = Predictor(root / "v2/model.pt", encoder, extractor=create_extractor("v2"))
    assert (
        model.predict(pair.profile, pair.vacancy).relevance
        == Predictor(root / "v2/model.pt", encoder, extractor=create_extractor("v2"))
        .predict(pair.profile, pair.vacancy)
        .relevance
    )
    saved = json.loads((root / "v2/metrics.json").read_text())
    repeated = evaluate_saved(root / "data", root / "v2", encoder, create_extractor("v2"))
    assert repeated == saved["test"]


def test_old_and_new_extractor_checkpoint_mismatches_fail(trained):
    root, encoder = trained
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        Predictor(root / "legacy/model.pt", encoder, extractor=create_extractor("v2"))
    with pytest.raises(ValueError, match="extractor identity"):
        Predictor(root / "v2/model.pt", encoder)


@pytest.mark.parametrize("change", ["missing_identity", "changed_identity", "pipeline", "taxonomy"])
def test_invalid_semantic_provenance_rejected_before_inference(tmp_path, trained, change):
    root, encoder = trained
    data = torch.load(root / "v2/model.pt", weights_only=True)
    if change == "missing_identity":
        data.pop("extractor_identity")
    elif change == "changed_identity":
        data["extractor_identity"] = "different-extraction-semantics"
    elif change == "pipeline":
        data["pipeline_version"] = 99
    else:
        data["taxonomy_identity"] = "different-taxonomy"
    path = tmp_path / "incompatible.pt"
    torch.save(data, path)
    with pytest.raises(ValueError, match="mismatch"):
        Predictor(path, encoder, extractor=create_extractor("v2"))


def test_precompute_preserves_legacy_key_and_isolates_changed_semantics(tmp_path):
    encoder = LexicalSmokeEncoder()
    records = _splits()["test"]
    legacy = FeatureEngineering(encoder)
    precompute(records, legacy, tmp_path / "legacy-cache", "source-sha")
    original_components = [
        "source-sha",
        encoder.identity,
        1,
        legacy.extractor.normalizer.taxonomy.identity,
        FEATURE_NAMES,
    ]
    expected = hashlib.sha256(json.dumps(original_components).encode()).hexdigest() + ".npz"
    assert [p.name for p in (tmp_path / "legacy-cache").glob("*.npz")] == [expected]
    actual = FeatureEngineering(encoder, create_extractor("v2"))
    for identity in ("v2-semantics-a", "v2-semantics-b"):
        builder = SimpleNamespace(
            encoder=encoder,
            extractor=actual.extractor,
            pipeline_version=2,
            extractor_identity=identity,
            build=actual.build,
        )
        precompute(records, builder, tmp_path / "new-cache", "source-sha")
    assert len(list((tmp_path / "new-cache").glob("*.npz"))) == 2


def test_v2_training_cannot_overwrite_existing_checkpoint_or_metrics(trained, tmp_path):
    root, encoder = trained
    original = (root / "legacy/model.pt").read_bytes()
    with pytest.raises(FileExistsError, match="new training output"):
        train(root / "data", root / "legacy", encoder, extractor=create_extractor("v2"))
    assert (root / "legacy/model.pt").read_bytes() == original
    protected = tmp_path / "metrics-only"
    protected.mkdir()
    (protected / "metrics.json").write_text("protected")
    with pytest.raises(FileExistsError, match="protected"):
        train(root / "data", protected, encoder, extractor=create_extractor("v2"))
    assert (protected / "metrics.json").read_text() == "protected"


def test_unknown_subclass_cannot_inherit_legacy_checkpoint_compatibility():
    class ChangedExtraction(SkillExtractor):
        def extract(self, text, *, vacancy=False):
            return super().extract(text, vacancy=vacancy)

    with pytest.raises(ValueError):
        FeatureEngineering(LexicalSmokeEncoder(), ChangedExtraction())


def test_custom_normalizer_cannot_masquerade_as_legacy_extraction():
    class ChangedNormalizer(SkillNormalizer):
        def normalize(self, skill, context=""):
            return "different-meaning"

    extractor = SkillExtractor(ChangedNormalizer())
    with pytest.raises(ValueError):
        FeatureEngineering(LexicalSmokeEncoder(), extractor)
