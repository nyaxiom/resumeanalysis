"""Authored model/runtime fixtures; real pretrained smoke is explicitly opt in."""

import io
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from resumeanalysis.rerank import assets
from resumeanalysis.rerank import model as runtime
from resumeanalysis.rerank.config import RerankConfig
from resumeanalysis.rerank.model import CrossEncoder, _chunks


def asset_fixture(tmp_path, monkeypatch):
    values = {"config.json": b'{"fixture":true}', "model.safetensors": b"authored fake bytes"}
    expected = {}
    for name, value in values.items():
        path = tmp_path / name
        path.write_bytes(value)
        expected[name] = (len(value), assets.sha256_file(path))
    monkeypatch.setattr(assets, "FILES", expected)
    license_path = (
        Path(__file__).resolve().parents[1] / "LICENSES/references/reranker-Apache-LICENSE.txt"
    )
    (tmp_path / "APACHE-2.0.txt").write_bytes(license_path.read_bytes())
    (tmp_path / assets.MANIFEST).write_text(json.dumps(assets.expected_manifest()))
    return values


def test_asset_verification_checks_all_files_before_model_loading(tmp_path, monkeypatch):
    asset_fixture(tmp_path, monkeypatch)
    assert assets.verify_assets(tmp_path) == assets.expected_manifest()
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    monkeypatch.setattr(
        runtime.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: pytest.fail("loaded corrupt asset"),
    )
    with pytest.raises(ValueError, match="checksum"):
        CrossEncoder(RerankConfig(model_path=str(tmp_path)))


def test_manifest_identity_cannot_be_rewritten_to_authorize_different_weights(
    tmp_path, monkeypatch
):
    asset_fixture(tmp_path, monkeypatch)
    manifest = assets.expected_manifest()
    manifest["revision"] = "main"
    (tmp_path / assets.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        assets.verify_assets(tmp_path)


def test_missing_manifest_and_asset_fail_explicitly(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="explicit download"):
        assets.verify_assets(tmp_path)
    asset_fixture(tmp_path, monkeypatch)
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="config.json"):
        assets.verify_assets(tmp_path)


def test_retained_license_cannot_be_replaced_by_a_symlink(tmp_path, monkeypatch):
    asset_fixture(tmp_path, monkeypatch)
    license_path = tmp_path / "APACHE-2.0.txt"
    target = tmp_path / "other-license.txt"
    license_path.rename(target)
    license_path.symlink_to(target)
    with pytest.raises(ValueError, match="license notice mismatch"):
        assets.verify_assets(tmp_path)


def test_explicit_downloader_is_pinned_idempotent_and_refuses_modified_existing_bytes(
    tmp_path, monkeypatch
):
    seed = tmp_path / "seed"
    seed.mkdir()
    values = asset_fixture(seed, monkeypatch)
    license_bytes = (seed / "APACHE-2.0.txt").read_bytes()
    calls = []

    def read(url, **kwargs):
        calls.append(url)
        if url == assets.LICENSE_SOURCE:
            return io.BytesIO(license_bytes)
        assert "/resolve/" + runtime.RerankConfig().revision + "/" in url
        return io.BytesIO(values[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(assets.urllib.request, "urlopen", read)
    output = tmp_path / "download"
    assets.download_reranker(output)
    assert len(calls) == 3
    assets.download_reranker(output)
    assert len(calls) == 3
    (output / "config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        assets.download_reranker(output)


def test_bad_download_does_not_publish_partial_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"not pinned")
    )
    with pytest.raises(ValueError, match="checksum"):
        assets.download_reranker(tmp_path)
    assert not (tmp_path / assets.MANIFEST).exists()
    assert not list(tmp_path.glob("*.partial"))


class TinyTokenizer:
    model_max_length = 512

    def __init__(self):
        self.tokenized = []

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def __call__(self, values, **kwargs):
        self.tokenized.extend(values)
        return {"input_ids": [[3 + i for i, _ in enumerate(text.split())] for text in values]}

    def build_inputs_with_special_tokens(self, a, b):
        return [0, *a, 2, 2, *b, 2]

    def pad(self, features, **kwargs):
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] + [1] * (width - len(f["input_ids"])) for f in features]
            ),
            "attention_mask": torch.tensor(
                [f["attention_mask"] + [0] * (width - len(f["attention_mask"])) for f in features]
            ),
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            model_type="xlm-roberta", num_labels=1, hidden_size=384, num_hidden_layers=12
        )
        self.batch_sizes = []

    def forward(self, input_ids, attention_mask):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.batch_sizes.append(len(input_ids))
        return SimpleNamespace(logits=attention_mask.sum(1, keepdim=True).float() * self.value)


def fake_runtime(monkeypatch, **config):
    tokenizer, model, calls = TinyTokenizer(), TinyModel(), []
    monkeypatch.setattr(runtime, "verify_assets", lambda path: {"authored": "fixture"})

    def load_tokenizer(*args, **kwargs):
        calls.append(kwargs)
        return tokenizer

    def load_model(*args, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(runtime.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(runtime.AutoModelForSequenceClassification, "from_pretrained", load_model)
    return CrossEncoder(RerankConfig(**config)), tokenizer, model, calls


def test_offline_frozen_batched_joint_inference_reuses_duplicate_text(monkeypatch):
    scorer, tokenizer, model, calls = fake_runtime(monkeypatch, batch_size=2)
    pairs = [
        ("Python SQL", "Python developer"),
        ("Python SQL", "Разработчик Python"),
        ("mixed Python", "Python developer"),
    ]
    assert scorer.score_pairs(pairs) == [8.0, 8.0, 8.0]
    assert model.batch_sizes == [2, 1]
    assert len(tokenizer.tokenized) == len(set(text for pair in pairs for text in pair))
    assert all(not p.requires_grad for p in model.parameters())
    assert all(c["local_files_only"] and c["trust_remote_code"] is False for c in calls)
    assert calls[1]["use_safetensors"] is True
    assert all(not d["truncated"] and d["windows"] == 1 for d in scorer.last_diagnostics)


def test_empty_batch_never_calls_tokenizer_or_model_and_clears_diagnostics(monkeypatch):
    scorer, tokenizer, model, _ = fake_runtime(monkeypatch)
    scorer.last_diagnostics = [{"stale": True}]
    assert scorer.score_pairs([]) == []
    assert scorer.last_diagnostics == tokenizer.tokenized == model.batch_sizes == []


@pytest.mark.parametrize("pairs", [None, (("a", "b"),), ["ab"], [("a", 3)], [("a", "b", "c")]])
def test_invalid_pair_inputs_fail_before_inference(monkeypatch, pairs):
    scorer, _, model, _ = fake_runtime(monkeypatch)
    with pytest.raises(TypeError):
        scorer.score_pairs(pairs)
    assert not model.batch_sizes


def test_long_pairs_are_not_silently_truncated_under_error_or_window_policy(monkeypatch):
    scorer, _, model, _ = fake_runtime(
        monkeypatch, max_length=32, overlap_tokens=2, long_document_policy="error"
    )
    with pytest.raises(ValueError, match="exceeds max_length"):
        scorer.score_pairs([("short", "short"), ("word " * 40, "other " * 40)])
    assert not model.batch_sizes
    scorer, _, _, _ = fake_runtime(
        monkeypatch, max_length=32, overlap_tokens=2, long_document_policy="window", max_windows=16
    )
    assert len(scorer.score_pairs([("word " * 40, "other " * 40)])) == 1
    diagnostic = scorer.last_diagnostics[0]
    assert diagnostic["windows"] > 1 and diagnostic["aggregation"] == "max"
    assert diagnostic["resume_tokens_retained"] == diagnostic["vacancy_tokens_retained"] == 40
    assert diagnostic["truncated"] is False


def test_window_budget_exhaustion_fails_without_omitting_tail(monkeypatch):
    scorer, _, model, _ = fake_runtime(monkeypatch, max_length=32, overlap_tokens=2, max_windows=1)
    with pytest.raises(ValueError, match="max_windows"):
        scorer.score_pairs([("word " * 40, "other " * 40)])
    assert not model.batch_sizes


def test_explicit_truncation_reports_loss_for_both_inputs(monkeypatch):
    scorer, _, _, _ = fake_runtime(monkeypatch, max_length=32, long_document_policy="truncate")
    assert scorer.score_pairs([("word " * 40, "other " * 40)]) == [32.0]
    diagnostic = scorer.last_diagnostics[0]
    assert diagnostic["truncated"] is True and diagnostic["windows"] == 1
    assert diagnostic["resume_tokens_discarded"] == diagnostic["vacancy_tokens_discarded"] == 26


def test_window_chunks_cover_tail_and_empty_document(monkeypatch):
    tokens = list(range(39))
    assert set().union(*map(set, _chunks(tokens, 14, 2))) == set(tokens)
    scorer, _, _, _ = fake_runtime(monkeypatch, max_length=32, overlap_tokens=2)
    assert len(scorer.score_pairs([("", "word " * 40)])) == 1
    assert scorer.last_diagnostics[0]["resume_tokens"] == 0
    assert scorer.last_diagnostics[0]["vacancy_tokens_retained"] == 40


def test_runtime_identity_changes_for_sequence_policy_not_file_location(monkeypatch):
    a, _, _, _ = fake_runtime(monkeypatch, model_path="one")
    b, _, _, _ = fake_runtime(monkeypatch, model_path="two")
    c, _, _, _ = fake_runtime(monkeypatch, max_length=256)
    assert a.identity == b.identity and a.identity != c.identity


@pytest.mark.parametrize(
    "mutation", ["config", "training", "gradient", "weights", "model", "tokenizer"]
)
def test_loaded_runtime_mutation_fails_before_scoring(monkeypatch, mutation):
    scorer, _, model, _ = fake_runtime(monkeypatch)
    if mutation == "config":
        scorer.config = RerankConfig(max_length=256)
    elif mutation == "training":
        model.train()
    elif mutation == "gradient":
        model.requires_grad_(True)
    elif mutation == "weights":
        with torch.no_grad():
            model.value.add_(1)
    elif mutation == "model":
        scorer.model = TinyModel().eval().requires_grad_(False)
    else:
        scorer.tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="runtime identity changed"):
        scorer.score_pairs([("Python", "Python required")])


@pytest.mark.parametrize(
    "bad_logits", [torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])]
)
def test_nonfinite_logits_fail_explicitly(monkeypatch, bad_logits):
    scorer, _, model, _ = fake_runtime(monkeypatch)
    monkeypatch.setattr(model, "forward", lambda **kwargs: SimpleNamespace(logits=bad_logits))
    with pytest.raises(ValueError, match="nonfinite"):
        scorer.score_pairs([("a", "b")])
    assert scorer.last_diagnostics == []


@pytest.mark.parametrize("bad_logits", [torch.zeros(1), torch.zeros(1, 2), torch.zeros(2, 1)])
def test_unexpected_classifier_shape_fails_explicitly(monkeypatch, bad_logits):
    scorer, _, model, _ = fake_runtime(monkeypatch)
    monkeypatch.setattr(model, "forward", lambda **kwargs: SimpleNamespace(logits=bad_logits))
    with pytest.raises(ValueError, match="logit shape"):
        scorer.score_pairs([("a", "b")])


def test_unreviewed_architecture_fails_before_inference(monkeypatch):
    model = TinyModel()
    model.config.num_hidden_layers = 6
    monkeypatch.setattr(runtime, "verify_assets", lambda path: {"authored": "fixture"})
    monkeypatch.setattr(runtime.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    monkeypatch.setattr(
        runtime.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **kw: model
    )
    with pytest.raises(ValueError, match="architecture differs"):
        CrossEncoder()
    assert model.batch_sizes == []


@pytest.mark.parametrize(
    "values",
    [
        {"revision": "main"},
        {"batch_size": 0},
        {"batch_size": True},
        {"max_length": 1024},
        {"max_length": 32, "overlap_tokens": 16},
    ],
)
def test_invalid_or_unreviewed_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        RerankConfig(**values)


@pytest.mark.skipif(
    os.getenv("RESUMEANALYSIS_RUN_RERANK_TESTS") != "1",
    reason="opt-in explicitly downloaded pretrained reranker",
)
def test_actual_pretrained_model_loads_and_scores_offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline inference attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    scorer = CrossEncoder(
        RerankConfig(model_path=os.getenv("RESUMEANALYSIS_RERANK_PATH", "artifacts/reranker"))
    )
    scores = scorer.score_pairs(
        [
            ("Python SQL developer", "Python SQL developer needed"),
            ("Разработчик Python", "Нужен разработчик Python"),
            ("Python SQL developer", "An experienced pastry chef is needed"),
        ]
    )
    assert len(scores) == 3 and all(isinstance(value, float) for value in scores)
    assert scores[0] > scores[2]
    assert scorer.score_pairs([]) == []
