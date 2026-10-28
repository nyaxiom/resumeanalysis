"""Exact-retrieval mechanics; random local weights never establish matching quality."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from resumeanalysis_legacy.features.feature_builder import profile_text, vacancy_text
from resumeanalysis_legacy.rerank.retrieval import ExactRetriever
from transformers import BertConfig, BertModel, BertTokenizerFast

from resumeanalysis.embeddings.encoder import TextEncoder
from resumeanalysis.schemas import Profile, Vacancy


class CountingEncoder:
    identity = "authored-retrieval-fixture-v1"
    dimension = 3

    def __init__(self):
        self.calls = []

    def encode(self, text, role="query"):
        self.calls.append((text, role))
        if role == "query":
            return np.array([3, 0, 0], dtype=np.float32)
        if "alpha" in text:
            return np.array([4, 0, 0], dtype=np.float32)
        if "beta" in text:
            return np.array([1, 1, 0], dtype=np.float32)
        return np.array([0, 5, 0], dtype=np.float32)


def test_exact_cosine_normalizes_and_ties_use_original_positions():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder)
    profile = Profile(resume_text="Python and SQL")
    vacancies = [Vacancy(description=t) for t in ["beta", "alpha", "alpha", "other"]]
    scores = retriever.score_all(profile, vacancies)
    assert [c.index for c in scores] == [0, 1, 2, 3]
    assert [c.cosine_score for c in scores] == pytest.approx([2**-0.5, 1, 1, 0])
    ranked = retriever.retrieve(profile, vacancies, preselect_k=3)
    assert [c.index for c in ranked] == [1, 2, 0]
    assert ranked[0].vacancy == vacancies[1]
    # Query once and each distinct vacancy once, even across repeated rankings.
    assert sum(role == "query" for _, role in encoder.calls) == 1
    assert sum(role == "passage" for _, role in encoder.calls) == 3


def test_existing_text_helpers_are_reused_without_job_title_skill_inference():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder)
    profile = Profile(resume_text="Есть опыт", projects=["SQL проект"], denied_skills=["Go"])
    vacancy = Vacancy(title="alpha title-only skill", description="Работа с данными")
    retriever.retrieve(profile, [vacancy])
    assert encoder.calls == [
        (profile_text(profile), "query"),
        (vacancy_text(vacancy), "passage"),
    ]
    assert "alpha title-only skill" not in encoder.calls[1][0]


def test_profile_cache_invalidates_only_for_changed_embedding_text():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder)
    a, b = Profile(resume_text="English"), Profile(resume_text="Русский Python")
    vacancies = [Vacancy(description="alpha")]
    retriever.retrieve(a, vacancies)
    retriever.retrieve(b, vacancies)
    retriever.retrieve(a.model_copy(update={"allowed_locations": ["London"]}), vacancies)
    assert sum(role == "query" for _, role in encoder.calls) == 2
    assert sum(role == "passage" for _, role in encoder.calls) == 1


def test_optional_batch_encoder_receives_deduplicated_inputs():
    class Batch(CountingEncoder):
        def __init__(self):
            super().__init__()
            self.batches = []

        def encode_many(self, texts, role="passage"):
            self.batches.append((list(texts), role))
            return np.stack([super(Batch, self).encode(t, role) for t in texts])

    encoder = Batch()
    retriever = ExactRetriever(encoder)
    values = retriever.encode_many(["alpha", "beta", "alpha"], "passage")
    assert encoder.batches == [(["alpha", "beta"], "passage")]
    assert values.shape == (3, 3)
    assert np.array_equal(values[0], values[2])


def test_caller_cannot_poison_cached_vectors_and_lru_is_bounded():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder, memory_cache_size=2)
    a = retriever.encode_many(["alpha"])
    a[:] = 100
    assert retriever.encode_many(["alpha"])[0].tolist() == [1, 0, 0]
    retriever.encode_many(["beta", "other"])
    assert len(retriever._cache) == 2
    before = len(encoder.calls)
    retriever.encode_many(["alpha"])
    assert len(encoder.calls) == before + 1


def test_disk_cache_is_bound_to_identity_content_role_and_configuration(tmp_path):
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder, tmp_path)
    expected = retriever.encode_many(["alpha"], "passage")
    another = CountingEncoder()
    restored = ExactRetriever(another, tmp_path)
    assert np.array_equal(restored.encode_many(["alpha"], "passage"), expected)
    assert not another.calls
    assert restored.cache_stats["disk_hits"] == 1
    restored.encode_many(["alpha"], "query")
    assert another.calls == [("alpha", "query")]
    another.identity = "authored-retrieval-fixture-v2"
    changed = ExactRetriever(another, tmp_path)
    changed.encode_many(["alpha"], "passage")
    assert len(another.calls) == 2


@pytest.mark.parametrize("corruption", ["bytes", "identity", "nan", "shape"])
def test_corrupted_disk_cache_is_recomputed_without_pickle(tmp_path, corruption):
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder, tmp_path, memory_cache_size=0)
    expected = retriever.encode_many(["alpha"])
    cache = next(tmp_path.glob("*.npz"))
    if corruption == "bytes":
        cache.write_bytes(b"not a numpy zip")
    else:
        with np.load(cache, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        if corruption == "identity":
            values["identity"] = np.asarray("another model")
        elif corruption == "nan":
            values["vector"] = np.array([np.nan, 0, 0])
        else:
            values["vector"] = np.zeros(4)
        np.savez(cache, **values)
    assert np.array_equal(retriever.encode_many(["alpha"]), expected)
    assert retriever.cache_stats["invalid_disk_entries"] == 1
    assert len(encoder.calls) == 2


def test_encoder_identity_mutation_fails_instead_of_using_stale_cache():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder)
    retriever.encode_many(["alpha"])
    encoder.identity = "different model"
    with pytest.raises(ValueError, match="changed"):
        retriever.encode_many(["alpha"])


@pytest.mark.parametrize("vector", [np.array([np.nan, 0, 0]), np.zeros(2), np.ones(3) * np.inf])
def test_nonfinite_or_misshaped_encoder_vectors_are_rejected(vector):
    class Invalid(CountingEncoder):
        def encode(self, *args, **kwargs):
            return vector

    with pytest.raises(ValueError, match="embedding"):
        ExactRetriever(Invalid()).encode_many(["alpha"])


def test_zero_vectors_produce_zero_similarity_and_stable_order():
    class Empty(CountingEncoder):
        def encode(self, *args, **kwargs):
            return np.zeros(3)

    ranked = ExactRetriever(Empty()).retrieve({}, [{}, {}])
    assert [c.index for c in ranked] == [0, 1]
    assert [c.cosine_score for c in ranked] == [0, 0]


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "2"])
def test_preselect_limit_is_explicit_positive_integer(limit):
    with pytest.raises(ValueError, match="positive integer"):
        ExactRetriever(CountingEncoder()).retrieve({}, [], limit)


def test_empty_vacancy_list_does_not_encode_profile():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder)
    assert retriever.retrieve({}, [], 1) == []
    assert retriever.score_all({}, []) == []
    assert encoder.calls == []
    assert retriever.vacancy_matrix([]).shape == (0, 3)


def test_concurrent_queries_do_not_duplicate_missing_vacancy_encodings():
    encoder = CountingEncoder()
    retriever = ExactRetriever(encoder)
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _: retriever.encode_many(["alpha", "beta"]), range(8)))
    assert len(encoder.calls) == 2
    assert all(np.array_equal(values[0], v) for v in values)


def tiny_e5(tmp_path):
    path = tmp_path / "tiny"
    path.mkdir()
    (path / "vocab.txt").write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "query",
                "passage",
                ":",
                "python",
                "sql",
                "developer",
                "data",
                "analysis",
            ]
        )
    )
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), model_max_length=512)
    tokenizer.save_pretrained(path)
    torch.manual_seed(123)
    BertModel(
        BertConfig(
            vocab_size=13,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).save_pretrained(path)
    (path / "resumeanalysis_model.json").write_text(
        json.dumps(
            {
                "model_id": "intfloat/multilingual-e5-small",
                "revision": "authored-random-test-fixture",
                "sha256": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest(),
            }
        )
    )
    return TextEncoder(path, tmp_path / "legacy-cache", max_length=16, overlap=2, batch_size=3)


def test_real_local_transformer_batching_matches_legacy_pooling_and_keeps_cache_separate(tmp_path):
    encoder = tiny_e5(tmp_path)
    texts = ["python", "sql data analysis", "python sql " * 30, "", "SQL и анализ данных"]
    expected = np.stack([encoder.encode(text, "passage") for text in texts])
    old_files = {p.name: p.read_bytes() for p in encoder.cache_dir.glob("*")}
    retriever = ExactRetriever(encoder, tmp_path / "retrieval-cache")
    actual = retriever.encode_many(texts, "passage")
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert all(not p.requires_grad for p in encoder.model.parameters())
    assert {p.name: p.read_bytes() for p in encoder.cache_dir.glob("*")} == old_files
    before = retriever.cache_stats["transformer_batches"]
    assert np.array_equal(actual, retriever.encode_many(texts))
    assert retriever.cache_stats["transformer_batches"] == before
    encoder.max_length = 20
    with pytest.raises(ValueError, match="configuration changed"):
        retriever.encode_many(["python"])


def test_short_vacancies_share_actual_transformer_batches(tmp_path):
    encoder = tiny_e5(tmp_path)
    batches = []

    def capture(module, args, kwargs):
        batches.append(kwargs["input_ids"].shape[0])

    handle = encoder.model.register_forward_pre_hook(capture, with_kwargs=True)
    retriever = ExactRetriever(encoder)
    retriever.encode_many(["python", "sql", "data analysis"], "passage")
    handle.remove()
    assert batches == [3]
    assert retriever.cache_stats["transformer_batches"] == 1


def test_e5_checksum_is_verified_before_retrieval_inference(tmp_path):
    encoder = tiny_e5(tmp_path)
    path = tmp_path / "tiny/model.safetensors"
    path.write_bytes(path.read_bytes() + b"tampered after safe model loading")
    with pytest.raises(ValueError, match="weights checksum"):
        ExactRetriever(encoder)


def test_tokenizer_content_changes_cache_identity_even_with_same_revision(tmp_path):
    encoder = tiny_e5(tmp_path)
    first = ExactRetriever(encoder)
    path = tmp_path / "tiny/tokenizer_config.json"
    metadata = json.loads(path.read_text())
    metadata["test_only_metadata"] = "distinct source configuration"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="asset changed"):
        first.encode_many(["python"])
    second = ExactRetriever(encoder)
    assert first.identity != second.identity


def test_missing_e5_tokenizer_asset_fails_before_retrieval(tmp_path):
    encoder = tiny_e5(tmp_path)
    (tmp_path / "tiny/tokenizer.json").unlink()
    with pytest.raises(ValueError, match="Missing/unreadable"):
        ExactRetriever(encoder)


def test_e5_revision_mismatch_fails_before_inference(tmp_path):
    encoder = tiny_e5(tmp_path)
    path = tmp_path / "tiny/resumeanalysis_model.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "another-snapshot"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest mismatch"):
        ExactRetriever(encoder)


def test_e5_manifest_requires_an_object(tmp_path):
    encoder = tiny_e5(tmp_path)
    (tmp_path / "tiny/resumeanalysis_model.json").write_text("[]")
    with pytest.raises(ValueError, match="object"):
        ExactRetriever(encoder)


def test_e5_model_cannot_be_made_trainable_after_cache_population(tmp_path):
    encoder = tiny_e5(tmp_path)
    retriever = ExactRetriever(encoder)
    retriever.encode_many(["python"])
    encoder.model.train()
    with pytest.raises(ValueError, match="frozen eval model"):
        retriever.encode_many(["python"])


@pytest.mark.skipif(
    os.getenv("RESUMEANALYSIS_RUN_E5_TESTS") != "1", reason="opt-in downloaded E5 integration"
)
def test_downloaded_e5_batch_equivalence_ru_en_and_long_documents(tmp_path):
    encoder = TextEncoder(
        Path(os.getenv("RESUMEANALYSIS_E5_PATH", "artifacts/e5")),
        tmp_path / "legacy-cache",
        max_length=128,
        overlap=16,
        batch_size=4,
    )
    texts = [
        "Python developer with SQL",
        "Разработка сервисов и анализ данных",
        "Python SQL " * 250,
    ]
    expected = np.stack([encoder.encode(text, "passage") for text in texts])
    actual = ExactRetriever(encoder, tmp_path / "retrieval-cache").encode_many(texts)
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)
