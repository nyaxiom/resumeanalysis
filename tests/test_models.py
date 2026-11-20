from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BatchEncoding, BertConfig, BertModel, BertTokenizerFast

from resumeanalysis.config import Settings
from resumeanalysis.cover_letter import ContextTooLong, InvalidLetter, Qwen
from resumeanalysis.embeddings import Embeddings


@pytest.fixture
def tiny_e5(tmp_path):
    # Mechanics only: a tiny locally generated transformer, no downloaded weights.
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
                "sales",
            ]
        )
    )
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), model_max_length=512)
    tokenizer.save_pretrained(path)
    BertModel(
        BertConfig(
            vocab_size=10,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).save_pretrained(path)
    return Settings(
        e5_local_path=path,
        e5_max_length=16,
        e5_overlap=2,
        e5_batch_size=2,
        embedding_cache_dir=tmp_path / "cache",
    )


def test_e5_batches_documents_chunks_and_cache(tiny_e5):
    encoder = Embeddings(tiny_e5)
    assert encoder.model is None
    encoder._load()
    batches, prefixes = [], []

    def capture(module, args, kwargs):
        ids = kwargs["input_ids"]
        batches.append(tuple(ids.shape))
        prefixes.extend(ids[:, 1].tolist())

    handle = encoder.model.register_forward_pre_hook(capture, with_kwargs=True)
    a = encoder.encode_many(["python", "sales"])
    assert len(batches) == 1 and batches[0][0] == 2
    assert set(prefixes) == {encoder.tokenizer.convert_tokens_to_ids("passage")}
    assert np.allclose(np.linalg.norm(a, axis=1), 1)
    before = len(batches)
    assert a == encoder.encode_many(["python", "sales"])
    assert len(batches) == before
    long = "python " * 40
    vectors = encoder.encode_many([long, long])
    assert vectors[0] == vectors[1]
    assert max(length for _, length in batches) <= 16
    assert len(batches) > before + 1
    assert all(not p.requires_grad for p in encoder.model.parameters())
    cache_files = list(tiny_e5.embedding_cache_dir.glob("*.npy"))
    assert len(cache_files) == 3
    for cache in cache_files:
        cache.write_bytes(b"corrupt")
    assert np.allclose(a, encoder.encode_many(["python", "sales"]))
    for cache in cache_files:
        np.save(cache, np.array(["bad"] * 8), allow_pickle=False)
    assert np.allclose(a, encoder.encode_many(["python", "sales"]))
    handle.remove()


def test_local_asset_changes_invalidate_embedding_cache(tiny_e5):
    original = Embeddings(tiny_e5)
    old = original.identity
    original.encode_many(["python"])
    config = tiny_e5.e5_local_path / "tokenizer_config.json"
    config.write_text(config.read_text() + "\n")
    changed = Embeddings(tiny_e5)
    assert changed.identity != old
    changed.encode_many(["python"])
    assert len(list(tiny_e5.embedding_cache_dir.glob("*.npy"))) == 2
    settings = tiny_e5.model_copy(update={"e5_overlap": 3})
    assert Embeddings(settings).identity != changed.identity


def test_cache_can_be_disabled(tiny_e5):
    encoder = Embeddings(tiny_e5.model_copy(update={"embedding_cache_dir": None}))
    assert np.allclose(encoder.encode_many(["python"]), encoder.encode_many(["python"]))
    assert not tiny_e5.embedding_cache_dir.exists()


class TokenizerStub:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return "prompt"

    def __call__(self, text, **kwargs):
        return BatchEncoding(
            {"input_ids": torch.tensor([[1, 3, 4]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        )

    def decode(self, tokens, **kwargs):
        assert list(tokens) == [9, 2]
        return "Finished letter."


class ModelStub:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=4096)
    generation_config = SimpleNamespace(eos_token_id=2)

    def generate(self, **kwargs):
        assert not kwargs["do_sample"]
        assert not torch.is_grad_enabled()
        return torch.tensor([[1, 3, 4, 9, 2]])


def test_qwen_runtime_only_decodes_completion_and_checks_context():
    backend = Qwen(Settings())
    backend.tokenizer, backend.model = TokenizerStub(), ModelStub()
    assert backend.generate([{"role": "user", "content": "resume"}]) == "Finished letter."
    backend.model.config = SimpleNamespace(max_position_embeddings=4)
    with pytest.raises(ContextTooLong):
        backend.generate([])


def test_qwen_rejects_unfinished_generation():
    backend = Qwen(Settings(qwen_max_new_tokens=32))
    backend.tokenizer, backend.model = TokenizerStub(), ModelStub()
    backend.model.generate = lambda **kw: torch.tensor([[1, 3, 4] + [9] * 32])
    with pytest.raises(InvalidLetter, match="before finishing"):
        backend.generate([])
