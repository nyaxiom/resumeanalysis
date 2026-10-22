import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from resumeanalysis.embeddings.encoder import TextEncoder


def test_real_transformer_chunking_prefix_cache_and_freeze(tmp_path):
    # Tiny LOCAL randomly initialized transformer tests mechanics, never quality.
    path = tmp_path / "tiny"
    path.mkdir()
    (path / "vocab.txt").write_text(
        "\n".join(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", "passage", ":", "python", "sql"]
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
    (path / "resumeanalysis_model.json").write_text(
        json.dumps({"model_id": "intfloat/multilingual-e5-small", "revision": "unit-test-fixture"})
    )
    encoder = TextEncoder(path, tmp_path / "cache", max_length=16, overlap=2, batch_size=2)
    lengths, prefixes = [], []

    def capture(module, args, kwargs):
        ids = kwargs["input_ids"]
        lengths.append(ids.shape[1])
        prefixes.extend(ids[:, 1].tolist())

    handle = encoder.model.register_forward_pre_hook(capture, with_kwargs=True)
    a = encoder.encode("python " * 40)
    assert len(lengths) > 1 and max(lengths) <= 16
    assert set(prefixes) == {tokenizer.convert_tokens_to_ids("query")}
    assert np.linalg.norm(a) == pytest.approx(1)
    before = len(lengths)
    assert np.array_equal(a, encoder.encode("python " * 40))
    assert len(lengths) == before  # actual cache hit, no transformer calls
    prefixes.clear()
    encoder.encode("python " * 40, "passage")
    assert set(prefixes) == {tokenizer.convert_tokens_to_ids("passage")}
    assert all(not p.requires_grad for p in encoder.model.parameters())
    handle.remove()
    assert torch.isfinite(torch.from_numpy(encoder.encode(""))).all()
    with pytest.raises(ValueError):
        encoder.encode("x", "invalid")


@pytest.mark.skipif(
    os.getenv("RESUMEANALYSIS_RUN_E5_TESTS") != "1", reason="opt-in downloaded E5 integration"
)
def test_downloaded_e5_ru_en_and_long_text(tmp_path):
    path = Path(os.getenv("RESUMEANALYSIS_E5_PATH", "artifacts/e5"))
    encoder = TextEncoder(path, tmp_path, batch_size=2)
    assert encoder.dimension == 384
    assert encoder.encode("Python, SQL и машинное обучение").shape == (384,)
    a = encoder.encode("Python SQL " * 600)
    b = encoder.encode("Python SQL " * 600)
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1)
