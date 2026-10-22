"""Frozen multilingual E5, token-aware chunks, masked mean pooling and disk cache."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class Encoder(Protocol):
    dimension: int
    identity: str

    def encode(self, text: str, role: str = "query") -> np.ndarray: ...


class TextEncoder:
    def __init__(
        self,
        model_path: str | Path,
        cache_dir: str | Path = "artifacts/embeddings",
        max_length: int = 512,
        overlap: int = 32,
        batch_size: int = 8,
    ):
        model_path = Path(model_path)
        manifest = json.loads((model_path / "resumeanalysis_model.json").read_text())
        if manifest["model_id"] != "intfloat/multilingual-e5-small":
            raise ValueError("Expected multilingual-e5-small manifest")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        self.model.eval().requires_grad_(False).to("cpu")
        self.dimension = self.model.config.hidden_size
        limit = min(512, self.tokenizer.model_max_length)
        if not 16 <= max_length <= limit or not 0 <= overlap < max_length - 8:
            raise ValueError("Invalid E5 chunk configuration")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.max_length, self.overlap, self.batch_size = max_length, overlap, batch_size
        self.identity = f"e5:{manifest['revision']}:mean-v1:{max_length}:{overlap}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(self, text: str, role: str = "query") -> np.ndarray:
        if role not in {"query", "passage"}:
            raise ValueError("E5 role must be query or passage")
        key = hashlib.sha256(f"{self.identity}\0{role}\0{text}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False)
            if vector.shape == (self.dimension,) and np.isfinite(vector).all():
                return vector
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        prefix = self.tokenizer.encode(role + ": ", add_special_tokens=False)
        capacity = self.max_length - len(prefix) - self.tokenizer.num_special_tokens_to_add(False)
        if self.overlap >= capacity:
            raise ValueError("Overlap exceeds content chunk size")
        chunks, weights = [], []
        for start in range(0, max(1, len(tokens)), capacity - self.overlap):
            part = tokens[start : start + capacity]
            chunks.append(self.tokenizer.build_inputs_with_special_tokens(prefix + part))
            weights.append(max(1, len(part) - (self.overlap if start else 0)))
            if start + capacity >= len(tokens):
                break
        vectors = []
        with torch.inference_mode():
            for offset in range(0, len(chunks), self.batch_size):
                inputs = self.tokenizer.pad(
                    [
                        {"input_ids": c, "attention_mask": [1] * len(c)}
                        for c in chunks[offset : offset + self.batch_size]
                    ],
                    padding=True,
                    return_tensors="pt",
                )
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.extend(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        vector = np.average(vectors, axis=0, weights=weights).astype(np.float32)
        vector /= max(np.linalg.norm(vector), 1e-12)
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npy", delete=False) as f:
            np.save(f, vector, allow_pickle=False)
            temporary = f.name
        os.replace(temporary, path)
        return vector


class LexicalSmokeEncoder:
    """Explicit offline diagnostic only. Never silently substituted for pretrained E5."""

    dimension = 384
    identity = "lexical-smoke-hashing-v1:384"

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=self.dimension, alternate_sign=False, ngram_range=(1, 2), norm="l2"
        )

    def encode(self, text: str, role: str = "query") -> np.ndarray:
        if role not in {"query", "passage"}:
            raise ValueError("Unknown role")
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
