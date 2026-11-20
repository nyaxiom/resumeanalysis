"""Frozen E5 with the existing masked pooling/chunking, batched across vacancies."""

import hashlib
import json
import os
import tempfile
import threading

import numpy as np

from resumeanalysis.config import Settings


class Embeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self.tokenizer = None
        self._identity = None
        self.lock = threading.RLock()

    @property
    def identity(self) -> str:
        with self.lock:
            if self._identity is None:
                s = self.settings
                config = [
                    "e5-passage-title-description-v1",
                    s.e5_model,
                    s.e5_revision,
                    s.e5_device,
                    s.e5_max_length,
                    s.e5_overlap,
                ]
                if s.e5_local_path:
                    # Content identity includes tokenizer and weights. Replacing local
                    # assets between restarts invalidates both caches and user models.
                    files = sorted(
                        p
                        for p in s.e5_local_path.rglob("*")
                        if p.is_file()
                        and p.suffix in {".json", ".safetensors", ".model", ".txt"}
                        and ".cache" not in p.parts
                    )
                    if not any(p.suffix == ".safetensors" for p in files):
                        raise FileNotFoundError("Local E5 safetensors are missing")
                    for path in files:
                        with path.open("rb") as stream:
                            digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        config.append([str(path.relative_to(s.e5_local_path)), digest])
                self._identity = hashlib.sha256(json.dumps(config).encode()).hexdigest()
            return self._identity

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        s = self.settings
        device = s.e5_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        source = str(s.e5_local_path or s.e5_model)
        options = dict(
            revision=s.e5_revision, local_files_only=bool(s.e5_local_path), trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(source, **options)
        model = AutoModel.from_pretrained(source, use_safetensors=True, **options)
        model.eval().requires_grad_(False).to(device)
        self.tokenizer, self.model = tokenizer, model

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self.lock:
            self._load()
            cache = self.settings.embedding_cache_dir
            if cache:
                cache.mkdir(parents=True, exist_ok=True)
            unique = list(dict.fromkeys(texts))
            vectors, missing, paths = {}, [], {}
            for text in unique:
                key = hashlib.sha256(f"{self.identity}\0{text}".encode()).hexdigest()
                path = cache / f"{key}.npy" if cache else None
                paths[text] = path
                try:
                    vector = np.load(path, allow_pickle=False) if path else None
                    if isinstance(vector, np.ndarray) and (
                        vector.shape == (self.model.config.hidden_size,)
                        and np.issubdtype(vector.dtype, np.floating)
                        and np.isfinite(vector).all()
                        and np.isclose(np.linalg.norm(vector), 1, atol=1e-5)
                    ):
                        vectors[text] = vector
                        continue
                except (OSError, ValueError, EOFError, TypeError):
                    pass  # Missing or damaged cache entries are safely recomputed.
                missing.append(text)
            if missing:
                for text, vector in zip(missing, self._encode(missing), strict=True):
                    vectors[text] = vector
                    path = paths[text]
                    if path:
                        temporary = None
                        try:
                            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                                temporary = stream.name
                                np.save(stream, vector, allow_pickle=False)
                            os.replace(temporary, path)
                        finally:
                            if temporary and os.path.exists(temporary):
                                os.unlink(temporary)
            return [vectors[text].tolist() for text in texts]

    def _encode(self, texts):
        import torch

        s, tokenizer = self.settings, self.tokenizer
        prefix = tokenizer.encode("passage: ", add_special_tokens=False)
        capacity = min(s.e5_max_length, tokenizer.model_max_length) - len(prefix)
        capacity -= tokenizer.num_special_tokens_to_add(False)
        if capacity <= s.e5_overlap:
            raise ValueError("E5 overlap exceeds the available token budget")
        chunks, owners, weights = [], [], []
        for owner, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            for start in range(0, max(1, len(tokens)), capacity - s.e5_overlap):
                part = tokens[start : start + capacity]
                chunks.append(tokenizer.build_inputs_with_special_tokens(prefix + part))
                owners.append(owner)
                weights.append(max(1, len(part) - (s.e5_overlap if start else 0)))
                if start + capacity >= len(tokens):
                    break
        result = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float64)
        with torch.inference_mode():
            for offset in range(0, len(chunks), s.e5_batch_size):
                batch = chunks[offset : offset + s.e5_batch_size]
                inputs = tokenizer.pad(
                    [{"input_ids": c, "attention_mask": [1] * len(c)} for c in batch],
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled, dim=1).float().cpu().numpy()
                for i, vector in enumerate(vectors, offset):
                    result[owners[i]] += vector * weights[i]
        result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) < 0.99):
            raise RuntimeError("E5 returned invalid embeddings")
        return result.astype(np.float32)
