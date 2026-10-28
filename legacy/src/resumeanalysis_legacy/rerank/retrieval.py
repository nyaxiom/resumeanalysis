"""Exact cosine retrieval with separate, identity-bound embedding caches.

The existing E5 encoder and its checkpoint/cache identity remain unchanged.
Vacancy chunks can share transformer batches here, with the original E5 pooling
and chunk weights. Retrieval writes only its own cache namespace.
"""

import hashlib
import json
import os
import tempfile
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from resumeanalysis.embeddings.encoder import Encoder, TextEncoder
from resumeanalysis.jsonio import loads
from resumeanalysis.schemas import Profile, Vacancy
from resumeanalysis_legacy.features.feature_builder import profile_text, vacancy_text


@dataclass(frozen=True)
class RetrievedCandidate:
    index: int
    vacancy: Vacancy
    cosine_score: float


def _file_sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _e5_source_paths(encoder):
    return (
        Path(encoder.model.name_or_path).resolve(),
        Path(encoder.tokenizer.name_or_path).resolve(),
    )


def _e5_assets(encoder):
    """Strengthen the new retrieval boundary without changing legacy E5 loading."""
    model_dir, tokenizer_dir = _e5_source_paths(encoder)
    if model_dir != tokenizer_dir:
        raise ValueError("E5 model and tokenizer must come from the same local snapshot")
    manifest_path = model_dir / "resumeanalysis_model.json"
    try:
        manifest = loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("E5 retrieval requires a valid local asset manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("E5 retrieval requires an object in its local asset manifest")
    expected = manifest.get("sha256")
    if (
        manifest.get("model_id") != "intfloat/multilingual-e5-small"
        or not isinstance(manifest.get("revision"), str)
        or not manifest["revision"]
        or not isinstance(expected, str)
        or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected)
        or not encoder.identity.startswith(f"e5:{manifest['revision']}:mean-v1:")
    ):
        raise ValueError("E5 retrieval asset identity/checksum manifest mismatch")
    names = [
        "resumeanalysis_model.json",
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    names += [
        name
        for name in ("README.md", "vocab.txt", "added_tokens.json", "sentencepiece.bpe.model")
        if (model_dir / name).exists()
    ]
    checksums, stamps = {}, {}
    for name in names:
        path = model_dir / name
        try:
            before = _stamp(path)
            checksums[name] = _file_sha(path)
            stamps[path] = _stamp(path)
        except OSError as exc:
            raise ValueError(f"Missing/unreadable E5 retrieval asset: {name}") from exc
        if before != stamps[path]:
            raise ValueError("E5 snapshot changed while computing asset identities")
    if checksums["model.safetensors"] != expected:
        raise ValueError("E5 weights checksum does not match the pinned local manifest")
    return checksums, stamps


def _configuration(encoder):
    if not isinstance(encoder.identity, str) or not encoder.identity:
        raise ValueError("Retrieval requires a nonempty encoder identity")
    if type(encoder.dimension) is not int or encoder.dimension < 1:
        raise ValueError("Retrieval requires a positive encoder dimension")
    result = {
        "identity": encoder.identity,
        "dimension": encoder.dimension,
        "implementation": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
    }
    if isinstance(encoder, TextEncoder):
        if (
            type(encoder.max_length) is not int
            or not 16 <= encoder.max_length <= min(512, encoder.tokenizer.model_max_length)
            or type(encoder.overlap) is not int
            or not 0 <= encoder.overlap < encoder.max_length - 8
            or type(encoder.batch_size) is not int
            or encoder.batch_size < 1
            or encoder.model.training
            or any(parameter.requires_grad for parameter in encoder.model.parameters())
        ):
            raise ValueError(
                "E5 retrieval requires valid chunk configuration and a frozen eval model"
            )
        result.update(
            max_length=encoder.max_length,
            overlap=encoder.overlap,
            batch_size=encoder.batch_size,
            training=encoder.model.training,
        )
    return result


def _unit_vector(value, dimension):
    vector = np.asarray(value)
    if vector.shape != (dimension,) or vector.dtype.kind not in "fiu":
        raise ValueError("Invalid retrieval embedding shape or dtype")
    vector = vector.astype(np.float32, copy=True)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite retrieval embedding")
    # As in sklearn cosine similarity, an empty/zero diagnostic embedding has
    # similarity zero. It must not produce NaN or an arbitrary positive score.
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm:
        vector /= norm
    return vector


def _e5_batch(encoder: TextEncoder, texts: list[str], role: str):
    """Mirror TextEncoder.encode chunk/pool semantics without modifying that API."""
    prefix = encoder.tokenizer.encode(role + ": ", add_special_tokens=False)
    capacity = encoder.max_length - len(prefix) - encoder.tokenizer.num_special_tokens_to_add(False)
    if capacity < 1 or not 0 <= encoder.overlap < capacity:
        raise ValueError("Invalid E5 retrieval chunk capacity/overlap")
    chunks, owners, weights = [], [], [[] for _ in texts]
    for owner, text in enumerate(texts):
        tokens = encoder.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for start in range(0, max(1, len(tokens)), capacity - encoder.overlap):
            part = tokens[start : start + capacity]
            chunks.append(encoder.tokenizer.build_inputs_with_special_tokens(prefix + part))
            owners.append(owner)
            weights[owner].append(max(1, len(part) - (encoder.overlap if start else 0)))
            if start + capacity >= len(tokens):
                break
    vectors = [[] for _ in texts]
    batches = 0
    with torch.inference_mode():
        for offset in range(0, len(chunks), encoder.batch_size):
            inputs = encoder.tokenizer.pad(
                [
                    {"input_ids": chunk, "attention_mask": [1] * len(chunk)}
                    for chunk in chunks[offset : offset + encoder.batch_size]
                ],
                padding=True,
                return_tensors="pt",
            )
            hidden = encoder.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
            for owner, vector in zip(owners[offset : offset + encoder.batch_size], pooled):
                vectors[owner].append(vector)
            batches += 1
    # Original E5 normalizes each pooled chunk before weighted document pooling.
    return [
        _unit_vector(np.average(parts, axis=0, weights=weight), encoder.dimension)
        for parts, weight in zip(vectors, weights)
    ], batches


class ExactRetriever:
    def __init__(
        self,
        encoder: Encoder,
        cache_dir: str | Path | None = None,
        *,
        memory_cache_size: int = 4096,
    ):
        if type(memory_cache_size) is not int or memory_cache_size < 0:
            raise ValueError("memory_cache_size must be a nonnegative integer")
        self.encoder = encoder
        self._configuration = _configuration(encoder)
        self._asset_stamps, self._asset_sources = {}, None
        asset_checksums = None
        if isinstance(encoder, TextEncoder):
            asset_checksums, self._asset_stamps = _e5_assets(encoder)
            self._asset_sources = _e5_source_paths(encoder)
        self.identity = (
            "exact-cosine-v1:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "encoder": self._configuration,
                        "asset_sha256": asset_checksums,
                        "text": "existing-profile_text-vacancy_text-no-title-v1",
                        "pooling": "e5-mean-weighted-chunks-v1",
                        "normalization": "l2-float32-zero-to-zero",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache_size = memory_cache_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.cache_stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "invalid_disk_entries": 0,
            "encoded_texts": 0,
            "transformer_batches": 0,
        }

    def _check_encoder(self):
        if _configuration(self.encoder) != self._configuration:
            raise ValueError(
                "Encoder identity/configuration changed after retriever initialization"
            )
        if self._asset_sources is not None:
            if _e5_source_paths(self.encoder) != self._asset_sources:
                raise ValueError("E5 snapshot source changed after retrieval initialization")
            for path, expected in self._asset_stamps.items():
                try:
                    same = _stamp(path) == expected
                except OSError:
                    same = False
                if not same:
                    raise ValueError(
                        "E5 model/tokenizer asset changed after retrieval initialization"
                    )

    def _key(self, text, role):
        return hashlib.sha256(f"{self.identity}\0{role}\0{text}".encode()).hexdigest()

    def _remember(self, key, vector):
        if self.memory_cache_size:
            self._cache[key] = vector.copy()
            self._cache.move_to_end(key)
            while len(self._cache) > self.memory_cache_size:
                self._cache.popitem(last=False)

    def _disk_path(self, key):
        path = self.cache_dir / (key + ".npz")
        if not path.resolve().is_relative_to(self.cache_dir):
            raise ValueError("Embedding cache path escapes its configured directory")
        return path

    def _load(self, key, text, role):
        if key in self._cache:
            self.cache_stats["memory_hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        if self.cache_dir is None:
            return None
        path = self._disk_path(key)
        if not path.exists():
            return None
        try:
            if path.stat().st_size > max(1024 * 1024, self.encoder.dimension * 16 + 4096):
                raise ValueError("Oversized embedding cache entry")
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"vector", "identity", "content_sha256", "role"}:
                    raise ValueError("Unexpected embedding cache fields")
                if (
                    data["identity"].shape != ()
                    or data["identity"].item() != self.identity
                    or data["role"].shape != ()
                    or data["role"].item() != role
                    or data["content_sha256"].shape != ()
                    or data["content_sha256"].item() != hashlib.sha256(text.encode()).hexdigest()
                ):
                    raise ValueError("Embedding cache identity mismatch")
                vector = _unit_vector(data["vector"], self.encoder.dimension)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            self.cache_stats["invalid_disk_entries"] += 1
            return None
        self.cache_stats["disk_hits"] += 1
        self._remember(key, vector)
        return vector.copy()

    def _store(self, key, text, role, vector):
        self._remember(key, vector)
        if self.cache_dir is None:
            return
        destination = self._disk_path(key)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as f:
                temporary = Path(f.name)
                np.savez(
                    f,
                    vector=vector,
                    identity=np.asarray(self.identity),
                    content_sha256=np.asarray(hashlib.sha256(text.encode()).hexdigest()),
                    role=np.asarray(role),
                )
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def encode_many(self, texts: Sequence[str], role: str = "passage") -> np.ndarray:
        """Deduplicate before inference; return normalized rows in original order."""
        with self._lock:
            return self._encode_many(texts, role)

    def _encode_many(self, texts, role):
        self._check_encoder()
        if role not in {"query", "passage"}:
            raise ValueError("Embedding role must be query or passage")
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a sequence of strings")
        if not texts:
            return np.empty((0, self.encoder.dimension), dtype=np.float32)
        unique = dict.fromkeys(texts)
        vectors, missing = {}, []
        for text in unique:
            key = self._key(text, role)
            vector = self._load(key, text, role)
            if vector is None:
                missing.append(text)
            else:
                vectors[text] = vector
        if missing:
            if isinstance(self.encoder, TextEncoder):
                computed, batches = _e5_batch(self.encoder, missing, role)
                self.cache_stats["transformer_batches"] += batches
            elif callable(getattr(self.encoder, "encode_many", None)):
                computed = self.encoder.encode_many(missing, role=role)
            else:
                computed = [self.encoder.encode(text, role) for text in missing]
            if len(computed) != len(missing):
                raise ValueError("Batch encoder returned the wrong number of embeddings")
            checked = [_unit_vector(vector, self.encoder.dimension) for vector in computed]
            self._check_encoder()
            for text, vector in zip(missing, checked):
                vectors[text] = vector
                self._store(self._key(text, role), text, role, vector)
            self.cache_stats["encoded_texts"] += len(missing)
        return np.stack([vectors[text] for text in texts])

    def encode_profile(self, profile: Profile | dict) -> np.ndarray:
        profile = profile if isinstance(profile, Profile) else Profile.model_validate(profile)
        return self.encode_many([profile_text(profile)], role="query")[0]

    def vacancy_matrix(self, vacancies: Sequence[Vacancy | dict]) -> np.ndarray:
        vacancies = [v if isinstance(v, Vacancy) else Vacancy.model_validate(v) for v in vacancies]
        return self.encode_many([vacancy_text(v) for v in vacancies], role="passage")

    def score_all(
        self, profile: Profile | dict, vacancies: Sequence[Vacancy | dict]
    ) -> list[RetrievedCandidate]:
        vacancies = [v if isinstance(v, Vacancy) else Vacancy.model_validate(v) for v in vacancies]
        if not vacancies:
            self._check_encoder()
            return []
        query = self.encode_profile(profile)
        matrix = self.vacancy_matrix(vacancies)
        scores = np.clip(matrix @ query, -1.0, 1.0)
        return [
            RetrievedCandidate(index=index, vacancy=vacancy, cosine_score=float(score))
            for index, (vacancy, score) in enumerate(zip(vacancies, scores))
        ]

    def retrieve(
        self,
        profile: Profile | dict,
        vacancies: Sequence[Vacancy | dict],
        preselect_k: int = 20,
    ) -> list[RetrievedCandidate]:
        if type(preselect_k) is not int or preselect_k < 1:
            raise ValueError("preselect_k must be a positive integer")
        candidates = self.score_all(profile, vacancies)
        return sorted(candidates, key=lambda candidate: (-candidate.cosine_score, candidate.index))[
            :preselect_k
        ]
