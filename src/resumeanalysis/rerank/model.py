"""Frozen CPU joint encoder with explicit long-document coverage and offline assets."""

import hashlib
import itertools
import json
from typing import Protocol

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from resumeanalysis.rerank.assets import asset_identity, verify_assets
from resumeanalysis.rerank.config import RerankConfig


class PairScorer(Protocol):
    identity: str
    last_diagnostics: list[dict]

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _budgets(left, right, capacity):
    a, b = min(left, capacity // 2), min(right, capacity - capacity // 2)
    spare = capacity - a - b
    extra = min(left - a, spare)
    a += extra
    b += min(right - b, spare - extra)
    return a, b


def _chunks(tokens, capacity, overlap):
    if not tokens:
        return [[]]
    if capacity <= overlap:
        raise ValueError("Window overlap exceeds content capacity")
    return [
        tokens[i : i + capacity]
        for i in range(0, len(tokens), capacity - overlap)
        if i == 0 or i + overlap < len(tokens)
    ]


class CrossEncoder:
    def __init__(self, config: RerankConfig | None = None):
        self.config = config or RerankConfig()
        manifest = verify_assets(self.config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.model.eval().requires_grad_(False).to("cpu")
        if (
            self.model.config.model_type != "xlm-roberta"
            or self.model.config.num_labels != 1
            or self.model.config.hidden_size != 384
            or self.model.config.num_hidden_layers != 12
        ):
            raise ValueError("Cross-encoder architecture differs from the reviewed model")
        special = self.tokenizer.num_special_tokens_to_add(pair=True)
        if self.config.max_length > min(512, self.tokenizer.model_max_length):
            raise ValueError("Cross-encoder sequence limit exceeds tokenizer capacity")
        self.capacity = self.config.max_length - special
        self.identity = (
            "cross-encoder-v1:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "assets": asset_identity(manifest),
                        "config": self.config.model_dump(exclude={"model_path"}),
                        "transformers": transformers.__version__,
                        "pair_packing": "balanced-heads-v1;cartesian-windows-max-v1",
                        "runtime_guard": "frozen-runtime-v1",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        self.last_diagnostics = []
        self._loaded_runtime = self._runtime_signature()

    def _runtime_signature(self):
        return (
            id(self.model),
            id(self.tokenizer),
            self.config.model_dump_json(),
            self.capacity,
            self.model.training,
            self.tokenizer.model_max_length,
            self.model.config.model_type,
            self.model.config.num_labels,
            self.model.config.hidden_size,
            self.model.config.num_hidden_layers,
            tuple(
                (
                    id(parameter),
                    parameter._version,
                    parameter.requires_grad,
                    str(parameter.device),
                    str(parameter.dtype),
                )
                for parameter in self.model.parameters()
            ),
        )

    def _assert_runtime(self):
        if self._runtime_signature() != self._loaded_runtime:
            raise ValueError("Cross-encoder runtime identity changed; reload the verified assets")

    def _pack(self, a, b):
        aa, bb = _budgets(len(a), len(b), self.capacity)
        overlong = len(a) + len(b) > self.capacity
        policy = self.config.long_document_policy
        if overlong and policy == "error":
            raise ValueError(
                "Resume/vacancy pair exceeds max_length; choose explicit truncate/window policy"
            )
        if overlong and policy == "window":
            ac = _chunks(a, aa, min(self.config.overlap_tokens, max(0, aa - 1)))
            bc = _chunks(b, bb, min(self.config.overlap_tokens, max(0, bb - 1)))
            if len(ac) * len(bc) > self.config.max_windows:
                raise ValueError(
                    "Long-document window count exceeds max_windows; no silent omission"
                )
            pairs = list(itertools.product(ac, bc))
            kept_a, kept_b = len(a), len(b)
        else:
            pairs = [(a[:aa], b[:bb])]
            kept_a, kept_b = min(len(a), aa), min(len(b), bb)
        features = []
        for left, right in pairs:
            ids = self.tokenizer.build_inputs_with_special_tokens(left, right)
            if len(ids) > self.config.max_length:
                raise ValueError("Pair packing exceeds the declared sequence limit")
            features.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        diagnostics = {
            "resume_tokens": len(a),
            "vacancy_tokens": len(b),
            "resume_tokens_retained": kept_a,
            "vacancy_tokens_retained": kept_b,
            "resume_tokens_discarded": len(a) - kept_a,
            "vacancy_tokens_discarded": len(b) - kept_b,
            "truncated": kept_a < len(a) or kept_b < len(b),
            "windows": len(features),
            "policy": policy,
            "aggregation": "max" if len(features) > 1 else "single",
        }
        return features, diagnostics

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.last_diagnostics = []
        self._assert_runtime()
        if not isinstance(pairs, list) or any(
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(not isinstance(value, str) for value in pair)
            for pair in pairs
        ):
            raise TypeError("Cross-encoder input must be a list of text pairs")
        if not pairs:
            return []
        unique = list(dict.fromkeys(text for pair in pairs for text in pair))
        tokenized = self.tokenizer(
            unique,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        lookup = dict(zip(unique, tokenized, strict=True))
        features, owners, diagnostics = [], [], []
        for owner, (left, right) in enumerate(pairs):
            packed, diagnostic = self._pack(lookup[left], lookup[right])
            features.extend(packed)
            owners.extend([owner] * len(packed))
            diagnostics.append(diagnostic)
        scores = [[] for _ in pairs]
        with torch.inference_mode():
            for start in range(0, len(features), self.config.batch_size):
                inputs = self.tokenizer.pad(
                    features[start : start + self.config.batch_size],
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(**inputs).logits
                if logits.ndim != 2 or logits.shape != (len(inputs["input_ids"]), 1):
                    raise ValueError("Unexpected cross-encoder logit shape")
                values = logits[:, 0].float().cpu().tolist()
                if not np.isfinite(values).all():
                    raise ValueError("Cross-encoder produced nonfinite logits")
                for owner, value in zip(owners[start : start + len(values)], values, strict=True):
                    scores[owner].append(float(value))
        self.last_diagnostics = diagnostics
        self._assert_runtime()
        return [max(values) for values in scores]
