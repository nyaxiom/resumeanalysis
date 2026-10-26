"""Local safetensors loading and bounded, schema-constrained greedy generation."""

import json
import threading
from pathlib import Path
from typing import Protocol

from resumeanalysis.data.download import sha256_file
from resumeanalysis.qwen.config import QwenConfig


class Generator(Protocol):
    def generate(self, messages: list[dict], schema: dict) -> str: ...


class TransformersGenerator:
    def __init__(self, config: QwenConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        path = Path(config.model_path)
        if not path.is_dir() or not (path / "resumeanalysis_qwen.json").is_file():
            raise FileNotFoundError(
                "Qwen weights/manifest unavailable; run resumeanalysis-qwen download"
            )
        manifest = json.loads((path / "resumeanalysis_qwen.json").read_text())
        if (manifest.get("model_id"), manifest.get("revision")) != (
            config.model_id,
            config.revision,
        ):
            raise ValueError("Local Qwen model identity differs from configuration")
        if manifest.get("license") != "apache-2.0" or not manifest.get("sha256"):
            raise ValueError("Local Qwen manifest lacks license/checksum provenance")
        for name, expected in manifest["sha256"].items():
            artifact = (path / name).resolve()
            if (
                not artifact.is_relative_to(path.resolve())
                or not artifact.is_file()
                or sha256_file(artifact) != expected
            ):
                raise ValueError("Local Qwen artifact checksum mismatch")
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no automatic model or quantization fallback")
        if device == "cpu" and (config.quantization != "none" or config.dtype != "float32"):
            raise ValueError("CPU requires quantization=none and dtype=float32")
        if config.structured_generation:
            try:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )
            except ImportError as exc:
                raise RuntimeError("Install the qwen optional dependencies") from exc
            self.schema_parser = JsonSchemaParser
            self.prefix_builder = build_transformers_prefix_allowed_tokens_fn
        options = dict(local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **options)
        kwargs = {"torch_dtype": getattr(torch, config.dtype), "use_safetensors": True}
        if config.quantization == "bnb4":
            try:
                import accelerate  # noqa: F401
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Install qwen-cuda dependencies and CUDA-enabled PyTorch"
                ) from exc
            kwargs.update(
                device_map={"": "cuda:0"},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=getattr(torch, config.dtype),
                ),
            )
        self.model = AutoModelForCausalLM.from_pretrained(path, **options, **kwargs)
        if config.quantization == "none":
            self.model.to(device)
        self.model.eval().requires_grad_(False)
        self.config = config
        self.lock = threading.Lock()

    def generate(self, messages: list[dict], schema: dict) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
        )
        size = tokens.input_ids.shape[1]
        context = getattr(
            self.model.config, "max_position_embeddings", self.config.max_input_tokens
        )
        if size > self.config.max_input_tokens or size + self.config.max_new_tokens > context:
            raise ValueError("Input exceeds configured Qwen context budget; no silent truncation")
        tokens = tokens.to(self.model.device)
        kwargs = {}
        if self.config.structured_generation:
            kwargs["prefix_allowed_tokens_fn"] = self.prefix_builder(
                self.tokenizer, self.schema_parser(schema)
            )
        # Greedy decoding avoids sampling state and does not change E5's RNG state.
        with self.lock, torch.inference_mode():
            result = self.model.generate(
                **tokens,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        return self.tokenizer.decode(result[0, size:], skip_special_tokens=True).strip()
