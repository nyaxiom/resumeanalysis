"""Historical classification settings layered over shared local generation."""

from typing import Literal

from pydantic import Field

from resumeanalysis.cover_letters.qwen_config import GenerationConfig


class QwenConfig(GenerationConfig):
    device: Literal["cpu", "cuda", "auto"] = "cuda"
    quantization: Literal["none", "bnb4"] = "bnb4"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    threshold: float = Field(default=0.5, ge=0, le=1)
    retries: int = Field(default=1, ge=0, le=3)
    structured_generation: bool = True
