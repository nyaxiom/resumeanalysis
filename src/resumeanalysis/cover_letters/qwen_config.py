from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from resumeanalysis.jsonio import loads
from resumeanalysis.schemas import InputModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


class GenerationConfig(InputModel):
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    model_path: str = "artifacts/qwen"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    quantization: Literal["none", "bnb4"] = "none"
    dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    max_input_tokens: int = Field(default=4096, ge=64)
    max_new_tokens: int = Field(default=768, ge=32)
    seed: int = Field(default=42, ge=0)
    structured_generation: bool = False

    @model_validator(mode="after")
    def compatible(self):
        if self.quantization == "bnb4" and self.device == "cpu":
            raise ValueError("bnb4 requires CUDA; select quantization=none for CPU")
        if self.device == "cpu" and self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("CPU requires explicit float32 or bfloat16")
        return self


def read_config(path=None):
    return (
        GenerationConfig.model_validate(loads(Path(path).read_text()), strict=True)
        if path
        else GenerationConfig()
    )
