from typing import Literal

from pydantic import Field, model_validator

from resumeanalysis.schemas import InputModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


class QwenConfig(InputModel):
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    model_path: str = "artifacts/qwen"
    device: Literal["cpu", "cuda", "auto"] = "cuda"
    quantization: Literal["none", "bnb4"] = "bnb4"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    max_input_tokens: int = Field(default=4096, ge=64)
    max_new_tokens: int = Field(default=768, ge=32)
    threshold: float = Field(default=0.5, ge=0, le=1)
    retries: int = Field(default=1, ge=0, le=3)
    seed: int = Field(default=42, ge=0)
    structured_generation: bool = True

    @model_validator(mode="after")
    def compatible(self):
        if self.quantization == "bnb4" and self.device == "cpu":
            raise ValueError("bnb4 requires CUDA; select quantization=none for CPU")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("Use float32 for the supported CPU configuration")
        return self
