"""Explicit CPU cross-encoder configuration; public scores remain uncalibrated."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_path: str = "artifacts/reranker"
    model_id: str = MODEL_ID
    revision: str = REVISION
    batch_size: int = Field(default=8, ge=1, le=64, strict=True)
    max_length: int = Field(default=512, ge=32, le=512, strict=True)
    long_document_policy: Literal["error", "truncate", "window"] = "window"
    max_windows: int = Field(default=16, ge=1, le=256, strict=True)
    overlap_tokens: int = Field(default=32, ge=0, le=128, strict=True)

    @model_validator(mode="after")
    def pinned_model(self):
        if self.model_id != MODEL_ID or self.revision != REVISION:
            raise ValueError("Reranker model/revision differs from the reviewed pinned asset")
        if (
            self.long_document_policy == "window"
            and self.overlap_tokens >= (self.max_length - 4) // 2
        ):
            raise ValueError("Window overlap must be smaller than each side's token capacity")
        return self
