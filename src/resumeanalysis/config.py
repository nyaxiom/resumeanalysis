"""Environment configuration; reading it never loads model weights."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    database_url: SecretStr = SecretStr("postgresql+asyncpg://localhost/resumeanalysis")
    embedding_cache_dir: Path | None = Path("artifacts/service-embeddings")
    e5_model: str = "intfloat/multilingual-e5-small"
    e5_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    e5_local_path: Path | None = None
    e5_device: Literal["cpu", "cuda", "auto"] = "cpu"
    e5_batch_size: int = Field(default=16, ge=1, le=256)
    e5_max_length: int = Field(default=512, ge=16, le=512)
    e5_overlap: int = Field(default=32, ge=0)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    l2: float = Field(default=0.001, ge=0, le=1)
    qwen_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    qwen_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    qwen_local_path: Path | None = None
    qwen_device: Literal["cpu", "cuda", "auto"] = "cpu"
    qwen_dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    qwen_max_input_tokens: int = Field(default=4096, ge=64, le=32768)
    qwen_max_new_tokens: int = Field(default=512, ge=32, le=2048)
    letter_max_words: int = Field(default=180, ge=40, le=350)

    @field_validator("database_url")
    @classmethod
    def postgres_url(cls, value):
        try:
            url = make_url(value.get_secret_value())
            if url.drivername != "postgresql+asyncpg" or not url.database:
                raise ValueError
        except (ArgumentError, ValueError):
            raise ValueError(
                "DATABASE_URL must use postgresql+asyncpg and name a database"
            ) from None
        return value

    @model_validator(mode="after")
    def validate_models(self):
        if self.e5_overlap >= self.e5_max_length - 8:
            raise ValueError("E5 overlap must leave room for content and prefix tokens")
        for model in ("e5", "qwen"):
            revision = getattr(self, f"{model}_revision")
            if not getattr(self, f"{model}_local_path") and (
                len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)
            ):
                raise ValueError(f"{model.upper()} revision must be an immutable commit SHA")
        if self.qwen_device == "cpu" and self.qwen_dtype == "float16":
            raise ValueError("Use float32 or bfloat16 for Qwen on CPU")
        return self

    @classmethod
    def from_env(cls):
        values = {}
        for name in cls.model_fields:
            value = os.getenv(f"RESUMEANALYSIS_{name.upper()}")
            if value is not None:
                values[name] = None if value == "" and name.endswith(("_path", "_dir")) else value
        return cls.model_validate(values)
