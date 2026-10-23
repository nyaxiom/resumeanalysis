"""Loss-aware skill annotations; classification labels never acquire invented spans."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Annotation(StrictModel):
    category: str
    surface: str | None = None
    start: int | None = None
    end: int | None = None
    skill_label: str | None = None
    skill_id: str | None = None
    attributes: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def paired_offsets(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Offsets must be both present or both absent")
        return self


class Provenance(StrictModel):
    dataset: str
    repository: str
    revision: str
    source_file: str
    source_sha256: str
    source_rows: list[int]
    split: str
    role: Literal["train", "validation", "external_evaluation", "auxiliary", "research_only"]
    license: str
    license_url: str | None = None
    license_evidence_url: str | None = None
    attribution: str
    group_id: str | None = None


class SkillRecord(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    text: str
    text_origin: Literal["source", "space_joined_tokens"]
    language: str
    annotation_type: Literal[
        "span", "surface_normalization", "sentence_multilabel", "title_multilabel"
    ]
    annotations: list[Annotation]
    tokens: list[str] | None = None
    provenance: Provenance
    # Original input objects/CSV fields, including BIO/POS, remain recoverable.
    source_records: list[dict]

    @model_validator(mode="after")
    def literal_evidence(self):
        if not self.text.strip():
            raise ValueError("Empty source text")
        if self.text_origin == "space_joined_tokens":
            if not self.tokens or " ".join(self.tokens) != self.text:
                raise ValueError("Token text must retain the declared one-space reconstruction")
        for annotation in self.annotations:
            if annotation.start is not None:
                if self.annotation_type != "span":
                    raise ValueError("Non-span data cannot acquire offsets")
                if not 0 <= annotation.start < annotation.end <= len(self.text):
                    raise ValueError("Out-of-range annotation")
                if self.text[annotation.start : annotation.end] != annotation.surface:
                    raise ValueError("Span surface differs from source text")
            elif self.annotation_type == "span":
                raise ValueError("Span data requires offsets")
        return self
