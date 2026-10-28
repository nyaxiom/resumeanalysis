import hashlib
import math
import random
from pathlib import Path
from typing import Literal

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.schemas import InputModel, Profile, Vacancy
from resumeanalysis_legacy.data.adapters import digest
from resumeanalysis_legacy.data.dataset import SPLITS, PairRecord, assert_no_leakage

OPTIONAL_FIELDS = (
    "experience_years",
    "projects",
    "education",
    "seniority",
    "remote_only",
    "allowed_locations",
)


def populated(value):
    # Zero years is known information; it must not compare equal to boolean False.
    return value is not None and value is not False and value != [] and value != ""


class PartialObservation(InputModel):
    observation_id: str
    source_record_sha256: str
    candidate_id: str
    vacancy_id: str
    profile: Profile
    vacancy: Vacancy
    source_split: Literal["train", "validation", "test"]
    source_provenance: str
    source_label_metadata: Literal[0, 1]
    label: None = None
    augmentation: dict
    retained_skill_fraction: float | None
    retained_field_fraction: float | None
    # Retention ratios count populated structured fields and skill entries, not semantics.
    retained_information_fraction: float | None


def partial_profile(
    record: PairRecord,
    split: str,
    ratio: float,
    seed: int = 42,
    optional_fields: tuple[str, ...] = OPTIONAL_FIELDS,
) -> PartialObservation:
    if split not in SPLITS or not 0 < ratio <= 1:
        raise ValueError("A known source split and retention ratio in (0,1] are required")
    if not set(optional_fields) <= set(OPTIONAL_FIELDS):
        raise ValueError("Unsupported optional field")
    p = record.profile.model_dump()
    original = record.profile.model_dump()
    # Stable per candidate/profile, independent of the matched vacancy and input order.
    identity = digest([record.candidate_id, original, seed])
    rng = random.Random(int(identity, 16))
    indices = list(range(len(p["skills"])))
    rng.shuffle(indices)
    retained = set(indices[: math.floor(len(indices) * ratio)])
    present = [name for name in optional_fields if populated(p[name])]
    fields_order = present.copy()
    rng.shuffle(fields_order)
    retained_fields = set(fields_order[: math.floor(len(fields_order) * ratio)])
    masked = []
    if ratio < 1:
        p["skills"] = [v for i, v in enumerate(p["skills"]) if i in retained]
        # Text can repeat removed skills. Do not re-extract them into the partial view.
        for field in ("resume_text", "projects", "education"):
            if p[field]:
                masked.append(field)
            p[field] = "" if field == "resume_text" else []
        for name in present:
            if name not in retained_fields:
                p[name] = (
                    False if name == "remote_only" else ([] if isinstance(p[name], list) else None)
                )
                masked.append(name)
    final = Profile.model_validate(p)
    skill_fraction = len(final.skills) / len(original["skills"]) if original["skills"] else None
    populated_fields = [k for k, v in original.items() if populated(v) and k != "skills"]
    kept_fields = sum(final.model_dump()[k] == original[k] for k in populated_fields)
    field_fraction = kept_fields / len(populated_fields) if populated_fields else None
    denominator = len(original["skills"]) + len(populated_fields)
    retained_fraction = (len(final.skills) + kept_fields) / denominator if denominator else None
    source_hash = digest(record.model_dump())
    parameters = {
        "type": "structured_skill_and_optional_field_dropout",
        "requested_ratio": ratio,
        "seed": seed,
        "optional_fields": list(optional_fields),
        "masked_fields": sorted(set(masked)),
        "skill_source": "explicit profile.skills only; no text re-extraction",
        "semantics": "Unknown is not confirmed absence; source label is unverified for this view",
    }
    observation_id = digest([source_hash, split, parameters, final.model_dump()])
    return PartialObservation(
        observation_id=observation_id,
        source_record_sha256=source_hash,
        candidate_id=record.candidate_id,
        vacancy_id=record.vacancy_id,
        profile=final,
        vacancy=record.vacancy,
        source_split=split,
        source_provenance=record.provenance,
        source_label_metadata=record.label,
        augmentation=parameters,
        retained_skill_fraction=skill_fraction,
        retained_field_fraction=field_fraction,
        retained_information_fraction=retained_fraction,
    )


def load_splits(path) -> dict[str, list[PairRecord]]:
    path = Path(path)
    if (path / "conversion_report.json").exists():
        from resumeanalysis_legacy.data.conversion import audit_converted

        audit_converted(path)
    splits = {}
    for name in SPLITS:
        rows = []
        for line in (path / (name + ".jsonl")).read_text().splitlines():
            value = loads(line)
            if not isinstance(value, dict) or type(value.get("label")) is not int:
                raise ValueError("Canonical records require explicit integer labels")
            rows.append(PairRecord.model_validate(value, strict=True))
        if not rows:
            raise ValueError("All original partitions must contain records")
        splits[name] = rows
    assert_no_leakage(splits)
    return splits


def augment_dataset(source, output, ratios=(1.0, 0.75, 0.5, 0.25), seed=42) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Augmentations must be outside the official dataset directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty augmentation output directory")
    splits = load_splits(source)  # Validate all partitions before deriving anything.
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation"):
        rows = [partial_profile(r, split, ratio, seed) for r in splits[split] for ratio in ratios]
        (output / (split + ".observations.jsonl")).write_text(
            "".join(r.model_dump_json() + "\n" for r in rows)
        )
        counts[split] = len(rows)
    manifest = {
        "format_version": 1,
        "seed": seed,
        "ratios": list(ratios),
        "counts": counts,
        "labels": "All observations are unlabeled; original labels are metadata only",
        "test": "No augmented official test file created",
        "source_hashes": {
            s: hashlib.sha256((source / (s + ".jsonl")).read_bytes()).hexdigest() for s in SPLITS
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest
