"""Canonical pair records and strict entity/content leakage checks."""

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from resumeanalysis.schemas import Profile, Vacancy

SPLITS = ("train", "validation", "test")


class PairRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    vacancy_id: str
    profile: Profile
    vacancy: Vacancy
    label: Literal[0, 1]
    provenance: str
    hard_negative: bool = False


def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(re.findall(r"\w+|[+#]", text.casefold())).encode()).hexdigest()


def entity_keys(record: PairRecord) -> set[str]:
    """Both IDs and normalized text must be disjoint, even with changed metadata."""
    p, v = record.profile, record.vacancy
    ptext = p.resume_text.strip() or json.dumps(sorted(p.skills), ensure_ascii=False)
    vtext = v.description.strip() or json.dumps(sorted(v.required_skills), ensure_ascii=False)
    return {
        "p-id:" + record.candidate_id,
        "v-id:" + record.vacancy_id,
        "p-text:" + fingerprint(ptext),
        "v-text:" + fingerprint(vtext),
    }


def assert_no_leakage(splits: dict[str, list[PairRecord]]) -> None:
    seen: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            for key in entity_keys(record):
                if key in seen and seen[key] != split:
                    raise ValueError(f"Entity leakage between {seen[key]} and {split}: {key}")
                seen[key] = split


def grouped_split(records: list[PairRecord], seed: int = 42) -> dict[str, list[PairRecord]]:
    """Connected components on the candidate/vacancy bipartite graph.

    Fails on inseparable data; never falls back to splitting individual pairs.
    """
    parent = list(range(len(records)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        for key in entity_keys(record):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    if len(groups) < 3:
        raise ValueError(
            "Fewer than three independent entity components; cannot split without leakage"
        )
    items = list(groups.values())
    random.Random(seed).shuffle(items)
    items.sort(key=len, reverse=True)
    result = {s: [] for s in SPLITS}
    targets = dict(zip(SPLITS, (0.7 * len(records), 0.15 * len(records), 0.15 * len(records))))
    for group in items:
        destination = min(SPLITS, key=lambda s: len(result[s]) / max(1, targets[s]))
        result[destination].extend(group)
    if any(not rows for rows in result.values()):
        raise ValueError("Empty split after component allocation")
    assert_no_leakage(result)
    return result


def load_records(path: str | Path) -> list[PairRecord]:
    return [
        PairRecord.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def save_splits(
    splits: dict[str, list[PairRecord]], destination: str | Path, metadata: dict
) -> dict:
    from .download import sha256_file

    assert_no_leakage(splits)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, records in splits.items():
        if not records:
            raise ValueError(f"Empty {split} split")
        path = destination / f"{split}.jsonl"
        path.write_text("".join(r.model_dump_json() + "\n" for r in records))
        summary[split] = {
            "rows": len(records),
            "positives": sum(r.label for r in records),
            "hard_negatives": sum(r.hard_negative for r in records),
            "candidates": len({r.candidate_id for r in records}),
            "vacancies": len({r.vacancy_id for r in records}),
            "sha256": sha256_file(path),
        }
    manifest = {**metadata, "splits": summary, "leakage_check": "passed: IDs and normalized texts"}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
