"""Inventory and fail-closed partition eligibility from exact and approximate text overlap."""

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.skill_data.guard import NearIndex, load_guard, shingles, text_hash
from resumeanalysis.skill_data.schema import SkillRecord
from resumeanalysis.skill_data.sources import SEALED_SHAS, digest, file_sha, write_jsonl

RANK = {"auxiliary": 0, "train": 1, "validation": 2, "external_evaluation": 3}


def read_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = SkillRecord.model_validate(loads(line))
            if record.provenance.source_sha256 in SEALED_SHAS:
                raise ValueError("Sealed test annotations cannot be loaded")
            yield record


def annotation_hash(record):
    values = [a.model_dump() for a in record.annotations]
    values.sort(key=lambda a: json.dumps(a, sort_keys=True))
    return digest(json.dumps(values, sort_keys=True).encode())


def audit(processed, guard_path, output, *, threshold=0.85):
    processed, output = Path(processed).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Audit requires a fresh output directory")
    if output.is_relative_to(processed) or processed.is_relative_to(output):
        raise ValueError("Audit and processed directories must be separate")
    manifest = loads((processed / "preprocessing.json").read_text())
    if manifest["status"] != "passed":
        raise ValueError("Failed preprocessing cannot be used")
    guard = load_guard(guard_path)
    output.mkdir(parents=True)
    index = NearIndex(threshold)
    for key, values in sorted(guard["fingerprints"].items()):
        index.add("guard:" + key, set(values))
    texts, metadata, file_records = {}, {}, {}
    inventory, all_labels, all_surfaces = [], set(), set()
    languages, types, categories = Counter(), Counter(), Counter()
    groups = defaultdict(list)
    for file in manifest["files"]:
        if "normalized_file" not in file:
            continue
        relative = file["normalized_file"]
        path = processed / relative
        if file_sha(path) != file["normalized_sha256"]:
            raise ValueError("Normalized source changed since preprocessing")
        item = {
            "dataset": file["dataset"],
            "split": file["asset"]["split"],
            "role": file["asset"]["role"],
            "source_rows": file["input_rows"],
            "samples": 0,
            "annotation_count": 0,
            "skill_annotations": 0,
            "span_mentions": 0,
            "surface_mentions": 0,
        }
        counts, labels, surfaces, exact, ids = Counter(), set(), set(), set(), []
        for record in read_records(path):
            key = text_hash(record.text)
            texts.setdefault(key, record.text)
            info = {
                "id": record.id,
                "file": relative,
                "dataset": record.provenance.dataset,
                "split": record.provenance.split,
                "role": record.provenance.role,
                "text_hash": key,
                "annotation_hash": annotation_hash(record),
            }
            if record.id in metadata:
                raise ValueError("Duplicate record identity")
            metadata[record.id] = info
            ids.append(record.id)
            exact.add(key)
            item["samples"] += 1
            item["annotation_count"] += len(record.annotations)
            languages[record.language] += 1
            types[record.annotation_type] += 1
            for a in record.annotations:
                counts[a.category] += 1
                if a.category in {"SKILL", "KNOWLEDGE"}:
                    item["skill_annotations"] += 1
                    item["span_mentions"] += a.start is not None
                    item["surface_mentions"] += a.surface is not None
                    if a.surface:
                        surfaces.add(a.surface.casefold())
                if a.skill_label:
                    labels.add(a.skill_label)
            # Skill-XL supplies stable job UUIDs. SkillSpan indices and Green
            # sentence IDs cannot establish cross-split document identity.
            if record.provenance.dataset == "skill_xl" and record.provenance.group_id:
                groups[record.provenance.group_id].append(record.id)
        item.update(
            unique_texts=len(exact),
            duplicate_samples=item["samples"] - len(exact),
            label_categories=dict(counts),
            unique_skill_labels=len(labels),
            unique_surface_forms=len(surfaces),
        )
        inventory.append(item)
        file_records[relative] = ids
        categories.update(counts)
        all_labels.update(labels)
        all_surfaces.update(surfaces)
    by_hash = defaultdict(list)
    for info in metadata.values():
        by_hash[info["text_hash"]].append(info["id"])
    reasons = defaultdict(set)
    for identifier, info in metadata.items():
        if info["role"] == "research_only":
            reasons[identifier].add("research_only_license_or_task_review")
    overlap = Counter()

    def resolve(ids, reason):
        eligible = [i for i in ids if metadata[i]["role"] in RANK]
        if eligible:
            highest = max(RANK[metadata[i]["role"]] for i in eligible)
            for identifier in eligible:
                if RANK[metadata[identifier]["role"]] < highest:
                    reasons[identifier].add(reason)

    for key, ids in by_hash.items():
        if key in guard["fingerprints"]:
            for identifier in ids:
                reasons[identifier].add("exact_skillspan_test_text")
        resolve(ids, "exact_cross_role_overlap")
        partitions = defaultdict(list)
        for identifier in ids:
            partitions[metadata[identifier]["file"]].append(identifier)
        for duplicates in partitions.values():
            duplicates.sort()
            if len({metadata[i]["annotation_hash"] for i in duplicates}) > 1:
                for identifier in duplicates:
                    reasons[identifier].add("duplicate_text_conflicting_annotations")
            else:
                for identifier in duplicates[1:]:
                    reasons[identifier].add("within_partition_exact_duplicate")
        files = sorted(partitions)
        for a, b in itertools.combinations(files, 2):
            overlap[("exact", a, b)] += 1
    for ids in groups.values():
        resolve(ids, "cross_role_job_id")
    near_count = near_guard = 0
    near_path = output / "near_overlaps.jsonl"
    with near_path.open("x") as stream:
        for key in sorted(texts):
            values = shingles(texts[key])
            for other, score in index.query(values):
                if other.startswith("guard:"):
                    if other[6:] == key:
                        continue
                    near_guard += 1
                    for identifier in by_hash[key]:
                        reasons[identifier].add("near_skillspan_test_text")
                    event = {"left": key, "right": other, "jaccard": score}
                else:
                    other_key = other[5:]
                    near_count += 1
                    a_ids, b_ids = by_hash[key], by_hash[other_key]
                    resolve(a_ids + b_ids, "near_cross_role_overlap")
                    for a in {metadata[i]["file"] for i in a_ids}:
                        for b in {metadata[i]["file"] for i in b_ids}:
                            overlap[("near", *sorted((a, b)))] += 1
                    event = {"left": key, "right": other_key, "jaccard": score}
                stream.write(json.dumps(event) + "\n")
            index.add("data:" + key, values)
    outcomes = [
        {
            "id": identifier,
            "file": metadata[identifier]["file"],
            "eligible": not reasons[identifier],
            "reasons": sorted(reasons[identifier]),
        }
        for identifier in sorted(metadata)
    ]
    write_jsonl(output / "outcomes.jsonl", outcomes)
    partitions = []
    for file in manifest["files"]:
        if "normalized_file" not in file:
            continue
        relative = file["normalized_file"]
        target = output / "partitions" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = processed / relative
        if file_sha(source) != file["normalized_sha256"]:
            raise ValueError("Source changed during auditing")
        write_jsonl(target, (r.model_dump() for r in read_records(source) if not reasons[r.id]))
        count = sum(not reasons[i] for i in file_records[relative])
        partitions.append(
            {
                "path": str(target.relative_to(output)),
                "sha256": file_sha(target),
                "dataset": file["dataset"],
                "split": file["asset"]["split"],
                "role": file["asset"]["role"],
                "samples": count,
            }
        )
    report = {
        "schema_version": 1,
        "status": "passed",
        "catalog_sha256": manifest["catalog_sha256"],
        "preprocessing_sha256": file_sha(processed / "preprocessing.json"),
        "guard_sha256": file_sha(guard_path),
        "inventory": inventory,
        "totals": {
            "samples": len(metadata),
            "unique_texts": len(texts),
            "eligible_samples": sum(o["eligible"] for o in outcomes),
            "languages": dict(languages),
            "annotation_types": dict(types),
            "label_categories": dict(categories),
            "unique_skill_labels": len(all_labels),
            "unique_surface_forms": len(all_surfaces),
        },
        "quarantine_reasons": dict(
            sorted(Counter(r for rs in reasons.values() for r in rs).items())
        ),
        "cross_partition_overlap": [
            {"kind": kind, "left": left, "right": right, "unique_text_pairs": count}
            for (kind, left, right), count in sorted(overlap.items())
        ],
        "near_duplicates": {
            "corpus_unique_text_pairs": near_count,
            "heldout_guard_pairs": near_guard,
            "threshold": threshold,
            "metric": "3-token shingle Jaccard",
            "candidate_search": "32 XOR min-hashes; 8 bands of 4; deterministic approximate retrieval",
            "complete_pairwise_search": False,
            "truncated": False,
            "limitations": "Candidate retrieval can miss near-duplicates; semantic paraphrases are not detected. Same-role near duplicates are flagged, not removed. Document identity unavailable except Skill-XL UUIDs.",
        },
        "heldout_policy": "SkillSpan test labels never ingested; only pinned text fingerprints quarantine overlaps. TECH/HOUSE test annotations remain sealed. External gold is inventoried, never used for fitting or model selection.",
        "partitions": partitions,
        "outcomes_sha256": file_sha(output / "outcomes.jsonl"),
        "near_overlaps_sha256": file_sha(near_path),
    }
    write_json(output / "audit.json", report)
    return report


def partition_records(audit_root, *, roles, datasets=None):
    """The only experiment input path: verified audited partitions and explicit roles."""
    root = Path(audit_root)
    report = loads((root / "audit.json").read_text())
    if report["status"] != "passed":
        raise ValueError("Audit failed")
    for part in report["partitions"]:
        if part["role"] not in roles or (datasets is not None and part["dataset"] not in datasets):
            continue
        path = root / part["path"]
        if not path.resolve().is_relative_to(root.resolve()) or file_sha(path) != part["sha256"]:
            raise ValueError("Audited partition integrity mismatch")
        for record in read_records(path):
            if (
                record.provenance.role != part["role"]
                or record.provenance.dataset != part["dataset"]
            ):
                raise ValueError("Partition source role mismatch")
            yield record
