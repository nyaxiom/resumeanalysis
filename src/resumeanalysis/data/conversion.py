"""Conversion orchestration, count reconciliation, provenance and quality artifacts."""

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import Field, model_validator

from resumeanalysis.data.adapters import (
    AdapterSpec,
    convert_row,
    digest,
    fields,
    inspect_rows,
    source_file_policy,
    source_rows,
)
from resumeanalysis.data.dataset import PairRecord, assert_no_leakage
from resumeanalysis.data.download import sha256_file
from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.schemas import InputModel


class QualityPolicy(InputModel):
    max_missing_rates: dict[str, float] = Field(default_factory=dict)
    min_positive_rate: float | None = Field(default=None, ge=0, le=1)
    max_positive_rate: float | None = Field(default=None, ge=0, le=1)
    reference_positive_rate: float | None = Field(default=None, ge=0, le=1)
    max_balance_change: float | None = Field(default=None, ge=0, le=1)
    near_duplicate_similarity: float | None = Field(default=None, gt=0, le=1)
    max_near_duplicate_comparisons: int = Field(default=100000, ge=1)
    rationale: str = "No dataset-specific acceptance thresholds supplied; descriptive checks only."

    @model_validator(mode="after")
    def valid(self):
        if any(not 0 <= v <= 1 for v in self.max_missing_rates.values()):
            raise ValueError("Missing-rate thresholds must be in [0,1]")
        if self.max_balance_change is not None and self.reference_positive_rate is None:
            raise ValueError("Balance-change threshold requires a reference rate")
        if (
            self.min_positive_rate is not None
            and self.max_positive_rate is not None
            and self.min_positive_rate > self.max_positive_rate
        ):
            raise ValueError("Inverted positive-rate range")
        if not self.rationale.strip():
            raise ValueError("Quality policy requires its dataset-specific rationale")
        return self


class ConversionFailed(ValueError):
    pass


def distribution(values):
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "histogram": dict(Counter(map(str, values))),
    }


def quality_report(splits: dict[str, list[PairRecord]], policy: QualityPolicy) -> dict:
    records = [r for rows in splits.values() for r in rows]
    paths = set(
        fields(
            PairRecord(
                candidate_id="", vacancy_id="", profile={}, vacancy={}, label=0, provenance=""
            ).model_dump()
        )
    )
    missing = Counter()
    for record in records:
        for path, values in fields(record.model_dump()).items():
            if any(v is None or v == "" or v == [] for v in values):
                missing[path] += 1
    rates = {p: missing[p] / len(records) if records else None for p in sorted(paths)}
    labels = Counter(r.label for r in records)
    positive = labels[1] / len(records) if records else None
    violations = []
    for path, threshold in policy.max_missing_rates.items():
        if path not in rates:
            violations.append({"code": "unsupported_quality_policy_field", "field": path})
            continue
        if rates[path] is not None and rates[path] > threshold:
            violations.append({"code": "missing_rate", "field": path})
    if positive is not None:
        if policy.min_positive_rate is not None and positive < policy.min_positive_rate:
            violations.append({"code": "positive_rate_below_minimum"})
        if policy.max_positive_rate is not None and positive > policy.max_positive_rate:
            violations.append({"code": "positive_rate_above_maximum"})
        if (
            policy.max_balance_change is not None
            and abs(positive - policy.reference_positive_rate) > policy.max_balance_change
        ):
            violations.append({"code": "class_balance_change"})
    try:
        assert_no_leakage(splits)
        leakage = {"status": "passed", "scope": "candidate/vacancy IDs and normalized content"}
    except ValueError:
        # No candidate names, IDs or text in generic report errors.
        leakage = {"status": "failed", "scope": "candidate/vacancy IDs and normalized content"}
        violations.append({"code": "cross_split_leakage"})
    duplicates = Counter(digest(r.model_dump()) for r in records)
    near = {"status": "not_run", "reason": "No dataset-specific similarity threshold supplied"}
    if policy.near_duplicate_similarity is not None:
        checked, pairs, total = 0, [], len(records) * (len(records) - 1) // 2
        texts = [
            " ".join((r.profile.resume_text + " " + r.vacancy.description).casefold().split())
            for r in records
        ]
        for i in range(len(records)):
            for j in range(i):
                if checked >= policy.max_near_duplicate_comparisons:
                    break
                checked += 1
                if (
                    texts[i]
                    and texts[j]
                    and SequenceMatcher(None, texts[i], texts[j]).ratio()
                    >= policy.near_duplicate_similarity
                ):
                    pairs.append([i, j])
            if checked >= policy.max_near_duplicate_comparisons:
                break
        near = {
            "status": "complete" if checked == total else "partial",
            "method": "SequenceMatcher on normalized resume+vacancy text; not a leakage proof",
            "threshold": policy.near_duplicate_similarity,
            "comparisons": checked,
            "possible_comparisons": total,
            "pair_count": len(pairs),
            "example_record_indices": pairs[:10],
        }
    return {
        "record_count": len(records),
        "splits": {s: len(v) for s, v in splits.items()},
        "missing_rates": rates,
        "label_distribution": dict(labels),
        "positive_rate": positive,
        "skill_counts": {
            "profile": distribution([len(r.profile.skills) for r in records]),
            "required": distribution([len(r.vacancy.required_skills) for r in records]),
        },
        "text_lengths": {
            "resume": distribution([len(r.profile.resume_text) for r in records]),
            "vacancy": distribution([len(r.vacancy.description) for r in records]),
        },
        "exact_duplicate_records": sum(n - 1 for n in duplicates.values()),
        "near_duplicates": near,
        "leakage": leakage,
        "numerical_ranges": "Canonical Pydantic bounds validated for every converted record",
        "policy": policy.model_dump(),
        "violations": violations,
    }


def convert_dataset(
    input_path,
    output,
    spec: AdapterSpec,
    *,
    dataset: str,
    revision: str,
    provenance: str,
    permissive=False,
    retain_source=False,
    policy: QualityPolicy | None = None,
) -> dict:
    source, output = Path(input_path).resolve(), Path(output).resolve()
    if output == source or (source.is_dir() and output.is_relative_to(source)):
        raise ValueError("Conversion output must be separate from the source")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Conversion requires an empty output directory")
    if not all(s.strip() for s in (dataset, revision, provenance)):
        raise ValueError("Dataset name, revision and provenance are required")
    # Revalidate specifications supplied through Python after any caller mutation.
    spec = AdapterSpec.model_validate(spec.model_dump(), strict=True)
    rows = source_rows(source, dataset, revision)
    source_files = {row.file: {"sha256": row.file_sha256, "input_records": 0} for row in rows}
    for row in rows:
        source_files[row.file]["input_records"] += 1
    if source.is_dir():
        for file in sorted(source.glob("*.jsonl")):
            if file.is_file() and file.name not in source_files:
                source_files[file.name] = {"sha256": sha256_file(file), "input_records": 0}
    coverage = inspect_rows(rows, spec)  # Recursive schema analysis before conversion.
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "sources"
    if retain_source:
        snapshots.mkdir()
    outcomes, splits = [], {}
    source_labels = Counter()
    before = {
        "profile_skill_counts": [],
        "vacancy_required_skill_counts": [],
        "resume_text_lengths": [],
        "vacancy_text_lengths": [],
    }
    for row in rows:
        if row.error is None:
            from resumeanalysis.data.adapters import get_path

            for rule in spec.rules:
                key, expected = {
                    "/profile/skills": ("profile_skill_counts", list),
                    "/vacancy/required_skills": ("vacancy_required_skill_counts", list),
                    "/profile/resume_text": ("resume_text_lengths", str),
                    "/vacancy/description": ("vacancy_text_lengths", str),
                }.get(rule.target, (None, None))
                if key:
                    try:
                        value = get_path(row.value, rule.source)
                    except (KeyError, TypeError):
                        continue
                    if type(value) is expected:
                        before[key].append(len(value))
        record, status, issues = convert_row(row, spec, permissive)
        entry = {
            "source_id": row.source_id,
            "source_file": row.file,
            "source_line": row.line,
            "source_sha256": row.sha256,
            "source_split": row.split,
            "outcome": status,
            "issues": issues,
            "normalized_index": None,
        }
        if retain_source:
            (snapshots / (row.source_id + ".jsonl")).write_bytes(row.raw)
            entry["snapshot"] = "sources/" + row.source_id + ".jsonl"
        if record is not None:
            split = row.split or "records"
            entry["normalized_file"] = split + ".jsonl"
            entry["normalized_index"] = len(splits.setdefault(split, []))
            entry["normalized_sha256"] = digest(record.model_dump())
            splits[split].append(record)
        outcomes.append(entry)
        label_rule = next((r for r in spec.rules if r.target == "/label"), None)
        if row.error is None and label_rule:
            from resumeanalysis.data.adapters import get_path

            try:
                label = get_path(row.value, label_rule.source)
                # Labels are the only source values summarized. No arbitrary objects/text.
                declared = any(
                    type(m.source) is type(label) and m.source == label for m in spec.labels
                )
                if (type(label) is int and label in (0, 1)) or declared:
                    source_labels[json.dumps(label)] += 1
                else:
                    source_labels[type(label).__name__ + ":sha256:" + digest(label)] += 1
            except (KeyError, TypeError):
                source_labels["<missing>"] += 1
    counts = Counter(r["outcome"] for r in outcomes)
    quality = quality_report(splits, policy or QualityPolicy())
    for name, metadata in source_files.items():
        file = source / name if source.is_dir() else source
        if not file.is_file() or sha256_file(file) != metadata["sha256"]:
            quality["violations"].append(
                {"code": "source_changed_during_conversion", "source_file": name}
            )
        if metadata["input_records"] == 0:
            quality["violations"].append({"code": "empty_source_file", "source_file": name})
    # Link near-duplicate examples to opaque source IDs in canonical flattening order.
    ordered_ids = [
        o["source_id"] for s in splits for o in outcomes if o.get("normalized_file") == s + ".jsonl"
    ]
    near = quality["near_duplicates"]
    if "example_record_indices" in near:
        near["example_source_ids"] = [
            [ordered_ids[i], ordered_ids[j]] for i, j in near.pop("example_record_indices")
        ]
    raw_hashes = Counter(digest(r.value) for r in rows if r.error is None)
    quality["source_exact_duplicate_records"] = sum(n - 1 for n in raw_hashes.values())
    quality["source_schema_statistics"] = coverage
    quality["source_label_distribution"] = dict(source_labels)
    quality["before_conversion"] = {k: distribution(v) for k, v in before.items()}
    quality["before_conversion"]["scope"] = (
        "Mapped source fields with expected container types, before defaults/coercion; invalid types remain in schema and row diagnostics"
    )
    failed = bool(quality["violations"] or (counts["rejected"] and not permissive))
    report = {
        "format_version": 1,
        "dataset": dataset,
        "revision": revision,
        "provenance": provenance,
        "source_files": source_files,
        "source_file_policy": source_file_policy(source),
        "adapter": spec.model_dump(),
        "mode": "permissive" if permissive else "strict",
        "status": "failed" if failed else "passed",
        "retention": {
            "raw_retained": retain_source,
            "full_reversibility": retain_source,
            "limitation": None
            if retain_source
            else "Only hashes/diagnostics retained; originals must remain with the source custodian.",
        },
        **coverage,
        "outcome_counts": {
            s: counts[s] for s in ["converted", "converted_with_loss", "excluded", "rejected"]
        },
        "count_reconciliation": len(rows) == sum(counts.values()),
        "normalized_records": quality["record_count"],
        "normalized_written": 0 if failed else quality["record_count"],
        "record_outcomes": outcomes,
    }
    write_json(output / "conversion_report.json", report)
    write_json(output / "quality_report.json", quality)
    summary = (
        f"# Dataset conversion\n\nDataset: {dataset}; revision: {revision}.\n\n"
        f"Status: {report['status']}; mode: {report['mode']}.\n\n"
        f"Input: {len(rows)}; converted: {counts['converted']}; converted with loss: "
        f"{counts['converted_with_loss']}; excluded: {counts['excluded']}; rejected: {counts['rejected']}.\n\n"
        f"Normalized records written: {report['normalized_written']}. Counts reconcile: {report['count_reconciliation']}.\n\n"
        f"Field disposition coverage: {coverage['field_coverage_fraction']:.1%}. "
        "Coverage is not a claim that every field becomes an ML feature.\n\n"
        f"Full source reversibility available: {retain_source}. "
        "Source values are omitted from diagnostic reports; optional local snapshots may contain personal data.\n\n"
        "See conversion_report.json for every row outcome, field disposition, rename, conversion, "
        "omission, rejection and representative source ID. See quality_report.json for missingness, "
        "distributions, duplicate checks, leakage and configured acceptance criteria.\n"
    )
    summary += (
        "\n| Source field | Disposition | Present | Missing | Null |\n|---|---|---:|---:|---:|\n"
    )
    for field, entry in sorted(coverage["schema_coverage"].items()):
        summary += f"| {field.replace('|', '&#124;')} | {entry['disposition']} | {entry['records_present']} | {entry['records_missing']} | {entry['null_values']} |\n"
    issues_by_code = {}
    for outcome in outcomes:
        for issue in outcome["issues"]:
            entry = issues_by_code.setdefault(issue["code"], {"count": 0, "ids": []})
            entry["count"] += 1
            if len(entry["ids"]) < 3:
                entry["ids"].append(outcome["source_id"])
    summary += "\nProblem/action counts and representative opaque source identifiers:\n\n"
    summary += (
        "\n".join(
            f"- {code}: {entry['count']}; examples: {', '.join(entry['ids'])}"
            for code, entry in sorted(issues_by_code.items())
        )
        + "\n"
    )
    (output / "conversion_summary.md").write_text(summary)
    if not failed:
        for split, records in splits.items():
            (output / (split + ".jsonl")).write_text(
                "".join(r.model_dump_json() + "\n" for r in records)
            )
    files = {
        str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    manifest = {
        "format_version": 1,
        "status": report["status"],
        "dataset": dataset,
        "revision": revision,
        "provenance": provenance,
        "sha256": files,
        "splits": {s: {"rows": len(v)} for s, v in splits.items()} if not failed else {},
        "partitioning": "Preserved source split filenames; no automatic partitioning",
    }
    write_json(output / "manifest.json", manifest)
    if failed:
        raise ConversionFailed(
            "Conversion failed; diagnostic reports saved, normalized data not published"
        )
    return report


def audit_converted(path) -> dict:
    path = Path(path)
    manifest = loads((path / "manifest.json").read_text())
    if manifest["status"] != "passed":
        raise ConversionFailed("Conversion did not pass")
    for name, checksum in manifest["sha256"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path.resolve()) or sha256_file(file) != checksum:
            raise ValueError("Converted artifact checksum mismatch or invalid path")
    report = loads((path / "conversion_report.json").read_text())
    outcomes = report["record_outcomes"]
    if (
        len(outcomes) != report["total_input_records"]
        or len({o["source_id"] for o in outcomes}) != len(outcomes)
        or sum(report["outcome_counts"].values()) != len(outcomes)
    ):
        raise ValueError("Conversion count reconciliation failed")
    splits = {}
    for split, metadata in manifest["splits"].items():
        records = []
        for line in (path / (split + ".jsonl")).read_text().splitlines():
            record = PairRecord.model_validate(loads(line), strict=True)
            records.append(record)
        if len(records) != metadata["rows"]:
            raise ValueError("Normalized record count mismatch")
        lineage = [o for o in outcomes if o.get("normalized_file") == split + ".jsonl"]
        if len(lineage) != len(records):
            raise ValueError("Missing conversion lineage")
        for index, record in enumerate(records):
            match = [o for o in lineage if o["normalized_index"] == index]
            if len(match) != 1 or match[0]["normalized_sha256"] != digest(record.model_dump()):
                raise ValueError("Normalized record lineage mismatch")
        splits[split] = records
    if sum(map(len, splits.values())) != report["normalized_written"]:
        raise ValueError("Written record count mismatch")
    result = quality_report(
        splits,
        QualityPolicy.model_validate(loads((path / "quality_report.json").read_text())["policy"]),
    )
    result["artifact_integrity"] = "passed"
    return result
