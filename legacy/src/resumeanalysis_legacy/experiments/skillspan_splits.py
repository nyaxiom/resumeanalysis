"""Exact sentence-overlap diagnostics and a separate SkillSpan test subset.

This never edits input files, labels, extractors or phrase lexicons. Removing
exact normalized overlaps does not establish semantic/job-cluster independence.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json

NORMALIZATION = "Join the original tokens with one ASCII space, then Unicode casefold; no punctuation removal, stemming or fuzzy matching."


class SkillSpanSplitAuditError(ValueError):
    """Diagnostic metadata was retained, but no filtered test was published."""


@dataclass
class _Row:
    source_id: str
    line: int
    raw: bytes
    raw_sha256: str
    sentence_sha256: str | None
    errors: list[dict]


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate(value):
    errors = []
    if not isinstance(value, dict):
        return [{"code": "not_json_object"}]
    tokens = value.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not str or not t for t in tokens)
    ):
        errors.append({"code": "invalid_tokens", "field": "tokens"})
    for field, name in (("tags_skill", "skill"), ("tags_knowledge", "knowledge")):
        tags = value.get(field)
        allowed = {
            "O",
            "B",
            "I",
            "B-" + name,
            "I-" + name,
            "B-" + name.upper(),
            "I-" + name.upper(),
        }
        if not isinstance(tags, list) or any(type(t) is not str or t not in allowed for t in tags):
            errors.append({"code": "invalid_bio_tags", "field": field})
        elif isinstance(tokens, list) and len(tags) != len(tokens):
            errors.append({"code": "token_tag_length_mismatch", "field": field})
    return errors


def _read_split(path: Path, name: str):
    metadata = {
        "source_file": path.name,
        "source_sha256": None,
        "input_records": None,
        "valid_records": 0,
        "rejected_records": 0,
        "file_errors": [],
        "record_outcomes": [],
    }
    try:
        content = path.read_bytes()
    except OSError:
        metadata["file_errors"].append({"code": "unreadable_source_file"})
        return metadata, []
    checksum = metadata["source_sha256"] = _hash(content)
    lines = content.splitlines(keepends=True)
    metadata["input_records"] = len(lines)
    if not lines:
        metadata["file_errors"].append({"code": "empty_source_file"})
    rows = []
    for index, raw in enumerate(lines, 1):
        raw_hash = _hash(raw)
        source_id = _hash(f"{name}:{checksum}:{index}:{raw_hash}".encode())
        errors, sentence_hash = [], None
        try:
            value = loads(raw.decode("utf-8"))
            errors = _validate(value)
            if not errors:
                sentence_hash = _hash(" ".join(value["tokens"]).casefold().encode("utf-8"))
        except (ValueError, UnicodeError):
            errors = [{"code": "invalid_json_or_encoding"}]
        row = _Row(source_id, index, raw, raw_hash, sentence_hash, errors)
        rows.append(row)
        metadata["record_outcomes"].append(
            {
                "source_id": source_id,
                "source_line": index,
                "source_row_sha256": raw_hash,
                "normalized_sentence_sha256": sentence_hash,
                "outcome": "rejected" if errors else "validated",
                "errors": errors,
            }
        )
        metadata["rejected_records" if errors else "valid_records"] += 1
    counts = Counter(r.sentence_sha256 for r in rows if not r.errors)
    metadata.update(
        {
            "unique_normalized_sentences": len(counts),
            "internal_duplicate_rows": sum(n - 1 for n in counts.values()),
            "internal_duplicate_groups": sum(n > 1 for n in counts.values()),
            "duplicate_groups": [
                {
                    "normalized_sentence_sha256": sentence_hash,
                    "rows": count,
                    "source_ids": [r.source_id for r in rows if r.sentence_sha256 == sentence_hash],
                }
                for sentence_hash, count in sorted(counts.items())
                if count > 1
            ],
            "count_reconciliation": metadata["valid_records"] + metadata["rejected_records"]
            == len(lines),
        }
    )
    return metadata, rows


def audit_skillspan_splits(train, validation, test, output_dir) -> dict:
    """Inspect every JSONL row and publish a separate exact-overlap-filtered test.

    Only normalized token text controls filtering. Original row bytes, all source
    fields, BIO annotations and order are preserved in the separate output file.
    Invalid input fails the whole audit after reports are written when possible.
    """
    paths = {
        name: Path(path).resolve()
        for name, path in (("train", train), ("validation", validation), ("test", test))
    }
    output = Path(output_dir).resolve()
    if any(output == path or output in path.parents for path in paths.values()):
        raise ValueError("Audit output must be separate from the input files")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a fresh empty split-audit output directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata, splits = {}, {}
    for name, path in paths.items():
        metadata[name], splits[name] = _read_split(path, name)
    counters = {
        name: Counter(r.sentence_sha256 for r in rows if not r.errors)
        for name, rows in splits.items()
    }
    overlaps = {}
    for left, right in combinations(paths, 2):
        shared = set(counters[left]) & set(counters[right])
        overlaps[f"{left}/{right}"] = {
            "shared_unique_normalized_sentences": len(shared),
            "overlapping_rows": {
                name: sum(counters[name][h] for h in shared) for name in (left, right)
            },
            "matching_row_pairs": sum(counters[left][h] * counters[right][h] for h in shared),
            "representative_source_ids": [
                {
                    "normalized_sentence_sha256": h,
                    left: [r.source_id for r in splits[left] if r.sentence_sha256 == h][:3],
                    right: [r.source_id for r in splits[right] if r.sentence_sha256 == h][:3],
                }
                for h in sorted(shared)[:10]
            ],
        }
    for name, path in paths.items():
        checksum = metadata[name]["source_sha256"]
        if checksum is not None:
            try:
                changed = _hash(path.read_bytes()) != checksum
            except OSError:
                changed = True
            if changed:
                metadata[name]["file_errors"].append({"code": "source_changed_during_audit"})
    failed = any(m["file_errors"] or m["rejected_records"] for m in metadata.values())
    outcomes, retained = [], []
    for row in splits["test"]:
        reasons = []
        if row.errors:
            status = "rejected"
            reasons = [error["code"] for error in row.errors]
        elif failed:
            status, reasons = "blocked", ["input_validation_failed_no_filtered_data_published"]
        else:
            for name in ("train", "validation"):
                if row.sentence_sha256 in counters[name]:
                    reasons.append("normalized_sentence_present_in_" + name)
            status = "excluded" if reasons else "retained"
            if status == "retained":
                retained.append(row)
        outcomes.append(
            {
                "source_id": row.source_id,
                "source_line": row.line,
                "source_row_sha256": row.raw_sha256,
                "normalized_sentence_sha256": row.sentence_sha256,
                "outcome": status,
                "reasons": reasons,
            }
        )
    counts = Counter(row["outcome"] for row in outcomes)
    filtered_path = output / "filtered-test.jsonl"
    if not failed:
        filtered_path.write_bytes(b"".join(row.raw for row in retained))
    report = {
        "format_version": 1,
        "status": "failed" if failed else "passed",
        "normalization": NORMALIZATION,
        "source_splits": metadata,
        "cross_split_overlap": overlaps,
        "test_outcomes": outcomes,
        "test_outcome_counts": {
            name: counts[name] for name in ("retained", "excluded", "rejected", "blocked")
        },
        "test_count_reconciliation": len(outcomes) == (metadata["test"]["input_records"] or 0),
        "all_inputs_countable": all(m["input_records"] is not None for m in metadata.values()),
        "filtered_test": {
            "published": not failed,
            "file": filtered_path.name if not failed else None,
            "rows": len(retained),
            "sha256": _hash(filtered_path.read_bytes()) if not failed else None,
            "preserved": "Original test row bytes, all fields, BIO labels, duplicate rows and order",
        },
        "scope": "Exact casefolded joined-token sentence overlap only. Filtering removes test sentences present in train or validation; internal test duplicates are retained. This does not establish semantic, source-job, candidate, document or job-cluster independence. The original full benchmark remains unchanged and must still be reported separately.",
        "privacy": "Reports contain opaque row/sentence hashes and field error codes, never source sentence text. The separate filtered dataset still contains original data and must remain outside Git.",
    }
    write_json(output / "split_audit_report.json", report)
    summary = "# SkillSpan split-overlap diagnostic\n\n"
    summary += f"Status: {report['status']}. Normalization: {NORMALIZATION}\n\n"
    summary += "| Split | Input rows | Valid rows | Unique sentences | Internal duplicate rows | Rejected rows |\n|---|---:|---:|---:|---:|---:|\n"
    for name, m in metadata.items():
        summary += f"| {name} | {m['input_records']} | {m['valid_records']} | {m.get('unique_normalized_sentences', 0)} | {m.get('internal_duplicate_rows', 0)} | {m['rejected_records']} |\n"
    summary += "\n| Pair | Shared unique sentences | Left overlapping rows | Right overlapping rows |\n|---|---:|---:|---:|\n"
    for pair, info in overlaps.items():
        left, right = pair.split("/")
        summary += f"| {pair} | {info['shared_unique_normalized_sentences']} | {info['overlapping_rows'][left]} | {info['overlapping_rows'][right]} |\n"
    summary += f"\nFiltered test outcomes: retained {counts['retained']}; excluded {counts['excluded']}; rejected {counts['rejected']}; blocked {counts['blocked']}. "
    summary += f"Counts reconcile: {report['test_count_reconciliation']}.\n\n{report['scope']}\n\n"
    summary += "No extractor, training lexicon, label or original split was modified. See split_audit_report.json for every test-row outcome, source-file checksums and opaque overlap identifiers.\n"
    (output / "split_audit_summary.md").write_text(summary)
    files = {
        path.name: _hash(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()
    }
    write_json(
        output / "manifest.json",
        {
            "format_version": 1,
            "status": report["status"],
            "source_files": {name: m["source_sha256"] for name, m in metadata.items()},
            "sha256": files,
        },
    )
    if failed:
        raise SkillSpanSplitAuditError(
            "SkillSpan split audit failed; diagnostic reports saved, filtered data not published"
        )
    return report
