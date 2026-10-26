"""Synthetic split-overlap fixtures; no external downloads or skill-quality claims."""

import hashlib
import json

import pytest

from resumeanalysis.experiments.skillspan_splits import (
    SkillSpanSplitAuditError,
    audit_skillspan_splits,
)


def record(tokens, *, extra=None):
    return {
        "tokens": tokens,
        "tags_skill": ["O"] * len(tokens),
        "tags_knowledge": ["B", *(["I"] * (len(tokens) - 1))],
        **(extra or {}),
    }


def write(path, rows):
    path.write_bytes(b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in rows))
    return path


def sources(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    return {name: directory / (name + ".jsonl") for name in ("train", "validation", "test")}


def audit(paths, output):
    return audit_skillspan_splits(paths["train"], paths["validation"], paths["test"], output)


def test_duplicates_and_cross_split_counts_preserve_raw_filtered_rows(tmp_path):
    paths = sources(tmp_path)
    write(paths["train"], [record(["Alpha"]), record(["Beta"]), record(["alpha"])])
    write(paths["validation"], [record(["BETA"]), record(["Gamma"])])
    keep = record(
        ["Delta"], extra={"source": "authored fixture", "unmapped": {"preserved": [1, 2]}}
    )
    test_rows = [record(["ALPHA"]), record(["gamma"]), keep, keep]
    write(paths["test"], test_rows)
    originals = {name: p.read_bytes() for name, p in paths.items()}
    report = audit(paths, tmp_path / "audit")
    assert report["status"] == "passed" and report["test_count_reconciliation"]
    assert report["test_outcome_counts"] == {
        "retained": 2,
        "excluded": 2,
        "rejected": 0,
        "blocked": 0,
    }
    assert report["source_splits"]["train"]["unique_normalized_sentences"] == 2
    assert report["source_splits"]["train"]["internal_duplicate_rows"] == 1
    assert report["source_splits"]["test"]["internal_duplicate_rows"] == 1
    for pair in ("train/validation", "train/test", "validation/test"):
        assert report["cross_split_overlap"][pair]["shared_unique_normalized_sentences"] == 1
    assert report["cross_split_overlap"]["train/test"]["overlapping_rows"] == {
        "train": 2,
        "test": 1,
    }
    expected = b"".join(originals["test"].splitlines(keepends=True)[2:])
    assert (tmp_path / "audit/filtered-test.jsonl").read_bytes() == expected
    assert {name: p.read_bytes() for name, p in paths.items()} == originals
    assert report["filtered_test"]["rows"] == 2
    metadata = (tmp_path / "audit/split_audit_report.json").read_text()
    assert '"Alpha"' not in metadata and '"Delta"' not in metadata
    assert [o["source_line"] for o in report["test_outcomes"]] == [1, 2, 3, 4]
    repeated = audit(paths, tmp_path / "repeat")
    assert repeated == report


def test_joined_token_casefold_rule_does_not_strip_punctuation(tmp_path):
    paths = sources(tmp_path)
    write(paths["train"], [record(["Straße", "skill"])])
    write(paths["validation"], [record(["Other"])])
    write(paths["test"], [record(["STRASSE", "SKILL"]), record(["STRASSE", "SKILL", "."])])
    report = audit(paths, tmp_path / "audit")
    assert report["test_outcome_counts"]["excluded"] == 1
    assert report["test_outcome_counts"]["retained"] == 1
    assert report["test_outcomes"][0]["reasons"] == ["normalized_sentence_present_in_train"]


def test_overlap_with_both_reference_splits_records_both_reasons(tmp_path):
    paths = sources(tmp_path)
    for path in paths.values():
        write(path, [record(["Repeated", "sentence"])])
    report = audit(paths, tmp_path / "audit")
    assert report["test_outcomes"][0]["reasons"] == [
        "normalized_sentence_present_in_train",
        "normalized_sentence_present_in_validation",
    ]
    assert report["filtered_test"]["rows"] == 0
    assert (tmp_path / "audit/filtered-test.jsonl").read_bytes() == b""


@pytest.mark.parametrize(
    "bad",
    [
        b"\n",
        b"not json\n",
        b'{"tokens":[]}\n',
        b'{"tokens":["a"],"tokens":["b"]}\n',
        b'{"tokens":["a"],"tags_skill":["invalid"],"tags_knowledge":["O"]}\n',
        b'{"tokens":["a"],"tags_skill":[],"tags_knowledge":["O"]}\n',
        b'{"tokens":[123],"tags_skill":["O"],"tags_knowledge":["O"]}\n',
        b'{"tokens":["a"],"tags_skill":["O"],"tags_knowledge":["O"],"extra":NaN}\n',
    ],
)
def test_invalid_rows_are_reported_not_skipped_and_no_filtered_file_is_published(tmp_path, bad):
    paths = sources(tmp_path)
    write(paths["train"], [record(["Train"])])
    write(paths["validation"], [record(["Validation"])])
    write(paths["test"], [record(["Valid", "test"])])
    paths["test"].write_bytes(paths["test"].read_bytes() + bad)
    with pytest.raises(SkillSpanSplitAuditError, match="diagnostic reports saved"):
        audit(paths, tmp_path / "audit")
    report = json.loads((tmp_path / "audit/split_audit_report.json").read_text())
    assert report["status"] == "failed" and report["test_count_reconciliation"]
    assert report["source_splits"]["test"]["input_records"] == 2
    assert report["test_outcome_counts"]["rejected"] == 1
    assert report["test_outcome_counts"]["blocked"] == 1
    assert not (tmp_path / "audit/filtered-test.jsonl").exists()


def test_invalid_reference_data_blocks_filtering_and_reports_all_test_rows(tmp_path):
    paths = sources(tmp_path)
    paths["train"].write_bytes(b"invalid\n\n")
    write(paths["validation"], [record(["Validation"])])
    write(paths["test"], [record(["First"]), record(["Second"])])
    with pytest.raises(SkillSpanSplitAuditError):
        audit(paths, tmp_path / "audit")
    report = json.loads((tmp_path / "audit/split_audit_report.json").read_text())
    assert report["source_splits"]["train"]["rejected_records"] == 2
    assert report["test_outcome_counts"]["blocked"] == 2


def test_missing_and_empty_files_produce_diagnostics_when_other_inputs_are_readable(tmp_path):
    paths = sources(tmp_path)
    paths["validation"].write_bytes(b"")
    write(paths["test"], [record(["Test"])])
    with pytest.raises(SkillSpanSplitAuditError):
        audit(paths, tmp_path / "audit")
    report = json.loads((tmp_path / "audit/split_audit_report.json").read_text())
    assert not report["all_inputs_countable"]
    assert report["source_splits"]["train"]["file_errors"] == [{"code": "unreadable_source_file"}]
    assert report["source_splits"]["validation"]["file_errors"] == [{"code": "empty_source_file"}]
    assert report["test_outcome_counts"]["blocked"] == 1


def test_output_is_fresh_and_source_paths_cannot_be_replaced(tmp_path):
    paths = sources(tmp_path)
    for name, path in paths.items():
        write(path, [record([name])])
    target = tmp_path / "existing"
    target.mkdir()
    (target / "preserve.txt").write_text("original")
    with pytest.raises(FileExistsError):
        audit(paths, target)
    with pytest.raises(ValueError, match="separate"):
        audit(paths, paths["train"].parent)
    assert (target / "preserve.txt").read_text() == "original"


def test_crlf_unicode_original_labels_and_manifest_checksums_are_preserved(tmp_path):
    paths = sources(tmp_path)
    write(paths["train"], [record(["Train"])])
    write(paths["validation"], [record(["Validation"])])
    row = record(["Проверка", "Python"], extra={"id": "keep-all-fields", "score": 0.23})
    raw = json.dumps(row, ensure_ascii=False).encode() + b"\r\n"
    paths["test"].write_bytes(raw)
    output = tmp_path / "audit"
    report = audit(paths, output)
    assert (output / "filtered-test.jsonl").read_bytes() == raw
    assert json.loads((output / "filtered-test.jsonl").read_text()) == row
    assert report["filtered_test"]["sha256"] == hashlib.sha256(raw).hexdigest()
    manifest = json.loads((output / "manifest.json").read_text())
    for filename, checksum in manifest["sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == checksum
