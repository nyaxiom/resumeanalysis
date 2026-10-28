import json

import pytest
from pydantic import ValidationError
from resumeanalysis_legacy.data.adapters import (
    AdapterSpec,
    Exclusion,
    Rule,
    canonical_spec,
    inspect_rows,
    source_rows,
)
from resumeanalysis_legacy.data.conversion import (
    ConversionFailed,
    QualityPolicy,
    audit_converted,
    convert_dataset,
)


def pair(index=0, split="train"):
    return {
        "candidate_id": f"{split}-p{index}",
        "vacancy_id": f"{split}-v{index}",
        "profile": {"resume_text": f"{split} candidate {index}", "skills": ["Python"]},
        "vacancy": {"description": f"{split} vacancy {index}", "required_skills": ["Python"]},
        "label": index % 2,
        "provenance": "synthetic unit fixture",
    }


def source(tmp_path, records):
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def convert(path, out, spec=None, **kwargs):
    return convert_dataset(
        path,
        out,
        spec or canonical_spec(),
        dataset="fixture",
        revision="immutable-test-revision",
        provenance="authored test",
        **kwargs,
    )


@pytest.mark.parametrize(
    "mutate,problem",
    [
        (lambda r: r.update(secret="must be noticed"), "/secret"),
        (
            lambda r: r["profile"].update(contact={"email": "private@example.invalid"}),
            "/profile/contact/email",
        ),
        (
            lambda r: r["profile"].update(history=[{"employer": {"private": "value"}}]),
            "/profile/history/*/employer/private",
        ),
    ],
)
def test_strict_rejects_unknown_and_nested_information_loss(tmp_path, mutate, problem):
    record = pair()
    mutate(record)
    path = source(tmp_path, [record])
    out = tmp_path / "out"
    with pytest.raises(ConversionFailed):
        convert(path, out)
    report = json.loads((out / "conversion_report.json").read_text())
    assert report["schema_coverage"][problem]["disposition"] == "unsupported"
    assert report["outcome_counts"]["rejected"] == 1 and report["count_reconciliation"]
    assert not (out / "records.jsonl").exists()
    assert "private@example" not in (out / "conversion_report.json").read_text()


def test_permissive_preserves_original_bytes_and_records_omissions(tmp_path):
    record = pair()
    record["profile"]["contact"] = {"email": "private@example.invalid"}
    path = source(tmp_path, [record])
    report = convert(path, tmp_path / "out", permissive=True, retain_source=True)
    row = report["record_outcomes"][0]
    assert row["outcome"] == "converted_with_loss"
    assert (tmp_path / "out" / row["snapshot"]).read_bytes() == path.read_bytes()
    assert report["revision"] == "immutable-test-revision"
    assert report["retention"]["full_reversibility"]
    result = audit_converted(tmp_path / "out")
    assert result["artifact_integrity"] == "passed" and result["record_count"] == 1
    normalized = json.loads((tmp_path / "out/records.jsonl").read_text())
    assert "contact" not in normalized["profile"]


@pytest.mark.parametrize("value", ["positive", None, True, 2, -1, 0.5])
def test_unknown_ambiguous_and_invalid_labels_never_invented(tmp_path, value):
    record = pair()
    record["label"] = value
    with pytest.raises(ConversionFailed):
        convert(source(tmp_path, [record]), tmp_path / "out")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("candidate_id"),
        lambda r: r["profile"].update(experience_years="three"),
        lambda r: r["profile"].update(experience_years=99),
        lambda r: r["profile"].update(skills=[123]),
        lambda r: r.update(profile=None),
    ],
)
def test_required_types_and_numeric_ranges_are_validated(tmp_path, mutate):
    record = pair()
    mutate(record)
    with pytest.raises(ConversionFailed):
        convert(source(tmp_path, [record]), tmp_path / "out")


def test_explicit_auxiliary_exclusion_and_schema_version(tmp_path):
    spec = canonical_spec()
    spec.rules.extend(
        [
            Rule(
                source="/metadata",
                disposition="auxiliary",
                reason="Not an ML feature; retain locally",
            ),
            Rule(source="/version", disposition="excluded", reason="Version audited separately"),
        ]
    )
    spec.schema_version_field = "/version"
    spec.expected_schema_version = "v1"
    record = {**pair(), "version": "v1", "metadata": {"nested": "private"}}
    report = convert(source(tmp_path, [record]), tmp_path / "out", spec, retain_source=True)
    assert report["schema_coverage"]["/metadata/nested"]["disposition"] == "auxiliary"
    assert report["outcome_counts"]["converted_with_loss"] == 1
    record["version"] = "v2"
    with pytest.raises(ConversionFailed):
        convert(source(tmp_path, [record]), tmp_path / "changed", spec)
    assert "schema_version_change" in (tmp_path / "changed/conversion_report.json").read_text()


def test_explicit_policy_excludes_without_skipping_outcome(tmp_path):
    spec = canonical_spec()
    spec.exclusions = [
        Exclusion(field="/label", equals=0, reason="Positive-only exploratory subset")
    ]
    report = convert(source(tmp_path, [pair(0), pair(1)]), tmp_path / "out", spec)
    assert report["outcome_counts"]["excluded"] == 1
    assert report["outcome_counts"]["converted"] == 1
    assert report["total_input_records"] == 2
    assert report["record_outcomes"][0]["issues"][0]["reason"]


def test_partial_failures_count_reconciliation_and_no_default_coercion(tmp_path):
    valid, invalid = pair(0), pair(1)
    invalid["profile"]["experience_years"] = "2.0"
    path = source(tmp_path, [valid, invalid])
    with pytest.raises(ConversionFailed):
        convert(path, tmp_path / "strict")
    report = convert(path, tmp_path / "permissive", permissive=True)
    assert report["outcome_counts"] == {
        "converted": 1,
        "converted_with_loss": 0,
        "excluded": 0,
        "rejected": 1,
    }
    assert len(report["record_outcomes"]) == 2
    assert not report["retention"]["full_reversibility"]
    assert report["normalized_written"] == 1
    assert audit_converted(tmp_path / "permissive")["record_count"] == 1


def test_explicit_transform_is_audited_and_bad_values_rejected(tmp_path):
    spec = canonical_spec()
    spec.rules = [r for r in spec.rules if r.target != "/profile/experience_years"]
    spec.rules.append(
        Rule(
            source="/years",
            target="/profile/experience_years",
            disposition="transformed",
            transform="float",
            source_types=["str"],
            reason="Known source string encoding",
        )
    )
    good, bad = {**pair(), "years": "2.5"}, {**pair(1), "years": "not a number"}
    report = convert(source(tmp_path, [good, bad]), tmp_path / "out", spec, permissive=True)
    assert (
        report["outcome_counts"]["converted_with_loss"] == report["outcome_counts"]["rejected"] == 1
    )
    assert any(i["code"] == "transformed_field" for i in report["record_outcomes"][0]["issues"])
    assert "conversion_error" in json.dumps(report["record_outcomes"][1])


def test_mapped_labels_require_exact_types_and_no_ambiguous_mapping(tmp_path):
    spec = canonical_spec()
    spec.rules = [r for r in spec.rules if r.source != "/label"]
    spec.rules.append(
        Rule(
            source="/fit",
            target="/label",
            disposition="transformed",
            transform="label",
            source_types=["str"],
        )
    )
    spec = AdapterSpec.model_validate(
        {**spec.model_dump(), "labels": [{"source": "match", "target": 1}]}
    )
    record = pair()
    del record["label"]
    record["fit"] = "match"
    report = convert(source(tmp_path, [record]), tmp_path / "out", spec)
    assert report["normalized_written"] == 1
    with pytest.raises(ValidationError, match="Ambiguous"):
        AdapterSpec(labels=[{"source": "a", "target": 1}, {"source": "a", "target": 0}])


def test_duplicate_detection_missing_values_near_duplicates_and_quality_policy(tmp_path):
    record = pair()
    record["profile"]["experience_years"] = None
    report = convert(
        source(tmp_path, [record, record]),
        tmp_path / "out",
        policy=QualityPolicy(
            near_duplicate_similarity=0.95, rationale="Test near-identical fixture pairs"
        ),
    )
    assert report["normalized_written"] == 2
    quality = json.loads((tmp_path / "out/quality_report.json").read_text())
    assert quality["exact_duplicate_records"] == quality["source_exact_duplicate_records"] == 1
    assert quality["near_duplicates"]["pair_count"] == 1
    assert (
        quality["source_schema_statistics"]["schema_coverage"]["/profile/experience_years"][
            "null_values"
        ]
        == 2
    )
    with pytest.raises(ConversionFailed):
        convert(
            source(tmp_path, [record]),
            tmp_path / "bad-balance",
            policy=QualityPolicy(
                reference_positive_rate=0.5,
                max_balance_change=0.1,
                rationale="Curated balanced fixture expected",
            ),
        )


def test_leakage_is_checked_even_permissively(tmp_path):
    directory = tmp_path / "splits"
    directory.mkdir()
    record = pair()
    for split in ("train", "test"):
        (directory / (split + ".jsonl")).write_text(json.dumps(record) + "\n")
    with pytest.raises(ConversionFailed):
        convert(directory, tmp_path / "out", permissive=True)
    assert "cross_split_leakage" in (tmp_path / "out/quality_report.json").read_text()


def test_malformed_blank_duplicate_key_json_and_deterministic_source_ids(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_bytes(b'{"label":0,"label":1}\n\nnot json\n')
    rows = source_rows(path, "fixture", "rev")
    assert len(rows) == 3 and all(r.error for r in rows)
    assert [r.source_id for r in rows] == [r.source_id for r in source_rows(path, "fixture", "rev")]
    report = convert(path, tmp_path / "out", permissive=True)
    assert report["outcome_counts"]["rejected"] == 3 and report["count_reconciliation"]


def test_empty_canonical_profiles_and_no_automatic_splitting(tmp_path):
    record = pair()
    record["profile"] = {}
    report = convert(source(tmp_path, [record]), tmp_path / "out")
    assert report["normalized_written"] == 1
    assert (tmp_path / "out/records.jsonl").exists()
    assert not (tmp_path / "out/train.jsonl").exists()
    inspected = inspect_rows(
        source_rows(tmp_path / "source.jsonl", "fixture", "rev"), canonical_spec()
    )
    assert inspected["field_coverage_fraction"] == 1


def test_audit_detects_artifact_tampering(tmp_path):
    convert(source(tmp_path, [pair()]), tmp_path / "out")
    (tmp_path / "out/records.jsonl").write_text("")
    with pytest.raises(ValueError, match="checksum"):
        audit_converted(tmp_path / "out")


def test_numeric_parsing_and_implicit_pydantic_conversions_are_audited(tmp_path):
    record = pair()
    record["profile"]["experience_years"] = 2
    report = convert(source(tmp_path, [record]), tmp_path / "out")
    row = report["record_outcomes"][0]
    assert row["outcome"] == "converted_with_loss"
    assert any(i["code"] == "canonical_type_conversion" for i in row["issues"])
    path = tmp_path / "precision.jsonl"
    path.write_text(
        json.dumps(record).replace(
            '"experience_years": 2', '"experience_years": 2.1234567890123456789'
        )
        + "\n"
    )
    with pytest.raises(ConversionFailed):
        convert(path, tmp_path / "precision-out")
    assert (
        "numeric_precision_or_range_loss"
        in (tmp_path / "precision-out/conversion_report.json").read_text()
    )


def test_integer_conversion_never_rounds_and_strict_cannot_hide_fields_via_exclusions(tmp_path):
    spec = canonical_spec()
    spec.rules = [r for r in spec.rules if r.source != "/label"]
    spec.rules.append(
        Rule(
            source="/label",
            target="/label",
            disposition="transformed",
            transform="integer",
            source_types=["str"],
        )
    )
    record = pair()
    record["label"] = "1.000000000000000000001"
    with pytest.raises(ConversionFailed):
        convert(source(tmp_path, [record]), tmp_path / "rounding", spec)
    spec = canonical_spec()
    spec.exclusions = [Exclusion(field="/label", equals=0, reason="Explicit subset")]
    record = {**pair(), "unnoticed": "private"}
    with pytest.raises(ConversionFailed):
        convert(source(tmp_path, [record]), tmp_path / "exclusion", spec)


def test_mutated_specifications_and_overlapping_targets_are_revalidated(tmp_path):
    spec = canonical_spec()
    spec.rules.append(Rule(source="/another-id", target="/candidate_id"))
    with pytest.raises(ValidationError, match="Duplicate target"):
        convert(source(tmp_path, [pair()]), tmp_path / "out", spec)
    with pytest.raises(ValidationError, match="Overlapping target"):
        AdapterSpec(
            rules=[
                Rule(source="/a", target="/profile"),
                Rule(source="/b", target="/profile/skills"),
            ]
        )


def test_source_file_exclusions_empty_files_and_source_changes_are_visible(tmp_path, monkeypatch):
    from resumeanalysis_legacy.data import conversion

    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "records.jsonl").write_text(json.dumps(pair()) + "\n")
    (directory / "notes.csv").write_text("outside,declared,jsonl,scope\n")
    (directory / "empty.jsonl").write_text("")
    with pytest.raises(ConversionFailed):
        convert(directory, tmp_path / "empty-out")
    report = json.loads((tmp_path / "empty-out/conversion_report.json").read_text())
    assert report["source_files"]["empty.jsonl"]["input_records"] == 0
    assert report["source_file_policy"]["excluded_entries"][0]["name"] == "notes.csv"
    (directory / "empty.jsonl").unlink()
    original = conversion.convert_row

    def change_source(row, spec, permissive):
        result = original(row, spec, permissive)
        (directory / "records.jsonl").write_text("changed while converting\n")
        return result

    monkeypatch.setattr(conversion, "convert_row", change_source)
    with pytest.raises(ConversionFailed):
        convert(directory, tmp_path / "changed-out")
    assert (
        "source_changed_during_conversion"
        in (tmp_path / "changed-out/quality_report.json").read_text()
    )
