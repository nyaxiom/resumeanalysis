"""Publish aggregate evidence only; large originals, text, and derived phrases stay local."""

from collections import Counter, defaultdict
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis_legacy.skill_data.sources import file_sha


def publish_reports(catalog_path, processed, audit_root, coverage_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = loads(Path(catalog_path).read_text())
    preprocessing = loads((Path(processed) / "preprocessing.json").read_text())
    audit = loads((Path(audit_root) / "audit.json").read_text())
    coverage = loads((Path(coverage_root) / "coverage.json").read_text())
    if not (
        preprocessing["status"] == audit["status"] == "passed"
        and preprocessing["catalog_sha256"] == audit["catalog_sha256"] == file_sha(catalog_path)
        and audit["preprocessing_sha256"] == file_sha(Path(processed) / "preprocessing.json")
        and coverage["audit_sha256"] == file_sha(Path(audit_root) / "audit.json")
    ):
        raise ValueError("Evidence chain changed; regenerate preprocessing/audit/coverage")
    inventory = {
        "schema_version": 1,
        "catalog_sha256": file_sha(catalog_path),
        "audit_sha256": file_sha(Path(audit_root) / "audit.json"),
        "units": {
            "samples": "Grouped source examples; CSV token/skill-label rows are counted separately as source_rows.",
            "unique_texts": "Casefolded, whitespace-collapsed texts; global count is not sum of per-file unique counts.",
            "skill_annotations": "SKILL/KNOWLEDGE annotations, including non-span target labels; not all are source mentions.",
            "span_mentions": "Original BIO-derived SKILL/KNOWLEDGE spans in declared space-joined-token text.",
            "unique_skill_labels": "Distinct nonempty normalization/classification strings, not verified taxonomy IDs.",
        },
        "totals": audit["totals"],
        "files": audit["inventory"],
        "sealed": [
            {
                "dataset": f["dataset"],
                "split": f["asset"]["split"],
                "source_sha256": f["asset"]["sha256"],
                "status": f["status"],
            }
            for f in preprocessing["files"]
            if "normalized_file" not in f
        ],
        "dataset_policies": [
            {
                k: d[k]
                for k in (
                    "id",
                    "repository",
                    "revision",
                    "license",
                    "license_evidence_url",
                    "license_evidence_kind",
                    "attribution",
                    "language",
                    "annotation_type",
                    "quality",
                    "risks",
                )
            }
            for d in catalog["datasets"]
        ],
    }
    write_json(output / "inventory.json", inventory)
    quality = {k: v for k, v in audit.items() if k not in {"inventory", "totals"}}
    outcomes_path = Path(audit_root) / "outcomes.jsonl"
    if file_sha(outcomes_path) != audit["outcomes_sha256"]:
        raise ValueError("Audit outcomes changed")
    dispositions = defaultdict(Counter)
    with outcomes_path.open() as stream:
        for line in stream:
            outcome = loads(line)
            dispositions[outcome["file"]].update(outcome["reasons"])
    quality["quarantine_by_partition"] = {
        key: dict(sorted(value.items())) for key, value in sorted(dispositions.items())
    }
    quality["preprocessing"] = [
        {k: v for k, v in f.items() if k not in {"asset", "normalized_file"}}
        | {"source_file": f["asset"]["path"]}
        for f in preprocessing["files"]
    ]
    quality["source_code_sha256"] = {
        p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    write_json(output / "quality_leakage.json", quality)
    write_json(output / "vocabulary_coverage.json", coverage)
    rows = [
        "# Measured dataset inventory",
        "",
        "Generated from checksum-verified originals. Sealed SkillSpan-derived test annotations are excluded from these measured totals. See [research and policy](README.md) for license and compatibility decisions.",
        "",
        "| Dataset / split | Source rows | Samples | Unique texts | Skill annotations | Span mentions | Eligible samples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in audit["inventory"]:
        part = next(
            p
            for p in audit["partitions"]
            if p["dataset"] == item["dataset"] and p["split"] == item["split"]
        )
        rows.append(
            f"| {item['dataset']} / {item['split']} | {item['source_rows']:,} | {item['samples']:,} | {item['unique_texts']:,} | {item['skill_annotations']:,} | {item['span_mentions']:,} | {part['samples']:,} |"
        )
    totals = audit["totals"]
    rows.extend(
        [
            "",
            f"Total: **{totals['samples']:,} samples**, **{totals['unique_texts']:,} unique texts** before quarantine. English: **{totals['languages']['en']:,}**; Russian title/list records: **{totals['languages']['ru']:,}**. Eligible samples include auxiliary classification data and must not be treated as one NER training corpus.",
            "",
            "Skill annotations count span/surface annotations and sentence/title target labels. BIO spans are measured against reconstructed tokenized text, not unavailable original advertisement offsets. Source rows mean sentences for SkillSpan/JSONL, token rows for Green, and label rows for TechWolf CSVs. Differences in these units prevent blind summation.",
            "",
            "The machine-readable [inventory](inventory.json), [quality/leakage report](quality_leakage.json) and [validation coverage results](vocabulary_coverage.json) retain counts, hashes and definitions. Original data, source text, fingerprint sketches, per-record outcomes and vocabulary artifacts stay outside Git.",
            "",
        ]
    )
    (output / "inventory.md").write_text("\n".join(rows))
    return {
        "reports": [
            "inventory.md",
            "inventory.json",
            "quality_leakage.json",
            "vocabulary_coverage.json",
        ]
    }
