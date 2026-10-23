"""Explicit adapters for incompatible annotation schemes; no inferred gold offsets."""

import csv
from collections import OrderedDict
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.skill_data.schema import Annotation, Provenance, SkillRecord
from resumeanalysis.skill_data.sources import (
    SEALED_SHAS,
    asset_path,
    catalog,
    digest,
    file_sha,
    verify,
    write_jsonl,
)


def bio_annotations(tokens, tags, default_category=None):
    if not isinstance(tokens, list) or not isinstance(tags, list):
        raise ValueError("Tokens and BIO tags must be arrays")
    if len(tags) != len(tokens):
        raise ValueError("BIO length mismatch")
    starts, ends, cursor = [], [], 0
    for token in tokens:
        if not isinstance(token, str) or not token or token.isspace():
            raise ValueError("Empty/nonstring token")
        starts.append(cursor)
        ends.append(cursor + len(token))
        cursor += len(token) + 1
    text = " ".join(tokens)
    spans, begin, category = [], None, None
    for i, tag in enumerate([*tags, "O"]):
        if not isinstance(tag, str):
            raise ValueError("Nonstring BIO tag")
        prefix, _, label = tag.partition("-")
        label = label or default_category
        if (
            prefix not in {"B", "I", "O"}
            or (prefix != "O" and not label)
            or (prefix == "O" and tag != "O")
        ):
            raise ValueError("Unknown BIO tag")
        if prefix == "I" and (begin is None or label != category):
            raise ValueError("Invalid BIO transition; no silent repair")
        if begin is not None and (prefix != "I" or label != category):
            spans.append(
                Annotation(
                    category=category.upper(),
                    start=starts[begin],
                    end=ends[i - 1],
                    surface=text[starts[begin] : ends[i - 1]],
                )
            )
            begin = None
        if prefix == "B":
            begin, category = i, label
    return spans


def source_groups(path, fmt):
    if file_sha(path) in SEALED_SHAS:
        raise ValueError(
            "SkillSpan test annotation ingestion is forbidden, including renamed files"
        )
    if fmt in {"skillspan", "ru_titles"}:
        with Path(path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                row = loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL source row must be an object")
                yield [number], [row]
        return
    groups = OrderedDict()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if len(set(reader.fieldnames or [])) != len(reader.fieldnames or []):
            raise ValueError("Duplicate CSV column")
        for number, row in enumerate(reader, 2):
            if None in row or None in row.values():
                raise ValueError("Malformed CSV row")
            if fmt == "green":
                key = row["sentence_id"]
            elif fmt == "skill_xl":
                key = (row["ID"], row["subset"], row["sentence"])
            else:
                key = row["sentence"]
            numbers, items = groups.setdefault(key, ([], []))
            numbers.append(number)
            items.append(row)
    yield from groups.values()


EXPECTED = {
    "skillspan": {"idx", "tokens", "tags_skill", "tags_knowledge", "source"},
    "green": {"sentence_id", "word", "pos", "tag"},
    "surface": {"sentence", "span", "sub_span", "label"},
    "sentence_labels": {"sentence", "label"},
    "skill_xl": {"ID", "subset", "title", "sentence", "relevant", "cluster", "skill"},
    "ru_titles": {"title", "skills", "experience", "source"},
}


def convert_group(dataset, asset, numbers, rows):
    fmt = asset["format"]
    if asset["role"] == "sealed_test" or asset["sha256"] in SEALED_SHAS:
        raise ValueError("Sealed annotations cannot be converted")
    expected = EXPECTED[fmt]
    if fmt == "sentence_labels" and dataset["id"] == "synthetic_esco":
        expected = {"sentence", "skill"}
    for row in rows:
        if set(row) != expected:
            raise ValueError(f"Unexpected/missing source fields for {fmt}: {set(row) ^ expected}")
    first, annotations, tokens, group = rows[0], [], None, None
    origin = "source"
    if fmt == "skillspan":
        tokens = first["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("Missing source tokens")
        annotations = bio_annotations(tokens, first["tags_skill"], "SKILL") + bio_annotations(
            tokens, first["tags_knowledge"], "KNOWLEDGE"
        )
        text, origin = " ".join(tokens), "space_joined_tokens"
        group = f"{first['source']}:{first['idx']}"
    elif fmt == "green":
        tokens = [r["word"] for r in rows]
        annotations = bio_annotations(tokens, [r["tag"] for r in rows])
        if any(
            a.category not in {"SKILL", "QUALIFICATION", "EXPERIENCE", "OCCUPATION", "DOMAIN"}
            for a in annotations
        ):
            raise ValueError("Unknown Green entity category")
        text, origin = " ".join(tokens), "space_joined_tokens"
        # A Green sentence id is NOT evidence of an original job/document id.
    elif fmt == "ru_titles":
        text = first["title"]
        if not isinstance(first["skills"], list) or any(
            not isinstance(s, str) or not s for s in first["skills"]
        ):
            raise ValueError("Invalid skill list")
        annotations = [Annotation(category="SKILL", skill_label=s) for s in first["skills"]]
    else:
        text = first["sentence"]
        for row in rows:
            if fmt == "surface":
                if not row["span"]:
                    raise ValueError("Missing source surface mention")
                unmapped = row["label"] == "LABEL NOT PRESENT"
                annotations.append(
                    Annotation(
                        category="UNMAPPED_SKILL" if unmapped else "SKILL",
                        surface=row["span"],
                        skill_label=None if unmapped else row["label"],
                        attributes={"sub_span": row["sub_span"], "source_label": row["label"]},
                    )
                )
            elif fmt == "sentence_labels":
                label = row.get("label", row.get("skill"))
                if not label:
                    raise ValueError("Missing sentence skill label")
                annotations.append(Annotation(category="SKILL", skill_label=label))
            elif fmt == "skill_xl":
                if row["relevant"] not in {"True", "False"}:
                    raise ValueError("Unrecognized relevance value")
                annotations.append(
                    Annotation(
                        category="SKILL" if row["skill"] else "RELEVANCE",
                        skill_label=row["skill"] or None,
                        attributes={
                            "relevant": row["relevant"] == "True",
                            "cluster": row["cluster"],
                        },
                    )
                )
                group = row["ID"]
    identity = digest(f"{dataset['id']}:{asset['sha256']}:{numbers}".encode())
    return SkillRecord(
        id=identity,
        text=text,
        text_origin=origin,
        language=dataset["language"],
        annotation_type=dataset["annotation_type"],
        annotations=annotations,
        tokens=tokens,
        provenance=Provenance(
            dataset=dataset["id"],
            repository=dataset["repository"],
            revision=dataset["revision"],
            source_file=asset["path"],
            source_sha256=asset["sha256"],
            source_rows=numbers,
            split=asset["split"],
            role=asset["role"],
            license=dataset["license"],
            license_url=(
                "https://creativecommons.org/licenses/by/4.0/"
                if dataset["license"] == "CC-BY-4.0"
                else "https://opensource.org/license/mit"
            ),
            license_evidence_url=dataset.get("license_evidence_url"),
            attribution=dataset["attribution"],
            group_id=group,
        ),
        source_records=rows,
    )


def preprocess(catalog_path, raw_root, output):
    output, raw_root = Path(output).resolve(), Path(raw_root).resolve()
    if output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError("Raw and processed directories must be separate")
    if output.exists():
        raise FileExistsError("Preprocessing requires a fresh output directory")
    output.mkdir(parents=True)
    document = catalog(catalog_path)
    report = {"schema_version": 1, "catalog_sha256": file_sha(catalog_path), "files": []}
    errors = False
    for dataset in document["datasets"]:
        for asset in dataset["assets"]:
            source = asset_path(raw_root, dataset, asset)
            verify(source, asset)
            if asset["role"] == "metadata":
                continue
            info = {"dataset": dataset["id"], "asset": asset, "converted": 0, "rejected": []}
            report["files"].append(info)
            if asset["role"] == "sealed_test":
                info["status"] = "sealed: bytes verified; annotations not decoded"
                continue
            target = output / dataset["id"] / (asset["split"] + ".jsonl")
            target.parent.mkdir(parents=True, exist_ok=True)
            info["normalized_file"] = str(target.relative_to(output))
            info["source_units"] = info["input_rows"] = 0

            def records():
                nonlocal errors
                try:
                    for numbers, rows in source_groups(source, asset["format"]):
                        info["source_units"] += 1
                        info["input_rows"] += len(rows)
                        try:
                            record = convert_group(dataset, asset, numbers, rows)
                        except (ValueError, KeyError, TypeError) as exc:
                            errors = True
                            info["rejected"].append(
                                {"source_rows": numbers, "reason": str(exc)[:500]}
                            )
                            continue
                        info["converted"] += 1
                        yield record.model_dump()
                except (ValueError, KeyError, TypeError) as exc:
                    errors = True
                    info["rejected"].append({"reason": str(exc)[:500], "scope": "file"})

            write_jsonl(target, records())
            if not info["source_units"]:
                errors = True
                info["rejected"].append({"reason": "Empty source dataset", "scope": "file"})
            verify(source, asset)
            info["normalized_sha256"] = file_sha(target)
            info["status"] = "failed" if info["rejected"] else "passed"
    report["status"] = "failed" if errors else "passed"
    write_json(output / "preprocessing.json", report)
    if errors:
        raise ValueError(
            "Preprocessing rejected source data; see preprocessing.json; audit disabled"
        )
    return report
