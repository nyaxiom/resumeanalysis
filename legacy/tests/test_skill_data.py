"""Authored fixtures only: never decode or evaluate the real SkillSpan test gold."""

import io
import json

import pytest
from pydantic import ValidationError
from resumeanalysis_legacy.skill_data import audit as auditing
from resumeanalysis_legacy.skill_data import guard, sources
from resumeanalysis_legacy.skill_data import preprocess as preprocessing
from resumeanalysis_legacy.skill_data.audit import audit, partition_records
from resumeanalysis_legacy.skill_data.coverage import evaluate_validation, fit_phrases
from resumeanalysis_legacy.skill_data.guard import NearIndex, shingles, text_hash, token_text_only
from resumeanalysis_legacy.skill_data.preprocess import bio_annotations, convert_group, preprocess
from resumeanalysis_legacy.skill_data.schema import Annotation, Provenance, SkillRecord
from resumeanalysis_legacy.skill_data.sources import TEST_SHA, digest, file_sha, write_jsonl


def dataset(identifier="skillspan", annotation_type="span"):
    return {
        "id": identifier,
        "repository": "https://example.test/fixture",
        "revision": "a" * 40,
        "base_url": "https://example.test/fixture/" + "a" * 40 + "/",
        "license": "MIT",
        "language": "en",
        "annotation_type": annotation_type,
        "attribution": "Authored test fixture",
    }


def asset(fmt="skillspan", role="train", split="train"):
    return {"format": fmt, "role": role, "split": split, "sha256": "b" * 64, "path": "source"}


def record(
    identifier, text, *, role="train", split="train", name="fixture", group=None, category="SKILL"
):
    return SkillRecord(
        id=identifier,
        text=text,
        text_origin="source",
        language="en",
        annotation_type="span",
        annotations=[Annotation(category=category, surface=text, start=0, end=len(text))],
        provenance=Provenance(
            dataset=name,
            repository="https://example.test",
            revision="a" * 40,
            source_file=split + ".jsonl",
            source_sha256="b" * 64,
            source_rows=[1],
            split=split,
            role=role,
            license="MIT",
            attribution="Authored test fixture",
            group_id=group,
        ),
        source_records=[{"fixture_id": identifier}],
    )


def test_bio_uses_unicode_character_offsets_and_preserves_distinct_labels():
    tokens = ["Нужны", "анализ", "данных", "и", "Python"]
    anns = bio_annotations(tokens, ["O", "B-Skill", "I-Skill", "O", "B-Domain"])
    text = " ".join(tokens)
    assert [(a.category, a.surface) for a in anns] == [
        ("SKILL", "анализ данных"),
        ("DOMAIN", "Python"),
    ]
    assert all(text[a.start : a.end] == a.surface for a in anns)
    assert anns[0].start == len("Нужны ")


@pytest.mark.parametrize(
    "tokens,tags",
    [
        (["Python"], []),
        (["Python"], ["I-Skill"]),
        (["Python", "SQL"], ["B-Skill", "I-Domain"]),
        (["Python"], ["X-Skill"]),
        (["Python"], ["O-Skill"]),
        (["Python"], "B"),
        ([""], ["O"]),
        (["Python"], [3]),
    ],
)
def test_invalid_bio_is_not_repaired_or_silently_dropped(tokens, tags):
    with pytest.raises((ValueError, TypeError)):
        bio_annotations(tokens, tags)


def test_skillspan_adapter_retains_original_tokens_tags_document_metadata():
    original = {
        "idx": 5,
        "tokens": ["Python", "and", "teamwork"],
        "tags_skill": ["O", "O", "B"],
        "tags_knowledge": ["B", "O", "O"],
        "source": "house",
    }
    converted = convert_group(dataset(), asset(), [19], [original])
    assert converted.text == "Python and teamwork"
    assert converted.text_origin == "space_joined_tokens"
    assert converted.tokens == original["tokens"]
    assert converted.source_records == [original]
    assert converted.provenance.source_rows == [19]
    assert converted.provenance.group_id == "house:5"
    assert {a.category for a in converted.annotations} == {"SKILL", "KNOWLEDGE"}


def test_surface_normalization_never_invents_offsets_for_ambiguous_mentions():
    row = {"sentence": "Python with Python", "span": "Python", "sub_span": "", "label": "Python"}
    converted = convert_group(
        dataset("tech", "surface_normalization"), asset("surface"), [2], [row]
    )
    assert converted.annotations[0].surface == "Python"
    assert converted.annotations[0].start is converted.annotations[0].end is None
    assert converted.annotations[0].skill_label == "Python"
    row = row | {"label": "LABEL NOT PRESENT"}
    converted = convert_group(
        dataset("tech", "surface_normalization"), asset("surface"), [3], [row]
    )
    assert converted.annotations[0].category == "UNMAPPED_SKILL"
    assert converted.annotations[0].skill_label is None
    assert converted.source_records == [row]


def test_title_labels_need_not_occur_in_title_and_remain_nonspan():
    row = {
        "title": "Разработчик",
        "skills": ["Python", "SQL"],
        "experience": "1-3",
        "source": "fixture",
    }
    ds = dataset("ru_titles", "title_multilabel") | {"language": "ru"}
    converted = convert_group(ds, asset("ru_titles", "research_only", "unsplit"), [1], [row])
    assert converted.text == row["title"]
    assert all(a.start is None and a.surface is None for a in converted.annotations)
    assert [a.skill_label for a in converted.annotations] == ["Python", "SQL"]
    assert converted.provenance.role == "research_only"


def test_green_retains_pos_and_non_skill_entities():
    rows = [
        {"sentence_id": "4", "word": "Python", "pos": "NN", "tag": "B-Skill"},
        {"sentence_id": "4", "word": "degree", "pos": "NN", "tag": "B-Qualification"},
    ]
    converted = convert_group(dataset("green"), asset("green"), [2, 3], rows)
    assert converted.source_records == rows
    assert [a.category for a in converted.annotations] == ["SKILL", "QUALIFICATION"]
    assert converted.provenance.group_id is None


def test_skill_xl_preserves_relevance_clusters_and_multiple_labels():
    row = {
        "ID": "job-1",
        "subset": "RANDOM",
        "title": "Developer",
        "sentence": "Write software",
        "relevant": "True",
        "cluster": "1",
        "skill": "develop software",
    }
    rows = [row, row | {"skill": "programming", "cluster": "2"}]
    converted = convert_group(
        dataset("skill_xl", "sentence_multilabel"), asset("skill_xl"), [2, 3], rows
    )
    assert converted.source_records == rows
    assert converted.provenance.group_id == "job-1"
    assert [a.skill_label for a in converted.annotations] == ["develop software", "programming"]
    assert [a.attributes["cluster"] for a in converted.annotations] == ["1", "2"]
    assert all(a.start is None for a in converted.annotations)


def test_unknown_fields_and_sealed_test_are_rejected():
    row = {"sentence": "Python", "label": "Python", "unrecognized": 1}
    with pytest.raises(ValueError, match="source fields"):
        convert_group(
            dataset("techwolf", "sentence_multilabel"), asset("sentence_labels"), [2], [row]
        )
    with pytest.raises(ValueError, match="Sealed"):
        convert_group(dataset(), asset(role="sealed_test"), [1], [{}])
    with pytest.raises(ValueError, match="Sealed"):
        convert_group(dataset(), asset() | {"sha256": TEST_SHA}, [1], [{}])


def test_renamed_test_bytes_are_blocked_before_annotation_decoder(monkeypatch, tmp_path):
    source = tmp_path / "renamed-train.jsonl"
    source.write_text("not JSON: should never be decoded")
    monkeypatch.setattr(preprocessing, "file_sha", lambda _: TEST_SHA)
    with pytest.raises(ValueError, match="renamed"):
        list(preprocessing.source_groups(source, "skillspan"))


def test_schema_rejects_fabricated_offsets_and_source_misalignment():
    original = record("a", "Python").model_dump()
    for update in [
        {"annotation_type": "sentence_multilabel"},
        {"annotations": [{"category": "SKILL", "surface": "SQL", "start": 0, "end": 6}]},
        {"annotations": [{"category": "SKILL", "surface": "Python", "start": -1, "end": 6}]},
        {"annotations": [{"category": "SKILL", "surface": "Python", "start": 0}]},
        {"text_origin": "space_joined_tokens", "tokens": ["SQL"]},
        {"annotations": [{"category": "SKILL", "surface": "Python"}]},
    ]:
        with pytest.raises(ValidationError):
            SkillRecord.model_validate(original | update)


def make_catalog(path, payload, *, role="train", fmt="skillspan"):
    ds = dataset()
    info = asset(fmt, role) | {
        "path": "source.jsonl",
        "sha256": digest(payload),
        "bytes": len(payload),
    }
    ds["assets"] = [info]
    path.write_text(json.dumps({"schema_version": 1, "datasets": [ds]}))
    return ds, info


def test_download_pins_bytes_is_idempotent_and_refuses_changed_original(monkeypatch, tmp_path):
    payload = b"original source bytes\r\n"
    config = tmp_path / "catalog.json"
    ds, info = make_catalog(config, payload)
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(sources.urllib.request, "urlopen", open_url)
    root = tmp_path / "raw"
    sources.download(config, root)
    target = sources.asset_path(root, ds, info)
    assert target.read_bytes() == payload
    sources.download(config, root)
    assert len(calls) == 1
    target.write_bytes(b"altered")
    with pytest.raises(ValueError, match="mismatch"):
        sources.download(config, root)
    assert target.read_bytes() == b"altered"


@pytest.mark.parametrize("downloaded", [b"bad", b"oversized bytes"])
def test_bad_download_never_publishes_a_source(monkeypatch, tmp_path, downloaded):
    config = tmp_path / "catalog.json"
    ds, info = make_catalog(config, b"good")
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(downloaded))
    with pytest.raises(ValueError):
        sources.download(config, tmp_path / "raw")
    destination = sources.asset_path(tmp_path / "raw", ds, info)
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_preprocessing_unknown_field_fails_closed_and_retains_raw_bytes(tmp_path):
    payload = (
        json.dumps(
            {
                "idx": 1,
                "tokens": ["Python"],
                "tags_skill": ["B"],
                "tags_knowledge": ["O"],
                "source": "tech",
                "unknown": 1,
            }
        )
        + "\n"
    ).encode()
    config = tmp_path / "catalog.json"
    ds, info = make_catalog(config, payload)
    source = sources.asset_path(tmp_path / "raw", ds, info)
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="rejected"):
        preprocess(config, tmp_path / "raw", output)
    report = json.loads((output / "preprocessing.json").read_text())
    assert report["status"] == "failed"
    assert report["files"][0]["converted"] == 0
    assert report["files"][0]["rejected"][0]["source_rows"] == [1]
    assert source.read_bytes() == payload


def test_empty_source_is_rejected_instead_of_reporting_a_successful_corpus(tmp_path):
    config = tmp_path / "catalog.json"
    ds, info = make_catalog(config, b"")
    source = sources.asset_path(tmp_path / "raw", ds, info)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="rejected"):
        preprocess(config, tmp_path / "raw", tmp_path / "processed")


def test_text_guard_decodes_only_tokens_even_when_annotations_are_not_json(monkeypatch, tmp_path):
    line = '{"idx": 1, "tokens": ["A", "frozen", "sentence"], "tags_skill": deliberately invalid}'
    assert token_text_only(line) == "A frozen sentence"
    source = tmp_path / "sealed.jsonl"
    source.write_text(line + "\n")
    monkeypatch.setattr(guard, "TEST_SHA", file_sha(source))
    # Pin validation is separately tested below; this authored fixture tests the
    # text-only decoder without requiring or touching any real held-out data.
    monkeypatch.setattr(guard, "load_guard", lambda path: json.loads(path.read_text()))
    output = tmp_path / "guard.json"
    guard.build_guard(source, output)
    text = output.read_text()
    assert "A frozen sentence" not in text
    assert "deliberately invalid" not in text
    result = json.loads(text)
    assert result["source_rows"] == 1
    assert list(result["fingerprints"]) == [text_hash("A frozen sentence")]


def test_near_duplicate_candidates_verify_jaccard_and_fail_closed_on_bucket_limit():
    index = NearIndex(threshold=0.85, bucket_limit=1)
    index.add("a", {1, 2, 3})
    assert index.query({1, 2, 3}) == [("a", 1.0)]
    index.add("b", {1, 2, 3})
    with pytest.raises(ValueError, match="incomplete"):
        index.query({1, 2, 3})


def make_audit_input(
    tmp_path, partitions, monkeypatch, heldout="An authored heldout sentence about skills"
):
    processed = tmp_path / "processed"
    processed.mkdir()
    files = []
    for relative, records in partitions.items():
        target = processed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(target, (r.model_dump() for r in records))
        files.append(
            {
                "dataset": records[0].provenance.dataset,
                "asset": {"split": records[0].provenance.split, "role": records[0].provenance.role},
                "input_rows": len(records),
                "normalized_file": relative,
                "normalized_sha256": file_sha(target),
            }
        )
    (processed / "preprocessing.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "catalog_sha256": "c" * 64,
                "files": files,
            }
        )
    )
    guard_path = tmp_path / "guard.json"
    guard_path.write_text(
        json.dumps(
            {
                "source_sha256": TEST_SHA,
                "source_rows": 3569,
                "fingerprints": {text_hash(heldout): sorted(shingles(heldout))},
            }
        )
    )
    # The production load_guard pin is intentionally bypassed only for authored
    # synthetic partition fixtures, never for the real datasets.
    monkeypatch.setattr(auditing, "load_guard", lambda path: json.loads(path.read_text()))
    return processed, guard_path


def test_audit_quarantines_heldout_and_cross_split_data_without_concatenating_tasks(
    tmp_path, monkeypatch
):
    frozen = "An authored heldout sentence about skills"
    shared = "Shared validation sentence about Python"
    partitions = {
        "fixture/train.jsonl": [
            record("test-overlap", frozen),
            record("validation-overlap", shared),
            record("duplicate-a", "Repeated fixture skills sentence"),
            record("duplicate-b", "Repeated fixture skills sentence"),
            record("conflict-a", "Contradictory fixture skills sentence", category="SKILL"),
            record("conflict-b", "Contradictory fixture skills sentence", category="KNOWLEDGE"),
            record("unique", "Unique training example involving databases"),
        ],
        "fixture/validation.jsonl": [record("dev", shared, role="validation", split="validation")],
        "research/unsplit.jsonl": [
            record(
                "research",
                "Title without skill spans",
                role="research_only",
                split="unsplit",
                name="research",
            )
        ],
    }
    processed, guard_path = make_audit_input(tmp_path, partitions, monkeypatch)
    output = tmp_path / "audit"
    report = audit(processed, guard_path, output)
    assert report["status"] == "passed"
    assert report["totals"]["samples"] == 9
    reasons = {
        r["id"]: r["reasons"]
        for r in map(json.loads, (output / "outcomes.jsonl").read_text().splitlines())
    }
    assert "exact_skillspan_test_text" in reasons["test-overlap"]
    assert "exact_cross_role_overlap" in reasons["validation-overlap"]
    assert "duplicate_text_conflicting_annotations" in reasons["conflict-a"]
    assert "duplicate_text_conflicting_annotations" in reasons["conflict-b"]
    assert "within_partition_exact_duplicate" in reasons["duplicate-b"]
    assert "research_only_license_or_task_review" in reasons["research"]
    assert {r.id for r in partition_records(output, roles={"train"})} == {"duplicate-a", "unique"}
    assert {r.id for r in partition_records(output, roles={"validation"})} == {"dev"}
    assert not list(partition_records(output, roles={"external_evaluation"}))
    assert not list(partition_records(output, roles={"train"}, datasets=set()))
    assert report["near_duplicates"]["complete_pairwise_search"] is False
    assert report["near_duplicates"]["truncated"] is False
    partition = output / "partitions/fixture/train.jsonl"
    partition.write_text(partition.read_text() + "\n")
    with pytest.raises(ValueError, match="integrity"):
        list(partition_records(output, roles={"train"}))


def test_document_ids_quarantine_distinct_sentences_from_same_skill_xl_job(tmp_path, monkeypatch):
    partitions = {
        "skill_xl/train.jsonl": [
            record("train", "Fit text with one set of skills", name="skill_xl", group="job-1")
        ],
        "skill_xl/validation.jsonl": [
            record(
                "dev",
                "Validation text with different skills",
                role="validation",
                split="validation",
                name="skill_xl",
                group="job-1",
            )
        ],
    }
    processed, guard_path = make_audit_input(tmp_path, partitions, monkeypatch)
    output = tmp_path / "audit"
    audit(processed, guard_path, output)
    assert not list(partition_records(output, roles={"train"}))
    assert [r.id for r in partition_records(output, roles={"validation"})] == ["dev"]


def test_audit_rejects_changed_normalized_source(tmp_path, monkeypatch):
    processed, guard_path = make_audit_input(
        tmp_path, {"fixture/train.jsonl": [record("a", "Training text")]}, monkeypatch
    )
    (processed / "fixture/train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed since preprocessing"):
        audit(processed, guard_path, tmp_path / "audit")


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "../raw"),
        ("id", "/tmp/escape"),
        ("id", "unsafe/slug"),
        ("revision", "main"),
        ("revision", "../raw"),
    ],
)
def test_catalog_disallows_source_namespace_escape_and_unpinned_revision(tmp_path, key, value):
    path = tmp_path / "catalog.json"
    make_catalog(path, b"fixture")
    config = json.loads(path.read_text())
    config["datasets"][0][key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        sources.catalog(path)


def test_catalog_disallows_split_output_path_escape(tmp_path):
    path = tmp_path / "catalog.json"
    make_catalog(path, b"fixture")
    config = json.loads(path.read_text())
    config["datasets"][0]["assets"][0]["split"] = "../escape"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        sources.catalog(path)


@pytest.mark.parametrize("sealed_sha", sorted(sources.SEALED_SHAS))
def test_catalog_cannot_relabel_skillspan_or_derived_test_as_training(tmp_path, sealed_sha):
    path = tmp_path / "catalog.json"
    make_catalog(path, b"fixture")
    config = json.loads(path.read_text())
    config["datasets"][0]["assets"][0]["sha256"] = sealed_sha
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="sealed"):
        sources.catalog(path)


def test_forged_or_incomplete_guard_cannot_disable_heldout_protection(tmp_path):
    path = tmp_path / "guard.json"
    path.write_text(
        json.dumps(
            {
                "source_sha256": TEST_SHA,
                "source_rows": 3569,
                "fingerprints": {},
            }
        )
    )
    with pytest.raises(ValueError, match="integrity"):
        guard.load_guard(path)


def test_vocabulary_fitter_reads_only_audited_training_spans(tmp_path, monkeypatch):
    partitions = {
        "fixture/train.jsonl": [record("fit", "PostgreSQL databases")],
        "fixture/validation.jsonl": [
            record("dev", "Python programming", role="validation", split="validation")
        ],
        "fixture/test.jsonl": [
            record(
                "external",
                "Never harvest this external label",
                role="external_evaluation",
                split="test",
            )
        ],
    }
    processed, guard_path = make_audit_input(tmp_path, partitions, monkeypatch)
    output = tmp_path / "audit"
    audit(processed, guard_path, output)
    result = fit_phrases(output, ["fixture"], min_frequency=1)
    assert result["phrases"] == ["postgresql databases"]
    assert result["training_rows"] == result["training_mentions"] == 1
    assert result["training_sources"] == {"fixture": "b" * 64}
    with pytest.raises(ValueError, match="training datasets"):
        fit_phrases(output, [])


def test_validation_diagnostic_distinguishes_known_unknown_and_multiword_mentions():
    from resumeanalysis.extraction.skill_extractor import SkillExtractor

    row = {
        "idx": 1,
        "tokens": ["Python", "and", "manage", "teams"],
        "tags_skill": ["O", "O", "B", "I"],
        "tags_knowledge": ["B", "O", "O", "O"],
        "source": "tech",
    }
    source = convert_group(dataset(), asset(role="validation", split="validation"), [1], [row])
    result = evaluate_validation([source], SkillExtractor())
    assert result["gold_mentions"] == 2
    assert result["token_exact"]["tp"] == 1
    assert result["token_exact"]["fn"] == 1
    assert result["vocabulary"]["gold_alias_coverage"] == 0.5
    assert result["groups"]["alias_known"]["recall"] == 1.0
    assert result["groups"]["alias_unknown"]["recall"] == 0.0
    assert result["groups"]["multi_word"]["recall"] == 0.0
    assert result["groups"]["contextual_surface_proxy"]["gold"] == 1
    with pytest.raises(ValueError, match="only audited SkillSpan validation"):
        evaluate_validation(
            [
                source.model_copy(
                    update={
                        "provenance": source.provenance.model_copy(
                            update={"role": "external_evaluation"}
                        )
                    }
                )
            ],
            SkillExtractor(),
        )
