import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from resumeanalysis.data.dataset import SPLITS, assert_no_leakage, load_records
from resumeanalysis.data.download import sha256_file
from resumeanalysis.data.prepare import partition, prepare_candidates, prepare_med2425
from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.features.feature_builder import FEATURE_NAMES, FeatureEngineering


def test_candidate_adapter_repeatability_and_hard_negatives(tmp_path):
    raw = tmp_path / "raw"
    tables = {k: [] for k in ["resumes", "jobs", "matches"]}
    # Produce independent authored entities for each deterministic partition.
    for split in SPLITS:
        profiles = []
        counter = 0
        while len(profiles) < 6:
            text = f"Document {split} candidate {counter}"
            counter += 1
            if partition(text, 42) != split:
                continue
            rid = f"{split}-r{len(profiles)}"
            profiles.append(rid)
            tables["resumes"].append(
                {
                    "resume_id": rid,
                    "summary": text,
                    "skills": ["Python", "SQL"] if len(profiles) <= 3 else ["SQL"],
                    "years_experience": 2,
                    "experience_bullets": [],
                }
            )
        counter = count = 0
        while count < 4:
            text = f"Document {split} vacancy {counter}"
            counter += 1
            if partition(text, 42) != split:
                continue
            jid = f"{split}-j{count}"
            count += 1
            tables["jobs"].append(
                {
                    "job_id": jid,
                    "description": text,
                    "job_title": "Engineer",
                    "must_have_skills": ["Python", "SQL"],
                    "nice_to_have_skills": [],
                }
            )
            tables["matches"].append({"job_id": jid, "relevant_resume_ids": profiles[:3]})
    checksums = {}
    for name, rows in tables.items():
        path = raw / name / "train-00000-of-00001.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        checksums[str(path.relative_to(raw))] = sha256_file(path)
    (raw / "manifest.json").write_text(json.dumps({"license": "mit", "sha256": checksums}))
    a = prepare_candidates(raw, tmp_path / "a", max_jobs=6, pairs_per_job=1)
    b = prepare_candidates(raw, tmp_path / "b", max_jobs=6, pairs_per_job=1)
    assert a == b
    splits = {s: load_records(tmp_path / "a" / f"{s}.jsonl") for s in SPLITS}
    assert_no_leakage(splits)
    builder = FeatureEngineering(LexicalSmokeEncoder())
    for rows in splits.values():
        assert {r.label for r in rows} == {0, 1}
        for row in rows:
            if row.label == 0:
                assert row.hard_negative
                f = builder.build(row.profile, row.vacancy)
                assert f.vector[FEATURE_NAMES.index("required_match")] < 0.6


def test_med2425_refuses_missing_license_evidence(tmp_path):
    path = tmp_path / "license.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="license"):
        prepare_med2425(
            tmp_path / "not-downloaded.parquet", tmp_path / "out", license_evidence=str(path)
        )
