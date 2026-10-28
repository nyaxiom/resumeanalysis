# Online recommendation validation record

Validation date: 2026-10-27. The feature branch is
`feature/online-job-recommender`, based on `feature/local-cover-letter-generator`
at `f31f48a`. Main is not a target for this migration.

## Executed tests

The complete local suite passed **600 tests in 64.02 seconds**, including the
optional checks using cached E5 and re-ranker models:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RESUMEANALYSIS_RUN_E5_TESTS=1 RESUMEANALYSIS_RUN_RERANK_TESTS=1 \
  .venv/bin/pytest -q
```

The new tests cover cold start, online updates, user isolation, restart, replay,
reset, idempotent retries, explicit corrections, commit failure rollback,
concurrent repository instances, unknown attributes, independent hard constraints,
feature/configuration compatibility and immutable profile versions. Integration
checks cover both observed ApplyPilot link shapes and its unmerged vacancy DTO,
the legacy/cover-letter adapters, CLI persistence and no network connections
during local ranking/feedback. Chronological evaluation tests verify prediction
before learning and isolation from other users and unobserved pool labels.

Profile tests exercise TXT/PDF/DOCX parsing, malformed/oversized inputs, strict
structured schemas, source evidence, versioned edits and private error handling.
Instructor's actual client is exercised with mocked provider responses, including
invalid outputs. This is not a live local-LLM extraction-quality evaluation.

The [benchmark script](../scripts/benchmark_online_recommender.py) ran with actual
local E5 and River. Its archived trace contains 128 chronological labels from an
explicitly synthetic scenario. The [CLI demonstration](../evaluation/online_recommendations/cli_demo.json)
ran recommendation, feedback and restored recommendation in separate processes.
See [measurements and interpretation](online_recommender_benchmarks.md).

Ruff lint and formatting checks passed, as did `git diff --check`. Both source and
wheel packages built successfully. Wheel inspection verified all 15 new Python
modules and 120 license/notice files, including the 41 new notices. No checkpoints,
SQLite databases or private attachments were packaged. The
[machine-readable validation record](../evaluation/online_recommendations/validation.json)
retains benchmark/script/lockfile hashes and executed-check results.

## Preservation and provenance

SHA-256 checks against the pre-migration snapshot covered 101 existing source,
configuration, checkpoint and evaluation files. The only changed file in that
snapshot is the root CLI dispatcher, which adds the new commands. A second
16-file baseline covering the original matching code, saved checkpoints and
SkillSpan artifacts remained unchanged. No matching model was retrained and no
held-out SkillSpan labels were used in this task.

Original `THIRD_PARTY_NOTICES.md` content remains byte-for-byte as the prefix of
the extended file. The dependency audit records 32 newly resolved releases and
41 complete license/notice files, with verified SHA-256 digests. Existing notices
and the project MIT license remain intact. Upstream research revisions and file
hashes are in [source evidence](online_source_evidence.json); no ApplyPilot,
Unstructured, River or Instructor implementation was copied.

Generated JSON schemas are exported from the actual Pydantic definitions using
[the schema export script](../scripts/export_online_schemas.py). Model weights,
SQLite databases, caches and private documents are excluded from the change.

## Limits and next steps

Real recommendation improvement is not established. The small synthetic fixture
has mixed ranking/loss results and adaptation lag after preferences reverse.
Interest scores are uncalibrated. Compact features remain the default; projected
features are an explicit experimental option.

ApplyPilot does not yet provide the necessary deployed profile/feedback contracts;
this branch supplies tested adapters and a [host integration guide](online_recommender_integration.md).
Its unlicensed source was inspected, not imported. Local Ollama deployment,
scanned-document OCR, large-history load, retention, configuration migrations and
probability calibration remain follow-up work. Existing cover-letter factuality
limitations remain unchanged. The feature is delivered for review as a draft PR.
