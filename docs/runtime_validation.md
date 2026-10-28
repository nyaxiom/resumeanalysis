# Runtime refactor validation

Validated on 2026-10-28 against the merged source at `52bd68f`. This is a structural
refactor, with no new recommendation-quality, extraction-quality or letter-quality
claim. The feature branch is `feature/runtime-cleanup`.

## Executed checks

| Check | Result |
|---|---|
| Runtime suite in a clean environment without the legacy distribution | **236 passed**, 14.79 seconds |
| Archived regression suite, including cached E5/cross-encoder checks | **385 passed**, 50.14 seconds |
| Ten captured pre-refactor compact/projected cases | Identical feature values, feature identities, qualification evidence and constraints |
| Existing SQLite feedback database opened by old and new code using actual E5 | Identical recommendations and preference scores; rejected vacancy remains suppressed |
| Existing saved MLP checkpoint loaded by old and archived CLI using actual E5 | Identical prediction JSON |
| Runtime and legacy source/wheel builds | Successful |
| Extracted runtime wheel exercised independently of source checkout | Exactly eight product packages; all imports and recommendation/feedback smoke pass |
| Clean runtime environment dependency checks | No `resumeanalysis_legacy`, PyArrow or format-enforcer installed |
| Retained wheel licenses/notices | 120 files, including all original notices |
| Ruff lint/format, whitespace and shell syntax checks | Passed |

There are **621 passing tests** across the two suites. Archived tests were moved
and their imports/paths updated; runtime behavior checks were added. The clean
runtime environment prevents accidental reliance on the archived installation.
No personal documents were used: compatibility examples and prior demonstration
state are authored fixtures.

The saved-feedback comparison used independent SQLite backups of the existing
demo database. The original database was not migrated or modified. The old source
ran from a temporary export of commit `52bd68f`; the new source used the same
local model and dependencies. Random recommendation IDs were excluded from the
comparison; every other returned field was equal.

## Reproduce

```bash
uv sync --frozen --extra dev --extra qwen --extra profile-llm --extra documents
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RESUMEANALYSIS_RUN_E5_TESTS=1 .venv/bin/pytest -q tests
uv build
.venv/bin/python scripts/check_runtime_wheel.py dist/resumeanalysis-*.whl

uv sync --frozen --group research --extra dev --extra qwen --extra profile-llm --extra documents
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RESUMEANALYSIS_RUN_E5_TESTS=1 RESUMEANALYSIS_RUN_RERANK_TESTS=1 \
  .venv/bin/pytest -q legacy/tests
uv build --package resumeanalysis-legacy
```

The explicitly enabled model tests require the corresponding previously downloaded
local weights. Standard CI skips those optional model-dependent checks and runs
both suites on Python 3.11 and 3.12. Runtime tests/build run before installing the
research group. No model downloads or fitting occur during normal installation.

The [machine-readable record](../evaluation/runtime_refactor/validation.json) retains
comparison hashes, source hashes and the 49-file preservation inventory.

## Preservation and compatibility

Saved checkpoints, E5/Qwen/cross-encoder model files, historical evaluation data,
original licenses and research configurations were checked against a SHA-256
snapshot. Research configurations moved under `legacy/configs`; the original
cover-letter configuration was retained there while its runtime copy removes the
unused classification retry option. Original third-party notice bytes remain
intact, followed by a short note about package separation. No third-party package
version changed in the lockfile.

The exact source-set compatibility map preserves existing extractor/feature
identities only for the reviewed refactor. A test modifies a copied source file
and verifies that it no longer receives the historical identity. This does not
disable normal configuration/version checks or permit arbitrary model reuse.

See the [architecture and migration notes](runtime_refactor.md) and
[archived command mapping](../legacy/README.md). Old scoring and transport imports
move intentionally in version 0.2.0; they are not silently redirected by the
runtime. Existing checkpoints remain usable with the optional legacy package.

## Remaining limitations

The existing synthetic recommendation results and cover-letter factuality limits
remain unchanged. No real-user quality improvement, probability calibration,
live Ollama extraction or deployed ApplyPilot integration is claimed. The new
letter command drafts for an explicitly selected vacancy and confirmed profile;
it does not run a legacy qualification classifier or send applications/messages.
