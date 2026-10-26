# Qwen feature implementation and validation

Development branch: `feature/qwen-instruct`, created from the then-current default
branch `main` at `f682c2427bdb2e1ea7ccb8a778b5d92caaa4c2f8`. No changes were made or
merged into `main`. The feature commit and review link are supplied in the completion
message; this document records the implementation rather than a self-referential SHA.

## Implemented safeguards

- Separate prompting-only Qwen backend, configurable local runtime and backend factory.
  The pinned default is Qwen2.5-7B-Instruct, chosen for RU/EN instruction following,
  supported local Transformers inference and verified Apache-2.0 terms. NF4 4-bit
  inference is configured for the requested 12 GB reference deployment; it is unverified.
- Strict JSON grammar where enabled, Pydantic validation, duplicate/nonfinite rejection,
  bounded retries, exact evidence citations and grounded skill lists. Unknown and
  explicitly denied skills stay distinct; scores are uncalibrated. No invented fallback
  qualifications, implicit model downloads, fine-tuning or reinforcement learning.
- Explicit preferred-role profile extension, preserving canonical `Profile`/`PairRecord`.
  Async ApplyPilot protocol with an injected local profile provider and independent backend.
- Canonical and explicit mapping JSONL adapters: default strict mode, nested coverage,
  typed label mappings, schema-version checks and per-row outcomes. Permissive omission
  and rejection are explicit, reported and separate from provenance/leakage enforcement.
- Unchanged source bytes or explicit metadata-only retention, deterministic row IDs,
  dataset/revision/provenance, original and normalized hashes/indexes, optional local
  snapshots, manifest integrity checks and human/machine-readable conversion reports.
- Count reconciliation, source field/null/missing/type statistics, before/after
  distributions, exact duplicates, configurable bounded near-duplicate diagnostics,
  mandatory cross-partition entity/content checks and dataset-specific quality policies.
  Unexpected integer/float normalization and numeric precision loss cannot pass unnoticed.
  Source file exclusions, empty files and changes during conversion are visible.
- Partition checks before augmentation; separate unlabeled observations carry source
  lineage and realized retention. No automatic label inheritance or augmented official
  test file. Shared evaluation scores only original labeled views; partial views report
  diagnostics. Backends must receive identical ordered observations before comparison.
- Original notices are retained. Six new original license files cover the default
  Qwen model and five optional packages. The root MIT remains unchanged.

## Verification actually executed

Date: 2026-10-23. Python 3.12, CPU PyTorch; CUDA unavailable.

| Check | Observed result |
|---|---|
| Full tests with `RESUMEANALYSIS_RUN_E5_TESTS=1` | **106 passed**, including the actual downloaded E5 integration |
| Local tiny Qwen Transformers integration | Random initialized test fixture: safetensors loading, frozen parameters, generation, real optional JSON grammar, context refusal and checksum failure verified; not pretrained quality |
| E5 saved-weight evaluation | All metrics exactly equal the committed results on all **148** held-out records |
| Original implementation preservation | Predictor, feature builder, MLP, training/evaluation modules, CPU config and saved MLP checkpoint hashes unchanged |
| Lock preservation | All original **48** external package/version/source records retained; five optional additions |
| Dataset CLI strict conversion and audit | **994** input records, **994** converted, zero rejected/excluded, counts reconcile, 100% observed field disposition coverage, leakage check passed |
| Source partitions | 696 train, 150 validation, 148 test; no automatic repartitioning |
| Augmentation CLI | 2,784 train and 600 validation observations; every label is `None`; official test unchanged |
| Shared evaluation path with real saved MLP | Eight-record diagnostic subset, four retention views; no partial-view supervised metrics |
| Both normalized consumer paths | Synthetic integration test converts data, trains/reloads existing lexical-smoke MLP and evaluates it alongside mocked Qwen on identical normalized examples |
| Lint, format, lock, whitespace | Passed |
| Wheel and sdist | Built; all **72** collected license/notice files included byte-for-byte |

Runtime reports, normalized records, partial observations and model fixtures remain
outside Git. No real Qwen benchmark metrics were generated. CI installs the optional
non-CUDA runtime and runs the suite on Python 3.11/3.12; downloaded E5 coverage remains
an explicit local opt-in. Check the PR for the observed hosted CI results.

```bash
uv sync --frozen --extra dev --extra data --extra qwen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
RESUMEANALYSIS_RUN_E5_TESTS=1 uv run --frozen pytest -q
uv run --frozen resumeanalysis evaluate --data data/processed --output artifacts/run
uv run --frozen resumeanalysis datasets inspect --input data/processed \
  --dataset michaelozon/candidate-matching-synthetic \
  --revision 178ab864dcad9910c5670d43e4bdbbb901a11f18
uv build
```

The E5 evaluation commands need the existing downloaded E5, prepared dataset and
trained checkpoint, as documented in the original README. Unit tests use synthetic
fixtures and need no downloads. See [Qwen instructions](qwen.md) for CUDA installation,
model download, prediction, augmentation and isolated comparison commands, and
[dataset instructions](dataset_adapters.md) for strict/permissive conversion and audit.

## Limits and manual review

Pretrained Qwen inference, 4-bit CUDA loading, 12 GB VRAM fit, GPU latency/memory and
the separate CUDA pip environment have not been tested. Real Russian/English matching
quality and comparisons on independently labeled data remain unvalidated. The existing
fit labels are synthetic/heuristic. There is no evidence of Qwen superiority.

Adapters currently cover JSONL canonical records and explicitly configured mappings;
other formats and public-source-specific semantics need dedicated reviewed adapters.
Historical preparation commands remain unchanged and are outside the new guarantees.
Whole object-array flattening is unsupported. Broad auxiliary subtree allowlists
preserve/report fields but need schema review. Source snapshots preserve rows, not
an entire dataset directory or arbitrary sidecars. Metadata-only retention explicitly
cannot reconstruct originals. Reports/IDs can themselves be sensitive.

Exact identity/content checks do not detect every paraphrase or hidden relationship.
Near-duplicate checks are optional, configurable, bounded and lexical; reports state
whether coverage is partial. No independent labels exist for partial observations;
supporting reviewed labels would require a separate provenance-preserving extension.
Retained-field fractions count fields/skill entries, not semantic information.
Skill grounding inherits the existing extractor's limited coverage, and exact input
citations do not verify the truth of input claims or semantic judgments.

Manual license review remains necessary for model training/output rights, the exact
distributed CUDA/NVIDIA/native runtime, dataset retention/privacy rights, and the
existing E5/data/ESCO/native-library findings. Read the preserved
[audit register](license_audit.md) and [notices](../THIRD_PARTY_NOTICES.md).

## File inventory

New implementation files:

- `src/resumeanalysis/qwen/{__init__,config,prompts,runtime,predictor,download,cli}.py`
- `src/resumeanalysis/inference/backends.py`
- `src/resumeanalysis/integration/{__init__,applypilot}.py`
- `src/resumeanalysis/data/{adapters,conversion,adapter_cli}.py`
- `src/resumeanalysis/experiments/{__init__,partial_profiles,comparison}.py`
- `src/resumeanalysis/jsonio.py`
- `tests/test_qwen.py`, `tests/test_loss_aware_adapters.py`, `tests/test_partial_comparison.py`

New configuration, examples, documentation and notices:

- `configs/qwen.json`, `configs/example_adapter.json`, `configs/example_quality_policy.json`
- `examples/lightweight_profile.json`
- `docs/qwen.md`, `docs/dataset_adapters.md`, `docs/qwen_validation.md`, `docs/qwen_license_evidence.json`
- `LICENSES/references/qwen-instruct-7b-LICENSE.txt`
- `LICENSES/dependencies/{accelerate-1.15.0,bitsandbytes-0.50.2,interegular-0.3.3,lm-format-enforcer-0.11.3,psutil-7.2.2}.txt`
- `.gitattributes` preserves original notice bytes and line endings; their upstream
  whitespace is exempt from source formatting checks.

Modified: `src/resumeanalysis/cli.py`, `pyproject.toml`, `uv.lock`,
`.github/workflows/ci.yml`, `.gitignore`, `README.md`, `THIRD_PARTY_NOTICES.md`,
`docs/license_audit.md`, `docs/dependency_licenses.json`.
