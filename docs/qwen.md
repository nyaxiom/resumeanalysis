# Optional Qwen Instruct backend

The original frozen E5 + MLP predictor, training scripts, feature definitions and
checkpoint format remain intact. `resumeanalysis predict` still selects MLP by
default. Qwen is a separate prompting-only backend: no fine-tuning, reinforcement
learning, cloud inference API or mandatory application service is introduced.

The default model is **Qwen/Qwen2.5-7B-Instruct**, pinned to
`a09a35458c702b33eeacc393d103063234e8bc28`. It provides multilingual instruction
following, a supported Transformers architecture and an actual Apache-2.0 LICENSE.
Its suitability for matching and its Russian accuracy require independent evaluation.
There are no measured Qwen quality results or claims of improvement over E5 here.

## Installation and reference deployment

The reference target requested for this backend is an NVIDIA RTX 3060 with 12 GB
VRAM, using batch size one, NF4 4-bit weights, double quantization and FP16 compute.
Context length, generation length, CUDA libraries and allocator overhead affect
memory; fitting this configuration in 12 GB has **not** been verified on that GPU.
Reduce the context budget if measurement requires it. There is no silent CPU fallback.

The existing `uv.lock` and `uv sync` intentionally continue to select CPU PyTorch.
For existing development plus optional prompt/grammar support:

```bash
uv sync --frozen --extra dev --extra data --extra qwen
```

Use a separate GPU environment so that the original CPU workflow remains reproducible:

```bash
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch wheel appropriate for the driver from pytorch.org.
# Example CUDA 12.8 index; check current PyTorch compatibility before selecting a version.
.venv-qwen/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-qwen/bin/python -m pip install -e '.[qwen-cuda]'
.venv-qwen/bin/python -c 'import torch; assert torch.cuda.is_available()'
```

This pip CUDA environment is separate from the CPU lock and has **not** been
resolved or tested here. Save its exact package versions, CUDA/driver versions and
additional binary notices for any reproducible GPU benchmark or distribution.
The optional dependencies are accelerate, lm-format-enforcer and bitsandbytes;
their original notices and new transitive notices are retained in the license index.

## Download and inference

Downloads are explicit, pinned and separate from inference. Start with an empty
destination. The downloader retains the model card, LICENSE and any supplied NOTICE,
checks LFS hashes including every safetensors shard, and writes a local manifest.
Inference verifies that manifest's identity and checksums, loads local safetensors
with remote code disabled, and never downloads missing files.

```bash
.venv-qwen/bin/resumeanalysis-qwen download --config configs/qwen.json
.venv-qwen/bin/resumeanalysis predict --backend qwen \
  --qwen-config configs/qwen.json --input examples/lightweight_profile.json
# Equivalent independent entry point:
.venv-qwen/bin/resumeanalysis-qwen predict \
  --config configs/qwen.json --input examples/lightweight_profile.json
```

Model ID, exact revision, local path, quantization (`bnb4`/`none`), device, dtype,
token budgets, threshold and validation retries are in `configs/qwen.json`.
For an explicitly selected CPU smoke model, Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775` has Apache-2.0 terms; set a separate model
directory, `device=cpu`, `quantization=none`, and `dtype=float32`. No pretrained
Qwen model, including this smaller option, was executed during this implementation.
Model substitutions require model/license review; they are not automatic fallbacks.

Generation is greedy, frozen and bounded. The seed is recorded in the configuration;
greedy generation does not consume it. Bitwise equality across CUDA builds is not
promised. The full rendered prompt must fit the configured input/context budget;
oversized input fails rather than silently truncating a resume. Invalid JSON,
duplicate keys, nonfinite values, missing/extra fields, ungrounded skills and
nonliteral citations fail validation. A bounded retry asks for a corrected response;
no heuristic result is substituted after failure.

## Profiles and output semantics

```python
from resumeanalysis.qwen.predictor import QwenPredictor
from resumeanalysis.qwen.config import QwenConfig

model = QwenPredictor(QwenConfig())
analysis = model.predict(
    {"skills": ["Python", "SQL"], "experience_years": 2, "preferred_roles": ["ML Engineer"]},
    {"title": "ML Engineer", "required_skills": ["Python", "Docker"]},
)
```

Qwen accepts the explicit `MatchingProfile(Profile)` extension with preferred roles.
The existing `Profile` and canonical `PairRecord` schemas are unchanged; the existing
dictionary-based MLP API still rejects this extra field. A preferred role expresses
intent, not a qualification. Omitted experience is unknown; empty skills are not
proof of absence. An omitted `remote_only` uses the existing schema's default of no
remote constraint. Set it only when that preference is known.

The model judges semantic relevance from full supplied input and returns a validated
decision, an **uncalibrated score**, skill lists and optional literal input citations.
To prevent fabricated skill facts, the three skill lists must exactly match a
deterministic catalog built from explicit skill lists and the existing RU/EN
extractor. Unclassified vacancy mentions do not become mandatory requirements;
provide `required_skills`/`optional_skills` when known. This limits open-ended skill
reporting to the existing dictionary plus explicitly supplied terms.

`missing_skills` combines unconfirmed and explicitly denied skills for public API
compatibility; `unknown_skills` and `contradicted_skills` retain the distinction.
Conflicting candidate evidence is unknown. Experience gaps use explicit numeric
fields only. Explanations are rendered from validated facts and optional quoted
input, not an unrestricted generated account of the candidate's career. Quoted
input is not independently verified and can itself be misleading. This validates
output grounding, not the truth of source claims or correctness of semantic scoring.
Remote/location constraints use the existing conservative postprocessing: unknown
required preferences can block the final boolean while retaining the model score.
Empty qualification evidence cannot yield a positive assessment.

`VacancyAnalysis` retains its future `summary`, `resume_id` and `cover_letter` slots
as `None`. Feedback, resume selection, summaries and letters can be added behind
the backend/application boundaries after separate evaluation and requirements.

## ApplyPilot boundary

```python
from resumeanalysis.integration.applypilot import LocalInferenceClient
from resumeanalysis.inference.backends import MatchingProfile


class Profiles:
    def get_profile(self):
        return MatchingProfile(skills=["Python"], preferred_roles=["ML Engineer"])


client = LocalInferenceClient(model, Profiles())
# analysis = await client.process(vacancy)
```

`InferenceClient.process(vacancy)` is the async application protocol. The local
provider supplies user context; the model-level API always takes both inputs.
Inference runs in a worker thread, with local Qwen generation serialized. There
are no imports from unavailable ApplyPilot modules and no Kafka requirement.
E5 ignores `preferred_roles` if given a subclass instance; use a base `Profile` for
E5 and do not expect role preferences to affect its score.

## Partial-profile experiments

```bash
resumeanalysis-qwen augment --data data/processed \
  --output data/partial-experiment --ratios 1 .75 .5 .25 --seed 42
```

Original partitions are checked for leakage **before** augmentation. Only separate
train/validation observation files are written. The official test set is never
augmented on disk. Derived observations record the original hash, entity IDs,
provenance, partition, seed, parameters, requested ratio and realized retention.
They use a separate `PartialObservation` schema with `label=None`; the original
label is explicitly named source metadata and cannot be fed to canonical training.

Skill dropout uses explicit skills, not re-extracted resume vocabulary. The same
candidate/profile gets the same mask across vacancies. Ratios use nested subsets
with integer rounding. For partial views, raw resume text, projects and education
are removed to stop deleted skills reappearing through text extraction. Optional
field dropout also masks experience, seniority and preferences; explicit denials
remain explicit. Ratios count skill entries and populated profile fields, not
semantic information. Requested and realized ratios differ, especially for sparse
profiles and forced text masking. Retained negative evidence is never invented.

## Comparison

```bash
# Both backends in separate fresh processes, identical ordered source records/views:
.venv-qwen/bin/resumeanalysis-qwen compare --data data/processed \
  --qwen-config configs/qwen.json --output artifacts/comparison/report.json
# Independently test one backend; --limit is a diagnostic subset, not a benchmark:
resumeanalysis-qwen measure --backend mlp --data data/processed \
  --limit 8 --output artifacts/comparison/mlp-diagnostic.json
```

Both backends consume canonical normalized records. The shared framework checks
all source partitions and hashes, creates the same four views for both models,
and verifies observation IDs before combining results. Whole-source-view metrics
include accuracy, precision, recall, macro-F1 and a fixed `[0,1]` confusion matrix.
Metrics use the final boolean, including preference gating. ROC-AUC uses raw model
scores as rankings when two source classes exist; it does not imply calibration.

The 75%, 50% and 25% views report decision changes, scores, actual retention,
latency and measurable memory, with **no supervised metrics**. Independently reviewed
labels and their review provenance would need a separate extension before these
views could be used as supervised evaluations. Existing synthetic/heuristic labels
are never a basis for claims about real recruitment accuracy.

Latency excludes model initialization and configurable warmup (reported separately).
CUDA calls synchronize before timing; peak CUDA allocated/reserved bytes are measured
per view where available. CPU peak RSS is a process-lifetime high-water value and
is unavailable on platforms without `resource`. It includes initialization, warmup
and interpreter overhead. Separate subprocesses avoid cross-model contamination;
the reported CUDA allocator values do not include all driver/system allocations.
There is no measured GPU Qwen latency, VRAM peak, comparison or hiring-quality claim.

See [dataset adapters](dataset_adapters.md) for strict source normalization and
[third-party notices](../THIRD_PARTY_NOTICES.md) for asset terms. Apache-2.0 for
Qwen weights does not clear all training-data/output rights. RL and fine-tuning
are deferred until licensed independent relevance labels, a valid evaluation
design and an explicit training objective exist.
