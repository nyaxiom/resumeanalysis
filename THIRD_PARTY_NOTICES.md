# Third-party notices

ResumeAnalysis's original source remains MIT; see [LICENSE](LICENSE).
This does not replace the terms of third-party source, libraries, or model weights.
The minimal service uses the components below. No dataset, skill taxonomy, or
pretrained model weights are distributed in this repository or its wheel.

## Reused third-party source

The JSON input-boundary helper and input-boundary rules in
[`prompts.py`](src/resumeanalysis/prompts.py) are adapted from
`wihlarkop/applykit`, commit `002360a6d5af14f9eeedae29934291fc95b22f97`,
`backend/app/services/prompts.py`.

**Copyright (c) 2026 ApplyKit Contributors.** Licensed under MIT.
The original copyright and complete permission/disclaimer text are retained in
[LICENSES/cover_letters/applykit-LICENSE.txt](LICENSES/cover_letters/applykit-LICENSE.txt).
The source header remains. This version keeps only the input encoding helper and
boundary rules; the new resume-only cover-letter prompt is authored for this service.
No ApplyKit web application, database, or provider abstraction is included.

E5 pooling/chunking and Qwen inference were adapted from this repository's existing
MIT implementation. River, Transformers, FastAPI, and other libraries are imported
as dependencies, not copied into the source tree.

## Models

| Model | Pinned revision | Declared license and notices |
|---|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Model card declares MIT; see limitation below |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) | `cdbee75f17c01a7cc42f958dc650907174af0554` | Apache-2.0; [original license](LICENSES/cover_letters/qwen-4b-LICENSE.txt) |

The earlier audit found no standalone E5 LICENSE/NOTICE or asset-specific copyright
notice at the pinned revision. Confirm the appropriate notice and tokenizer/model
rights before redistributing those assets; the application's MIT license does not
supply missing upstream attribution. No rights holder is inferred from a username.
The service downloads weights separately or loads operator-supplied local snapshots.
Changing model IDs requires reviewing the replacement model's own terms. Keep model
cards, licenses, and any upstream NOTICE files with redistributed model assets.
No fine-tuned weights or dataset-derived artifacts are shipped by this service.

## Libraries

Exact versions, platform conditions, artifact hashes, and transitive dependencies
are recorded in [uv.lock](uv.lock). Existing complete upstream notice bundles remain
unchanged in `LICENSES/dependencies/` and `LICENSES/online_recommender/`.
New HTTP-stack notices, copied byte-for-byte from installed distribution metadata,
are in [LICENSES/service/](LICENSES/service/) with a provenance/hash inventory.

| Direct runtime dependency | License summary | Notice location |
|---|---|---|
| FastAPI | MIT | `LICENSES/service/fastapi-0.165.0-LICENSE.txt` |
| Uvicorn | BSD-3-Clause | `LICENSES/service/uvicorn-0.54.0-LICENSE.md.txt` |
| Pydantic | MIT | `LICENSES/dependencies/pydantic-2.14.1.txt` |
| NumPy | BSD-3-Clause and bundled component terms | `LICENSES/dependencies/numpy-*.txt` |
| PyTorch | BSD-style and bundled component terms | `LICENSES/dependencies/torch-*.txt` |
| Transformers | Apache-2.0 | `LICENSES/dependencies/transformers-4.57.6.txt` |
| River | BSD-3-Clause | `LICENSES/online_recommender/river-river-0.26.1.dist-info_licenses_LICENSE.txt` |

Starlette, httpx/httpcore (tests), and OpenTelemetry API (a FastAPI transitive
dependency) have complete notices in `LICENSES/service/`. The latter does not add
an exporter or monitoring service. Development tools also retain their original
notices. License labels are summaries, not substitutes for the full texts.

MIT/BSD attribution and disclaimers, Apache license/NOTICE and modification terms,
and applicable bundled-library terms remain required when redistributing software.
Some transitive packages include MPL/LGPL/GPL-with-exception components; review the
actual binaries and distribution method when shipping a complete environment.
CUDA builds require their own binary/license review. Dependencies installed
separately still retain their upstream notices in their distributions.

## Preserved historical notices

The previous notice document is preserved byte-for-byte as
[LICENSES/history/THIRD_PARTY_NOTICES.txt](LICENSES/history/THIRD_PARTY_NOTICES.txt).
Its descriptions, dependency inventory, and old documentation references describe
previous releases, not the current architecture. All pre-existing `LICENSES/`
files remain unchanged, including ResuMatch, Nesta, SkillAlign, SkillSpan, ESCO,
and removed experiment/model dependency attributions. Keeping these records does
not mean those components are still used by the minimal service. Original notices
have not been replaced with project copyright claims.

The root MIT license remains appropriate for original project code. It does not
clear third-party model/data rights or supersede their notice obligations.
