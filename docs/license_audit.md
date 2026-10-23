# Open-source license compliance audit

Audit date: 2026-10-22. Baseline: `c66b74966c7734f984a309aad36b726dcaa38eef`.
This review covers the tracked source, tests, documentation, examples, CI/setup files,
`uv.lock`, retained local model/data artifacts, and the pinned reference checkouts.
It distinguishes publishing this source repository from distributing installed wheels,
an application bundle, a cloud environment snapshot, model weights, or datasets.

## Conclusion

Keep the existing **MIT LICENSE for original ResumeAnalysis code**. It is not an
umbrella license for third-party software, ESCO material, model assets, or datasets.
Using dependencies under their existing licenses does not by itself require changing
the application's license to GPL, LGPL, MPL, or Apache. Conversely, an MIT file alone
does not satisfy their notice, source-availability, patent, or redistribution terms.

No definite incompatibility with the project's own MIT source license was established.
**This is not a clean bill of health for redistributing the entire installed runtime or
the trained/pretrained weights.** The unresolved items below require human review.

## Scope and evidence

* All 48 external package/version records (44 distinct package names) in `uv.lock`
  are indexed in [dependency_licenses.json](dependency_licenses.json). Runtime,
  optional data, development, build, transitive and platform-conditional dependencies
  are included. This is not just a list of the six direct runtime dependencies.
* Installed wheel METADATA and original license/NOTICE/COPYING/AUTHORS files were read.
  For noninstalled variants, an exact locked wheel was downloaded, its SHA256 checked
  against `uv.lock`, and its notices extracted without installation. The selected wheel
  is recorded; not every wheel for every architecture was audited.
* Original text is preserved byte-for-byte between clearly marked file boundaries in
  `LICENSES/dependencies/`. Hashes identify each original file and the resulting bundle.
  License identifiers in metadata are evidence, not a substitute for bundled notices.
* Upstream reference LICENSE/NOTICE files, tool licenses, supporting legal texts,
  official-policy URLs and pinned model/dataset metadata are indexed in
  [license_evidence.json](license_evidence.json).
* Python token comparison found no contiguous match of at least 35 noncomment tokens
  against the inspected ResuMatch/Nesta/SkillAlign Python sources. Imports, tree layout
  and implementation were also inspected. This screening is **not** a legal definition
  of substantial similarity or proof of clean-room authorship.
* No third-party implementation directories, dataset parquet files, downloaded
  weights or embedding caches are tracked. Evaluation JSON does contain short
  aggregated SkillSpan span phrases; those are accounted for below.

## Code provenance and taxonomy

| Component | Actual use | License evidence | Obligations / finding |
|---|---|---|---|
| ResuMatch, `6eb8d49b6a5de28222d538f5c9350c639caa380b` | Architecture/reference for matching, features, ranking and factual explanations; no vendored implementation found | Root MIT LICENSE, copyright (c) 2026 vdenterprises | Preserve copyright and permission notice if copying/adapting protected code. Full original notice included conservatively. Repository owner is not substituted for the named copyright holder. |
| Nesta OJD DAPS Skills, `e73c2b5045793d891678ad387b2c7fda1756d236` | Extraction/normalization/mapping reference; four ESCO concept IDs checked against its formatted table | Root MIT LICENSE, copyright (c) 2024, Nesta | Full MIT notice preserved. Root code license does not establish rights in every underlying taxonomy or externally downloaded model. Nesta spaCy/pickle models are not installed. |
| SkillAlign, `cf0681df995838b14823825f0a44ec18c3903108` | ESCO/skill graph/gap-analysis reference; no vendored implementation found | Root MIT LICENSE, copyright (c) 2025 YASSER; separate NOTICE expressly limits MIT to code | Both original LICENSE and NOTICE preserved. ESCO is not relicensed to MIT. Neo4j, FAISS, frontend and mpnet model are not included as project dependencies. |
| ESCO v1.1.1 | Four concept URIs in `normalization/taxonomy.py`, associated with Python, SQL, PostgreSQL and MySQL | Nesta formatted table; SkillAlign NOTICE asserts CC BY 4.0; official Commission legal notice confirms a general CC BY 4.0 default for EU-owned website content, subject to exceptions | Credit European Commission/European Union, link source and terms, state changes, retain supplied notices and do not imply endorsement. Version-specific ESCO export terms and database rights are not fully verified; see R3. |
| Original local aliases and weighted relations | Independently authored dictionary and directional support weights | Project history/source | Project MIT. They are not official ESCO equivalences or relationships. |

## Models and datasets actually used

| Asset | Observed terms and evidence | Status / required treatment |
|---|---|---|
| `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Pinned model card declares MIT; **no separate LICENSE/NOTICE file or explicit copyright notice found** | Frozen local weights and tokenizer, not Git. Card is retained locally. MIT declaration is not an absence of a license, but the missing asset-specific notice and upstream provenance require review before redistributing weights/tokenizer. Do not invent a Microsoft/intfloat copyright line. |
| `michaelozon/candidate-matching-synthetic`, revision `178ab864dcad9910c5670d43e4bdbbb901a11f18` | Pinned card declares MIT and links MIT; **no separate LICENSE/NOTICE or copyright notice found** | Downloaded for head training/evaluation; raw/processed records excluded from Git. Retain card, author/source attribution, revision, transformations and license declaration. Confirm authority and notice wording before redistributing data or making commercial rights assurances. |
| SkillSpan, commit `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b` | README releases code/data, repository MIT LICENSE, copyright (c) 2023 Mike Zhang; no separate data license found | External evaluation dataset. Aggregate metrics and short missed-span phrases are committed; full original MIT notice and requested paper citation included. The README requests scholarly citation; that is distinguished from MIT's mandatory copyright/permission notice. Upstream rights in the original job advertisements remain a provenance question. |
| Locally trained MLP and Logistic Regression weights | Original training code; trained on the synthetic dataset using frozen E5 representations | Untracked local checkpoints. Source-code MIT does not automatically resolve licensing, ownership or data/model provenance for these artifacts; record a separate release decision before shipping them. |

## Unused or indirect upstream assets

* `med2425/resume-job-fit-merged-v1`, pinned revision
  `f46f355acddab1368a7386cbe94d42fd42767d88`: no license in card/metadata and no separate
  license file. Its named source datasets `cnamuangtoun/resume-job-description-fit`
  and `kens1ang/resume-job-fit-augmented` also declare no license in the inspected
  metadata. Payloads are **not downloaded or used**. The adapter checks that a supplied
  manifest contains fields; it is not a legal validator or an upstream permission grant.
* `nestauk/en_skillner` and `nestauk/multiskill-classifier`: model cards declare MIT;
  metadata/reference only, not installed or redistributed. Their terms must be revisited
  if a future task enables Nesta NER.
* `sentence-transformers/all-MiniLM-L6-v2` and `all-mpnet-base-v2`: cards declare
  Apache-2.0; reference/provenance only, not local encoders or dependencies.
* E5 initializes from `microsoft/Multilingual-MiniLM-L12-H384` (MIT card, no separate
  license file found). The source dataset was generated upstream with
  `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0; actual upstream LICENSE retrieved).
  Neither model is separately downloaded or run here. A generator's Apache license
  is not automatically the license of its generated output.
* E5's model card names upstream pretraining sources mC4, CC News, NLLB, Wikipedia,
  Reddit, S2ORC, Stack Exchange, xP3 and miscellaneous SBERT data, and supervised
  sources MS MARCO, NQ, TriviaQA, NLI/SimCSE, ELI5, DuReader, KILT Fever/HotpotQA,
  SQuAD, Quora, Mr. TyDi and MIRACL. These corpora are **not downloaded or used as
  ResumeAnalysis training records**. Metadata spot checks are in the evidence index:
  mC4 declares ODC-BY, xP3 Apache-2.0, SQuAD CC-BY-SA-4.0; several others declare no
  license or `unknown`. This is not an audit of all original publishers' rights,
  and it does not establish whether any data terms transfer to pretrained weights.

## Dependency and runtime findings

Full per-version terms and original notices are linked from `THIRD_PARTY_NOTICES.md`.
Notable conditions that an MIT-only reading misses:

* **Apache-2.0**: transformers, huggingface-hub, tokenizers, safetensors, hf-xet,
  requests, Arrow core and bundled components. Redistributed copies need the license,
  relevant copyright/attribution and supplied NOTICE content; modified files need
  modification notices. Patent grant/termination and trademark exclusions remain.
* **MPL-2.0**: certifi and parts of tqdm. File-level conditions and source-availability
  rules can apply on redistribution; this does not automatically relicense the whole
  application. Original notices and the complete MPL text are included.
* **LGPL-2.1-or-later / GCC runtime exception**: inspected NumPy and SciPy Linux wheels
  actually include `libquadmath` and `libgfortran`; their composite license files
  identify LGPL libquadmath and GPL-3.0-or-later WITH GCC-exception-3.1 runtime code.
  Scikit-learn's COPYING identifies bundled `libgomp` and the exception. PyTorch also
  includes `torch/lib/libgomp.so.1`; its exact runtime provenance needs verification.
  Merely copying a BSD header or an SPDX field is insufficient for these binaries.
* **Composite binary notices**: NumPy's source expression includes BSD-3-Clause,
  0BSD, MIT, Zlib and CC0; PyTorch has BSD core notices and an aggregate expression
  including Apache/LLVM-exception, BSD, BSL and MIT. Arrow's full LICENSE/NOTICE also
  identifies BSD/MIT/Boost and other included works. All shipped notices were preserved,
  including notices for code not necessarily active in the installed CPU build.
* **regex** declares `Apache-2.0 AND CNRI-Python`; its file credits Secret Labs AB and
  states derivation from CPython's `re`. The complete CNRI-Python text is supplied in
  addition to the original regex notice. Treating this package as Apache-only is wrong.
* **packaging** uses `Apache-2.0 OR BSD-2-Clause` (a choice, not an AND requirement).
  **typing-extensions** declares PSF-2.0. Original multi-part text is retained.
* The installed **tokenizers 0.22.2 wheel has no license file**. Its exact GitHub
  `v0.22.2` tag resolves to `f383101a26663708484cac0727792aad74f78234`; the Apache-2.0
  LICENSE was recovered from that source. This repairs the local notice collection,
  not a complete audit of every statically linked Rust crate.
* Setuptools ships vendored autocommand, backports.tarfile, importlib_metadata,
  jaraco.text/context/functools, more-itertools, packaging, platformdirs, tomli,
  wheel and zipp. Original vendor license files are retained in its notice bundle.
  All nested file paths are indexed; vendor versions are not separate uv.lock roots.
* `actions/checkout@v4` is MIT, copyright (c) 2018 GitHub, Inc. and contributors;
  `astral-sh/setup-uv@v6` is MIT, copyright (c) 2024 Kevin Stillhammer. Actual resolved
  tag commits and notices are retained. External `uv 0.12.23` is MIT OR Apache-2.0;
  both notices are retained. These tools execute externally, not as vendored app code.
* Python 3.12.14 and the runner/OS are external platform components, not covered by
  the application MIT file. Python uses the PSF license agreement and historical
  notices. A distributable interpreter/container requires its own platform inventory.

## Changes and subsequent language cleanup

1. Expanded `THIRD_PARTY_NOTICES.md`; preserved the original named copyright holders.
2. Added original upstream/dependency license and NOTICE text under `LICENSES/`,
   plus evidence hashes and the complete locked dependency index.
3. Added PEP 639 `license-files` packaging configuration so notices ship in wheel/sdist;
   raised the build-backend minimum to setuptools 77, which supports those fields.
   Runtime dependencies, the lockfile, scoring/training logic, weights and data are unchanged.
   The user separately requested English documentation and messages; only explanation
   strings and their language assertion were translated in application/tests.
4. Kept the project's root `LICENSE` byte-for-byte unchanged.

## Verification of this update

* All 47 tests passed, including the downloaded pretrained E5 integration test.
* Saved-weight evaluation exactly reproduced the committed test metrics. English
  explanations were checked through offline CLI inference.
* Application and test edits for translation differ only in string literals;
  scoring/training/extraction code and checkpoint identities remain unchanged.
* Wheel and sdist were built and checked to contain all 66 collected license/notice
  files byte-for-byte, including `THIRD_PARTY_NOTICES.md` and original copyright text.
* The root MIT LICENSE and `uv.lock` are unchanged. Lint, formatting, lockfile checks
  and whitespace checks passed. Project Markdown and output messages are English;
  Russian is retained only as extraction vocabulary/patterns and multilingual inputs.

## Manual-review register

| ID | Issue | Required human action / scope |
|---|---|---|
| R1 — blocked data | med2425 and its two named inputs have no verified license grant | Keep payload/training disabled until licenses, authority and compatibility for all merged inputs are documented. Public availability and the adapter's manifest fields do not confer rights. |
| R2 — asset-specific MIT evidence | E5/MiniLM and Candidate Matching cards declare MIT but do not supply separate notices; training provenance includes third-party materials | Before redistributing pretrained weights, tokenizers, data, or a commercially released trained head, confirm scope, copyright/notice wording and authority with upstream or counsel. Do not assume a code-repository license resolves training-data rights. |
| R3 — taxonomy scope | General Commission CC BY 4.0 policy verified, ESCO v1.1.1 export-specific terms unavailable (ESCO site returned HTTP 403) | Confirm the exact export's rights and any database/third-party exclusions, especially before expanding or redistributing the taxonomy. Retain current conservative EU/Commission attribution, changes and no-endorsement notice. No conclusion is made about protectability of four isolated URIs. |
| R4 — bundled binaries | LGPL libquadmath, GCC runtime/exception, MPL files and other composite libraries | Before distributing an app image/environment snapshot/binaries, verify actual platform builds, source availability/offers where required, LGPL replacement/relinking conditions, GCC exception eligibility and recipients' notices. Full notices alone do not discharge every binary-distribution obligation. |
| R5 — incomplete native/platform provenance | tokenizers wheel omits a license file; compiled Rust/native dependency graphs and all architecture variants not exhaustively audited | Reconcile the exact shipped binaries with upstream notices/SBOMs, recover missing attributions and corresponding sources where required. This audit does not certify every crate/native subcomponent or future version selected by unfrozen pip. |
| R6 — data ownership | SkillSpan repository MIT and synthetic-data MIT declarations cannot establish rights in every upstream advertisement/output | Confirm underlying rights for intended commercial use or redistribution; retain SkillSpan MIT/citation and provenance. Short aggregate span excerpts are covered by the conservative notice collection. |
| R7 — release scope | Source-only Git distribution differs from shipping retained `.venv`, models, caches and reference clones | Define the artifact actually released; include its complete notices and source obligations. Avoid claiming that the whole cloud snapshot is MIT merely because the root repository is MIT. |

No legal clearance, rightsholder warranty, full patent/trademark assessment, or analysis
of jurisdiction-specific AI training/output rights is established by this technical audit.

## Optional Qwen extension — 2026-10-23

This addendum covers the feature-branch extension from
`f682c2427bdb2e1ea7ccb8a778b5d92caaa4c2f8`. The original audit and manual-review
register above remain historical findings; the lockfile now adds five optional
package/version entries without changing any of the original 48 versions.
[Qwen evidence](qwen_license_evidence.json) records actual license files, hashes,
model revision and inspected software artifacts. The current complete dependency
index has 53 external package/version entries; its baseline hash is retained too.

Qwen2.5-7B-Instruct supplies a real Apache-2.0 LICENSE at the pinned revision,
retained in the notice collection. Its weights are not committed or executed here.
No new source code, dataset or taxonomy was copied from ResuMatch, Nesta or
SkillAlign. The optional model and dependencies do not require changing the root
MIT for original code; they do require preserving their own terms and notices.
The notices for accelerate (Apache-2.0), bitsandbytes, lm-format-enforcer and
interegular (MIT), and psutil (BSD-3-Clause) are retained without changing originals.

Additional manual review: the GPU installation deliberately lives outside the
CPU lock. Check the exact CUDA-enabled PyTorch, NVIDIA libraries/driver and compiled
bitsandbytes components before distributing an installed GPU environment. No
platform-complete binary SBOM or upstream model training-data/output clearance is
claimed. Existing E5/data/ESCO and native-library risks R1–R7 still apply. The new
adapter's snapshots/diagnostics also require the custodian's dataset license and
privacy retention policy; conversion success is not a grant of data rights.
