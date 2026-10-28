# Third-party notices

ResumeAnalysis source is MIT, copyright 2026 ResumeAnalysis contributors.
That statement applies to the project's original code. It does not replace the
licenses of dependencies, taxonomy material, models or datasets. The root
[LICENSE](LICENSE) remains unchanged.

Audit date: 2026-10-22. Read the [compliance audit](docs/license_audit.md) for scope,
license compatibility, unresolved rights and the manual-review register. Exact
sources, revisions and hashes are in [license_evidence.json](docs/license_evidence.json)
and [dependency_licenses.json](docs/dependency_licenses.json).

The optional Qwen extension was reviewed on 2026-10-23. Its model and five additional
package notices are listed in the Qwen addendum at the end of this file and in
[qwen_license_evidence.json](docs/qwen_license_evidence.json). Original notices below
remain intact; this addendum does not clear the separate CUDA runtime for distribution.

The following original copyright and permission notices are retained. No upstream
copyright holder is replaced with the ResumeAnalysis contributors' name.

## Reference projects and source provenance

* ResuMatch — JananiV07/resumatch, MIT, copyright (c) 2026 vdenterprises.
* Nesta OJD DAPS Skills — nestauk/ojd_daps_skills, MIT, copyright (c) 2024 Nesta.
* SkillAlign — Y4SSERk/SkillAlign, MIT, copyright (c) 2025 YASSER.

These projects informed architecture, extraction/normalization, ranking and skill-gap
analysis. No vendored implementation modules or substantial verbatim Python blocks
were found in the inspected source. This is not a legal guarantee of independent
authorship. Their complete original MIT notices are reproduced below conservatively,
with SkillAlign's separate data NOTICE retained in
[LICENSES/references/skillalign-NOTICE.txt](LICENSES/references/skillalign-NOTICE.txt).
See [research](docs/research.md) for exact commits and reviewed files.

## ESCO taxonomy material

This service uses the ESCO classification of the European Commission.

Attribution: European Commission / European Union. The Commission's general website
copyright notice reads “© European Union, 1995-2026”. This is the website notice,
not an invented version-specific copyright statement for the four identifiers.

* Official source: <https://esco.ec.europa.eu/en/use-esco/download>.
* General official terms: <https://commission.europa.eu/legal-notice_en>.
* License applied conservatively to EU-owned classification material:
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  ([complete text](LICENSES/texts/CC-BY-4.0.txt)), subject to specific notices and exclusions.
* Provenance: four identifiers from the ESCO v1.1.1 formatted table distributed by
  Nesta, associated locally with Python, SQL, PostgreSQL and MySQL.
* Changes: selection of those identifiers and their association with local canonical
  names; aliases and weighted skill relations are independently authored and are
  not official ESCO relationships or equivalences.
* No endorsement by the European Union or European Commission is implied.

SkillAlign's original NOTICE describes ESCO as CC BY 4.0. The official Commission
legal notice confirms a general CC BY 4.0 default for EU-owned website content,
with credit and indication of changes, but excludes third-party rights and certain
other materials. The ESCO v1.1.1 export-specific terms were not accessible during
this audit (HTTP 403). Confirm those terms before expanding or redistributing the
taxonomy; no conclusion is made about copyright/database protection of four URIs.

## Models and training/evaluation data

### Local E5 encoder

Source: `intfloat/multilingual-e5-small`, revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`.
[Pinned model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md).

The card declares **MIT**. The download retains the card and verifies the safetensors
SHA256; weights and tokenizer are not tracked in Git. No separate LICENSE/NOTICE file
or asset-specific copyright notice was found. Do not infer or fabricate a copyright
holder from the publisher handle. Obtain/confirm the appropriate original notice and
scope before redistributing the model/tokenizer; the application's own MIT notice
is not a substitute. E5's MiniLM initialization and upstream training provenance are
recorded in the audit and do not establish that all underlying materials are MIT.

### Candidate Matching Synthetic

Source: `michaelozon/candidate-matching-synthetic`, revision
`178ab864dcad9910c5670d43e4bdbbb901a11f18`.
[Pinned dataset card](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic/blob/178ab864dcad9910c5670d43e4bdbbb901a11f18/README.md).

The card declares **MIT** and links to MIT, but no separate license file or explicit
copyright notice was found. Attribution is to the source/publisher as identified,
without inventing a rights holder. Preserve the card, revision, license declaration
and transformations, and confirm redistribution rights and notice wording with upstream.
Profiles/jobs are Qwen-generated, positives are heuristic, and this project adds
explicitly weak rule-derived hard negatives after combining structured/text evidence.
Raw/processed records are not included in Git.

The upstream generator `Qwen/Qwen2.5-0.5B-Instruct` declares Apache-2.0; its actual
[LICENSE is retained](LICENSES/references/qwen-data-generator-LICENSE.txt) for provenance.
It is not installed or run here. Its license is not automatically the generated
output's license. Locally trained MLP/LR weights also need an explicit artifact-release
rights review; the source MIT file does not resolve every model/data provenance issue.

### SkillSpan

Source: `kris927b/SkillSpan`, commit `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b`.
The README releases code and data; its root MIT LICENSE is reproduced below,
**Copyright (c) 2023 Mike Zhang**. No separate dataset license was found.
The complete corpus remains outside Git. Committed aggregate evaluation output
includes short missed-span phrases, so its source attribution is retained here too.
Underlying rights in original job advertisements still merit review before redistribution.

The README requests the following scholarly citation; MIT's mandatory notice
condition is the copyright and permission notice, separately preserved below:

Zhang, Mike; Jensen, Kristian Nørgaard; Sonniks, Sif; Plank, Barbara (2022).
**SkillSpan: Hard and Soft Skill Extraction from English Job Postings.**
NAACL-HLT, pp. 4962–4984. <https://aclanthology.org/2022.naacl-main.366>.

### Unlicensed data not used

`med2425/resume-job-fit-merged-v1` and its named inputs
`cnamuangtoun/resume-job-description-fit` and `kens1ang/resume-job-fit-augmented`
have no verified license declaration in the inspected metadata. Their payloads
are not downloaded or used. Do not treat public access or a user-written manifest
as a license grant. Obtain adequate permission for all merged sources before use.

### Other reference-only assets

Nesta's `en_skillner` and `multiskill-classifier` cards declare MIT;
`all-MiniLM-L6-v2` and `all-mpnet-base-v2` cards declare Apache-2.0. They are not
installed as project models. MiniLM/E5 upstream datasets, generator provenance and
model revisions are listed in [the audit](docs/license_audit.md) and evidence index;
no blanket license clearance is claimed for them.

## Locked software dependencies

The table covers every external package/version in `uv.lock`, including development,
optional, transitive and platform-specific records. License labels summarize observed
upstream terms; they do not replace full notices or certify every compiled subcomponent.

Every linked notice bundle contains the **original full license, copyright,
NOTICE/COPYING/AUTHORS text** supplied by the inspected artifact, with file boundaries
and hashes recorded in the inventory. Those files are part of this notice collection
and are included in wheel/sdist. Dependencies themselves are installed separately,
not vendored as source code by this repository.

| Package/version | Role | Observed license summary | Full copyright/license/NOTICE text |
|---|---|---|---|
| `annotated-types==0.8.0` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/annotated-types-0.8.0.txt) |
| `build==1.6.1` | direct development | MIT | [original notices](LICENSES/dependencies/build-1.6.1.txt) |
| `certifi==2026.10.19` | transitive (including platform-conditional dependencies) | MPL-2.0 | [original notices](LICENSES/dependencies/certifi-2026.10.19.txt) |
| `charset-normalizer==3.5.2` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/charset-normalizer-3.5.2.txt) |
| `cloudpickle==3.1.2` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/cloudpickle-3.1.2.txt) |
| `colorama==0.4.6` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/colorama-0.4.6.txt) |
| `filelock==4.2.4` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/filelock-4.2.4.txt) |
| `fsspec==2026.9.0` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/fsspec-2026.9.0.txt) |
| `hf-xet==1.7.0` | transitive (including platform-conditional dependencies) | Apache-2.0 | [original notices](LICENSES/dependencies/hf-xet-1.7.0.txt) |
| `huggingface-hub==0.36.2` | direct runtime | Apache-2.0 | [original notices](LICENSES/dependencies/huggingface-hub-0.36.2.txt) |
| `idna==3.20` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/idna-3.20.txt) |
| `iniconfig==2.3.1` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/iniconfig-2.3.1.txt) |
| `jinja2==3.1.6` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/jinja2-3.1.6.txt) |
| `joblib==1.6.0` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/joblib-1.6.0.txt) |
| `markupsafe==3.0.4` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/markupsafe-3.0.4.txt) |
| `mpmath==1.3.0` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/mpmath-1.3.0.txt) |
| `narwhals==2.27.1` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/narwhals-2.27.1.txt) |
| `networkx==3.6.1` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/networkx-3.6.1.txt) |
| `networkx==3.7` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/networkx-3.7.txt) |
| `numpy==2.4.6` | direct runtime | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [original notices](LICENSES/dependencies/numpy-2.4.6.txt) |
| `numpy==2.5.4` | direct runtime | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [original notices](LICENSES/dependencies/numpy-2.5.4.txt) |
| `packaging==26.3` | transitive (including platform-conditional dependencies) | Apache-2.0 OR BSD-2-Clause | [original notices](LICENSES/dependencies/packaging-26.3.txt) |
| `pluggy==1.7.0` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/pluggy-1.7.0.txt) |
| `pyarrow==23.0.1` | optional data | Apache-2.0 (plus bundled components) | [original notices](LICENSES/dependencies/pyarrow-23.0.1.txt) |
| `pydantic==2.14.1` | direct runtime | MIT | [original notices](LICENSES/dependencies/pydantic-2.14.1.txt) |
| `pydantic-core==2.50.1` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/pydantic-core-2.50.1.txt) |
| `pygments==2.21.0` | transitive (including platform-conditional dependencies) | BSD-2-Clause | [original notices](LICENSES/dependencies/pygments-2.21.0.txt) |
| `pyproject-hooks==1.3.3` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/pyproject-hooks-1.3.3.txt) |
| `pytest==9.1.1` | direct development | MIT | [original notices](LICENSES/dependencies/pytest-9.1.1.txt) |
| `pyyaml==6.0.3` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/pyyaml-6.0.3.txt) |
| `regex==2026.9.29` | transitive (including platform-conditional dependencies) | Apache-2.0 AND CNRI-Python | [original notices](LICENSES/dependencies/regex-2026.9.29.txt) |
| `requests==2.34.2` | transitive (including platform-conditional dependencies) | Apache-2.0 | [original notices](LICENSES/dependencies/requests-2.34.2.txt) |
| `ruff==0.17.1` | direct development | MIT | [original notices](LICENSES/dependencies/ruff-0.17.1.txt) |
| `safetensors==0.8.0` | transitive (including platform-conditional dependencies) | Apache-2.0 | [original notices](LICENSES/dependencies/safetensors-0.8.0.txt) |
| `scikit-learn==1.9.1` | direct runtime | BSD-3-Clause (plus bundled GCC runtime) | [original notices](LICENSES/dependencies/scikit-learn-1.9.1.txt) |
| `scipy==1.17.1` | transitive (including platform-conditional dependencies) | BSD-3-Clause (plus bundled components) | [original notices](LICENSES/dependencies/scipy-1.17.1.txt) |
| `scipy==1.18.1` | transitive (including platform-conditional dependencies) | BSD-3-Clause (plus bundled components) | [original notices](LICENSES/dependencies/scipy-1.18.1.txt) |
| `setuptools==84.2.1` | build backend and transitive | MIT (plus vendored packages) | [original notices](LICENSES/dependencies/setuptools-84.2.1.txt) |
| `sympy==1.14.0` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/sympy-1.14.0.txt) |
| `threadpoolctl==3.7.0` | transitive (including platform-conditional dependencies) | BSD-3-Clause | [original notices](LICENSES/dependencies/threadpoolctl-3.7.0.txt) |
| `tokenizers==0.22.2` | transitive (including platform-conditional dependencies) | Apache-2.0 | [original notices](LICENSES/dependencies/tokenizers-0.22.2.txt) |
| `torch==2.14.1` | direct runtime | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | [original notices](LICENSES/dependencies/torch-2.14.1.txt) |
| `torch==2.14.1+cpu` | direct runtime | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | [original notices](LICENSES/dependencies/torch-2.14.1+cpu.txt) |
| `tqdm==4.70.1` | transitive (including platform-conditional dependencies) | MPL-2.0 AND MIT | [original notices](LICENSES/dependencies/tqdm-4.70.1.txt) |
| `transformers==4.57.6` | direct runtime | Apache-2.0 | [original notices](LICENSES/dependencies/transformers-4.57.6.txt) |
| `typing-extensions==4.16.0` | transitive (including platform-conditional dependencies) | PSF-2.0 | [original notices](LICENSES/dependencies/typing-extensions-4.16.0.txt) |
| `typing-inspection==0.4.4` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/typing-inspection-0.4.4.txt) |
| `urllib3==2.8.0` | transitive (including platform-conditional dependencies) | MIT | [original notices](LICENSES/dependencies/urllib3-2.8.0.txt) |


### Conditions needing more than an MIT notice

* **MIT/BSD/0BSD/Zlib**: retain the original applicable copyright, conditions and
  disclaimers. BSD non-endorsement and Zlib origin/modification conditions remain.
* **Apache-2.0**: distribute the license and applicable original notices, preserve
  supplied NOTICE attribution, and mark modified files. Patent-termination and
  trademark terms remain. `packaging` offers Apache-2.0 **OR** BSD-2-Clause.
* **MPL-2.0** in certifi/tqdm: retain notices and meet applicable file-level source
  availability requirements when distributing covered material. The complete
  [MPL text](LICENSES/texts/MPL-2.0.txt) supplements upstream notices. It does not
  automatically turn the whole application into MPL software.
* **NumPy/SciPy/scikit-learn/PyTorch binaries**: retain their composite notices.
  The inspected NumPy/SciPy wheels include LGPL-2.1-or-later libquadmath and GCC
  runtime code under GPL-3.0-or-later WITH GCC-exception-3.1. Scikit-learn identifies
  libgomp and its runtime exception; PyTorch also contains libgomp. Verify exact
  binaries, applicable exception conditions, source availability/offers and LGPL
  replacement/relinking requirements before shipping an environment or binary bundle.
  [LGPL text](LICENSES/texts/LGPL-2.1-or-later.txt),
  [GPL text](LICENSES/texts/GPL-3.0-or-later.txt),
  [GCC exception](LICENSES/texts/GCC-exception-3.1.txt).
* **regex** includes CNRI-Python-derived material as well as Apache-2.0 additions.
  Its original notice and the complete [CNRI-Python text](LICENSES/texts/CNRI-Python.txt)
  are retained. **typing-extensions** has PSF-2.0 terms in its original notice bundle.
* **Nested libraries**: NumPy, SciPy, PyTorch, Arrow, scikit-learn, SymPy and setuptools
  include additional components. Complete shipped notices are retained, including
  vendor paths in the JSON inventory. This is not an exhaustive binary SBOM.
* **tokenizers 0.22.2**: the installed wheel omits a license file. The linked Apache
  text was recovered from its exact upstream release tag. Compiled Rust/native
  subdependency attribution completeness still requires review before redistribution.

## External development tools

These are not application code vendored by ResumeAnalysis:

* `actions/checkout@v4`: MIT, Copyright (c) 2018 GitHub, Inc. and contributors.
  [Original license](LICENSES/tools/checkout.txt).
* `astral-sh/setup-uv@v6`: MIT, Copyright (c) 2024 Kevin Stillhammer.
  [Original license](LICENSES/tools/setup-uv.txt).
* `uv 0.12.23`: MIT OR Apache-2.0; original
  [MIT](LICENSES/tools/uv-LICENSE-MIT.txt) and
  [Apache](LICENSES/tools/uv-LICENSE-APACHE.txt) texts retained.
* Python and the host/runner operating system have separate platform licenses.
  A redistributed interpreter or environment requires a platform-specific inventory;
  the project's root MIT file does not cover it.

Tag commits and all notice-source hashes are recorded in the evidence index.

## Original reference-project notices (verbatim)

These are retained conservatively for code provenance; their inclusion does not
claim that the projects' complete implementations are vendored here.

### ResuMatch

```text
MIT License

Copyright (c) 2026 vdenterprises

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

### Nesta OJD DAPS Skills

```text

The MIT License (MIT)
=====================
Copyright (c) 2024, Nesta

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

```

### SkillAlign source code

```text
MIT License

Copyright (c) 2025 YASSER

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---
This MIT license applies to the SkillAlign source code only. See NOTICE for third-party data licenses.
```

### SkillSpan code/data repository

```text
MIT License

Copyright (c) 2023 Mike Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## Qwen backend addendum — 2026-10-23

The optional inference model is `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`.
The actual pinned [upstream LICENSE](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/LICENSE)
is Apache-2.0 and is retained in full at
[LICENSES/references/qwen-instruct-7b-LICENSE.txt](LICENSES/references/qwen-instruct-7b-LICENSE.txt).
No separate NOTICE was listed in this model repository at the reviewed revision.
Do not invent an asset-specific copyright line from the publisher's name. Preserve
actual supplied copyright/license/NOTICE material, identify modifications, retain
Apache patent and trademark conditions, and review training/output provenance before
redistributing weights or making rights assurances. Weights/tokenizers are downloaded
separately, not committed. Inference retains and verifies their source manifest.

Five additional locked packages are optional. The linked bundles preserve their
original complete notices, including the named copyright holders, byte-for-byte
between file boundaries. Their original names are not replaced by this project's name.

| Package/version | Actual license | Original full notice |
|---|---|---|
| accelerate 1.15.0 | Apache-2.0 | [LICENSE](LICENSES/dependencies/accelerate-1.15.0.txt) |
| bitsandbytes 0.50.2 | MIT; copyright (c) Facebook, Inc. and its affiliates | [LICENSE](LICENSES/dependencies/bitsandbytes-0.50.2.txt) |
| lm-format-enforcer 0.11.3 | MIT; copyright (c) 2023 Noam Gat | [LICENSE](LICENSES/dependencies/lm-format-enforcer-0.11.3.txt) |
| interegular 0.3.3 | MIT; copyright (c) 2019 @MegaIng and @qntm | [LICENSE](LICENSES/dependencies/interegular-0.3.3.txt) |
| psutil 7.2.2 | BSD-3-Clause; original multiple copyright holders retained | [LICENSE](LICENSES/dependencies/psutil-7.2.2.txt) |

The first audit's 48 package/version entries remain unchanged; the complete current
index has 53. Four additions were inspected from installed wheel metadata/notices;
the exact locked Linux x86-64 bitsandbytes wheel was downloaded, hash-verified and
inspected without installation. This is not an exhaustive inventory of compiled
CUDA kernels, every platform wheel or external NVIDIA driver/runtime components.
A separate CUDA-enabled PyTorch installation can introduce additional NVIDIA and
native licenses that the CPU lock does not represent. Review the exact GPU release
artifact and its source/notice obligations before distributing a bundled runtime.

## Skill extractor v2 addendum — 2026-10-24

The reviewed references are markovml/skill-ner
(`c23846c19a084d48571b1e4a88f4095cba1094c0`), AnasAito/SkillNER
(`e8cdbb6f591211bf5d0fe2c8a61ddb95526a78b3`), SkillSpan
(`2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b`), and JobBot
(`e2f38a1a8cf0714e973bf9d4b713cacb5a4cafc1`).
No implementation source, EMSI/Lightcast skill dictionary or pretrained NER weights
from these references were copied or imported. The new curated vocabulary and
requirement rules are original project code under the root MIT. Research is not
an assertion that all referenced assets are MIT or cleared for reuse.

The optional phrase matcher derives a separately generated local vocabulary from
**SkillSpan's training annotations only**. SkillSpan's actual original MIT notice,
**Copyright (c) 2023 Mike Zhang**, is already retained above and in
[LICENSES/references/skillspan-LICENSE.txt](LICENSES/references/skillspan-LICENSE.txt).
Its full notice is also embedded in the generated derivative artifact with the
source revision/hash and declared transformations (exact phrase counting, frequency,
length and ambiguity/disjunction filters). The corpus and generated vocabulary
are excluded from Git. The SkillSpan scholarly citation above still applies.
Underlying advertisement rights and intended redistribution remain review items.

The two SkillNER repositories supply MIT code notices naming **Copyright (c) 2021,
Anas AIT AOMAR & Badr MOUFAD. All rights reserved.** No code is included here, so this
is provenance research rather than an incorporated-code notice. Their separate
default EMSI dictionary terms remain unresolved. JobBot has no verified license
grant; do not copy its code/assets based on public access or its academic statement.
The inspected JobBERT skill/knowledge weight repositories have no verified
asset-specific license declaration/file and were not downloaded or executed.

Read [the research matrix](docs/skill_extractor_research.md) and
[evidence index](docs/skill_extractor_evidence.json) for exact scope, terms and
excluded candidates. No new software dependencies or third-party source licenses
were introduced by this extraction update; existing model/data/taxonomy/native
runtime review items remain in force.
## Skill dataset research and preprocessing additions — 2026-10-23

The original MIT project license does not relicense external datasets. This
addition includes independently authored download/preprocessing code and a retained
license text; dataset payloads and derived phrase artifacts remain outside Git.
Original upstream README/license files are preserved unchanged in local downloads.

- **Green, Maynard and Lin job-description entity corpus**:
  https://github.com/acp19tag/skill-extraction-dataset/tree/ca1eadd2885a277853b01f0ec96fc89094adf71e.
  Thomas AF Green, Diana Maynard and Chenghua Lin, *Development of a Benchmark
  Corpus to Support Entity Recognition in Job Descriptions*, LREC 2022,
  https://aclanthology.org/2022.lrec-1.128/. The upstream README credits original
  job descriptions to Kaggle user `airiddha`, `trainrev1`. Released under
  **Creative Commons Attribution 4.0 International**. The original full license
  is retained at [LICENSES/datasets/green-LICENSE.txt](LICENSES/datasets/green-LICENSE.txt),
  unchanged (SHA-256 `7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661`).
  Conversions reconstruct tokenized text, retain original labels/POS and provenance,
  and report duplicate/leakage exclusions; original downloaded files are not edited.
- **TechWolf TECH, HOUSE, TECHWOLF and Synthetic ESCO Skill Sentences**:
  publisher dataset cards declare **CC BY 4.0**,
  https://creativecommons.org/licenses/by/4.0/. Attribution: TechWolf and
  Jens-Joris Decorte and coauthors, *Design of negative sampling strategies for
  distantly supervised skill extraction* (2022), https://arxiv.org/abs/2209.05987,
  and *Extreme multi-label skill extraction training using large language models*
  (2023), https://arxiv.org/abs/2307.10778, as applicable to each pinned dataset.
  These sources use **ESCO v1.1.0**, European Commission,
  https://esco.ec.europa.eu/. The European Commission is not responsible for any
  use of the information. TECH/HOUSE derive from SkillSpan; preserve SkillSpan's
  original attribution and notices as well. No standalone license files were
  listed in these dataset releases; the specific publisher declarations and
  pinned URLs are retained in [configs/skill_datasets.json](configs/skill_datasets.json).
  Original advertisement rights and version-specific ESCO terms remain review
  items before redistribution. Source rows are grouped without inventing offsets;
  derived metadata identifies these conversions and their sources.
- **Skill-XL**, TechWolf:
  https://huggingface.co/datasets/TechWolf/Skill-XL/tree/221cbec139621712633cbc2d27dcd2daa68e48ed.
  Publisher card declares **CC BY 4.0**, with no separate LICENSE or annotation
  protocol. Preserve TechWolf attribution, source/revision and the license link in
  derived metadata. Missing source-rights/annotation information requires review
  before any redistribution or additional training use.
- **IT job title to key skills (hh.ru, 2023)**, MrAlexGov:
  https://huggingface.co/datasets/MrAlexGov/it-job-title-skills-ru/tree/9863ee993040c43305627c72227fc50415e257ad.
  Publisher declares **CC BY 4.0** and credits Kaggle releases by
  `etietopabraham` (`jobs-raw-data`) and `ilyazawilsiv`
  (`it-vacancies-from-headhunter-website`). The original platform is hh.ru.
  Upstream license claims could not be independently verified; this asset is
  restricted to local research inventory in the new tooling, without eligible
  training/evaluation output. No payload is redistributed here.

The existing SkillSpan MIT notice above remains unchanged. Its author's separate
Hugging Face mirror declares CC BY 4.0; this discrepancy is recorded for manual
review rather than silently relicensing the pinned GitHub snapshot. New
train-derived experimental phrase artifacts carry applicable full notices,
attributions, source hashes and a description of modifications. JobBot,
unknown-license Sayfullina annotations, unresolved SkillSkape mirror data and new
model weights were not imported. Model-card research is not model redistribution.

## Optional multilingual cross-encoder (Retrieve → Re-rank experiment)

The explicit `download-reranker` command downloads
**cross-encoder/mmarco-mMiniLMv2-L12-H384-v1**, published by the
Sentence-Transformers / cross-encoder project, at revision
`1427fd652930e4ba29e8149678df786c240d8825`:
https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825.
The pinned publisher model card explicitly declares **Apache License 2.0**.
The upstream tree has no standalone LICENSE or NOTICE file. Its unchanged card
is retained in [LICENSES/references/reranker-model-card.txt](LICENSES/references/reranker-model-card.txt).
The complete Apache license, including the original copyright text of the
reference Sentence-Transformers project, is retained in
[LICENSES/references/reranker-Apache-LICENSE.txt](LICENSES/references/reranker-Apache-LICENSE.txt).
That reference license is not represented as an additional file in the model tree.

The checkpoint is used unchanged. Model weights are excluded from Git and software
packages. If distributing a model bundle, retain the upstream attribution and
license, identify any modifications and comply with Apache-2.0 section 4.
ResumeAnalysis's MIT license covers its original code and does not replace those
terms. This implementation does not copy Sentence-Transformers implementation code.

The publisher identifies an mMiniLMv2 model distilled from XLM-R Large and
Google-translated mMARCO / MS MARCO passage-ranking data as provenance. The final
publisher grant does not resolve every base-model or training-data rights question.
No mMARCO data is downloaded or redistributed here. See the pinned sources and
concrete manual-review limits in [docs/rerank_model_selection.md](docs/rerank_model_selection.md).
Independent relevance datasets reviewed for this experiment were not imported;
the authored annotation templates contain no ground-truth hiring judgments.

## Cover-letter generation components

**ApplyKit**, `wihlarkop/applykit`, revision
`002360a6d5af14f9eeedae29934291fc95b22f97`, is MIT licensed:
https://github.com/wihlarkop/applykit/tree/002360a6d5af14f9eeedae29934291fc95b22f97.
Copyright (c) 2026 ApplyKit Contributors. The complete original copyright and
permission notice is preserved in [LICENSES/cover_letters/applykit-LICENSE.txt](LICENSES/cover_letters/applykit-LICENSE.txt).
The cover-letter system prompt, JSON input-boundary helpers and prompt-builder
logic originate in `backend/app/services/prompts.py` and
`backend/app/routes/generate.py`. Source headers identify their origin.
Adaptations add project-native input schemas, explicit English/Russian output,
bounded length and data-only candidate context; database, PDF, UI and cloud
provider code are omitted. The final adaptation shortens the evidence paragraph,
removes the mandatory STAR/result instruction and adds source-grounding constraints.

**Resume-Matcher**, `srbhr/Resume-Matcher`, revision
`f6296a5e6a31c9d616620143688cc909623c542f`, is Apache-2.0 licensed:
https://github.com/srbhr/Resume-Matcher/tree/f6296a5e6a31c9d616620143688cc909623c542f.
The cover-letter prompt from `apps/backend/app/prompts/templates.py` is retained
for the reproducible comparison. Its complete upstream license is preserved
unchanged in [LICENSES/cover_letters/resume-matcher-LICENSE.txt](LICENSES/cover_letters/resume-matcher-LICENSE.txt).
No standalone NOTICE was found. The benchmark supplies an explicit Russian
language name and a common word limit, and uses the existing local Qwen runtime
instead of the upstream application/provider stack. These modifications do not
relicense the original material under the project's MIT license.

**Qwen2.5-1.5B-Instruct**, Qwen Team, Alibaba Cloud, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
The publisher grants Apache-2.0; the complete original model license is retained
in [LICENSES/cover_letters/qwen-1.5b-LICENSE.txt](LICENSES/cover_letters/qwen-1.5b-LICENSE.txt).
Model weights are not modified or committed. The explicit downloader retains the
upstream card, license and all artifact checksums. This smaller CPU benchmark
does not establish quality for the separately configured Qwen2.5-7B backend.

The stronger local comparison uses **Qwen3-4B-Instruct-2507**, Qwen Team,
revision `cdbee75f17c01a7cc42f958dc650907174af0554`:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554.
Its original Apache-2.0 license is preserved byte-for-byte in
[LICENSES/cover_letters/qwen-4b-LICENSE.txt](LICENSES/cover_letters/qwen-4b-LICENSE.txt).
Downloaded safetensors are unchanged and excluded from Git. Explicit CPU bfloat16
loading changes in-memory numeric representation, not the distributed weight files.

No code from the three candidates with incomplete license evidence was imported.
See [the source audit](docs/cover_letter_audit.md) and pinned evidence for the
scope of reuse and remaining quality/license limits. Original notices remain intact.

## Online recommendation and profile ingestion dependencies

The online recommendation path imports maintained library APIs. It does not copy
River, Instructor, Unstructured or ApplyPilot implementation code. Exact locked
versions, source links, license-file hashes and the 32 newly resolved dependency
releases are recorded in [docs/online_dependency_licenses.json](docs/online_dependency_licenses.json).
Complete notices from their distributions are preserved byte-for-byte in
[LICENSES/online_recommender](LICENSES/online_recommender), including bundled
third-party notices provided by OpenAI, lxml and other distributions. Existing
dependency, model, dataset and taxonomy notices above remain applicable.

| Direct library | Version | Actual license and use |
|---|---|---|
| [River](https://github.com/online-ml/river) | 0.26.1 | BSD-3-Clause; Copyright (c) 2020, the river developers. LogisticRegression, SGD, StandardScaler and online metrics; no modifications to the library. |
| [Instructor](https://github.com/567-labs/instructor) | 1.17.0 | MIT; Copyright (c) 2023 Jason Liu. Optional structured extraction through a configured local compatible provider. |
| [pypdf](https://github.com/py-pdf/pypdf) | 6.21.2 | BSD-3-Clause. Optional text-bearing PDF extraction; original authors' copyright notices remain in the complete distributed license. |
| [python-docx](https://github.com/python-openxml/python-docx) | 1.2.0 | MIT; Copyright (c) 2013 Steve Canny. Optional DOCX paragraph/table extraction. |

The Instructor dependency graph includes OpenAI SDK 3.3.0 (Apache-2.0); its API
client is configured for a local endpoint and is not a requirement to buy or call
OpenAI's service. Library licenses do not grant rights to a separately selected
LLM's weights. The existing E5 model and its model license are unchanged. No new
LLM weights were downloaded for profile extraction; deployments must retain the
license and exact model identity of their chosen local model.

MIT/BSD notices must be retained with redistributed copies; BSD non-endorsement
terms also apply. Apache-2.0 dependencies retain their complete licenses and any
distributed NOTICE material. PSF-2.0, ISC and dual MIT/Apache dependencies retain
their respective full terms. The project MIT file covers original ResumeAnalysis
code and does not replace these grants. The conditional `httpx2-jsfetch` wheel's
BSD-3-Clause license was verified separately; that WebAssembly dependency was not
installed or executed in the local CPU implementation.

Unstructured's Apache-2.0 source and dependency list were reviewed, but the library
was not installed or copied. The inspected ApplyPilot repository has no declared
license; only independently implemented adapters to its public data shapes are
included. See [source evidence](docs/online_source_evidence.json). A future import
of ApplyPilot implementation code would require license clarification.
