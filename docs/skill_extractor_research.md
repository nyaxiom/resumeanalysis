# Skill extractor v2: upstream research and selection

Reviewed on 2026-10-24. This research preceded implementation of the replacement
extractor option. It distinguishes repository code, dictionaries, training data
and model weights: a code license is not automatically a license for every
downloaded asset. No implementation source, skill database or NER weights from
the projects below were copied or imported for this change.

The selected approach is an independently authored CPU phrase/rule recognizer with
explicit source spans and a separate requirements/context layer, reusing the
project's existing normalization. The original extractor remains selectable.
External NER is deferred behind the replaceable extractor interface; no external
NER implementation or automatic model download is enabled. This choice addresses requirement semantics and
reproducibility without adding an unmeasured language-model stack. Its quality
must be assessed in the project's own comparison; research alone does not prove
an extraction improvement.

## Repository comparison

| Candidate, pinned source | Approach and language evidence | Runtime and maintenance evidence | Actual terms and decision |
|---|---|---|---|
| [markovml/skill-ner](https://github.com/markovml/skill-ner/tree/c23846c19a084d48571b1e4a88f4095cba1094c0), `c23846c19a084d48571b1e4a88f4095cba1094c0` | Fork of AnasAito/SkillNER, not an independent trained NER. spaCy phrase matching, abbreviations, stems, partial n-grams and heuristic/vector similarity; English default. A supplied translation callback is not native Russian extraction or preserved Russian offsets. | No CUDA/PyTorch requirement. Adds spaCy, pandas, NLTK, jellyfish, NumPy and IPython; imports SciPy. Last inspected commit 2023-01-21; last push 2023-08-22, not archived. No modern Python 3.11/3.12 execution or CPU/RAM measurement here. | [Actual MIT LICENSE](https://github.com/markovml/skill-ner/blob/c23846c19a084d48571b1e4a88f4095cba1094c0/LICENSE), **Copyright (c) 2021, Anas AIT AOMAR & Badr MOUFAD. All rights reserved.** Default EMSI-derived dictionary provenance/terms need separate review. Reference only. |
| [AnasAito/SkillNER](https://github.com/AnasAito/SkillNER/tree/e8cdbb6f591211bf5d0fe2c8a61ddb95526a78b3), `e8cdbb6f591211bf5d0fe2c8a61ddb95526a78b3` | Same underlying dictionary/PhraseMatcher approach; exact, abbreviated, unigram, lower-surface-form and partial-token passes. English spaCy model, English dictionary and Porter stemming; no Russian task evaluation found. | PyPI `skillNer==1.0.3` last released 2021-11-07. Development environment specifies Python 3.8; setuptools gives generic Python 3 with no bounded dependency versions. Last inspected source commit 2023-08-22; last push 2024-01-28, not archived. No measured throughput/RAM here. | [Actual MIT LICENSE](https://github.com/AnasAito/SkillNER/blob/e8cdbb6f591211bf5d0fe2c8a61ddb95526a78b3/LICENSE), same original copyright holders as the fork. Root MIT does not establish the source EMSI database's separate rights. Reference only. |
| [kris927b/SkillSpan](https://github.com/kris927b/SkillSpan/tree/2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b), `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b` | Expert-annotated English SKILL/KNOWLEDGE BIO spans; BERT/SpanBERT and job-domain-adapted variants with separate or multitask labeling. A research dataset/runner rather than a lightweight production extractor. No Russian benchmark supplied. | README retains Python 3.6/PyTorch 1.7 instructions and describes February 2024 migration to newer MaChAmp without AllenNLP. Modern runner compatibility was not tested. Last push 2025-02-13, not archived. Model CPU/RAM results unmeasured here. | [Actual MIT LICENSE](https://github.com/kris927b/SkillSpan/blob/2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b/LICENSE), **Copyright (c) 2023 Mike Zhang**. README releases code/data and requests the paper citation. Existing project notices retain these terms; underlying advertisement rights and exported model licenses remain separate. Retain the existing pinned evaluation dataset; do not copy the training stack. |
| [SwathyAnand100/JobBot](https://github.com/SwathyAnand100/JobBot/tree/e2f38a1a8cf0714e973bf9d4b713cacb5a4cafc1), `e2f38a1a8cf0714e973bf9d4b713cacb5a4cafc1` | Academic Colab notebooks: spaCy dictionary baseline, DistilBERT token classification, then ESCO normalization. English LinkedIn jobs/English ESCO. README describes cross-encoder reranking, while the examined normalization notebook uses SentenceTransformer; validate actual behavior before reuse. No Russian benchmark. | README Python 3.8+/Colab; training installs Transformers 4.57.3, accelerate, datasets and seqeval. Other notebooks add spaCy, pandas, rank-bm25 and sentence-transformers. Google Drive imports/paths prevent direct library use. Last push 2026-02-16, not archived. CPU latency/RAM unspecified and unmeasured here. | Complete inspected tree has **no LICENSE or NOTICE**. README says academic purposes for CS685 at UMass Amherst; this is not an open-source reuse/redistribution grant. Annotation/checkpoint payloads described in README were absent. Do not copy its code, annotations, checkpoints or ESCO exports without appropriate separate permission. |

### Why SkillNER is not a direct default dependency

The two SkillNER repositories recognize skill candidates, but their `annotate`
output does not distinguish mandatory, optional, negated or alternative requirements.
The returned text is transformed and lowercased; `doc_node_id` indexes transformed
tokens rather than unmodified input character offsets. Adopting it would still need
an independently tested alignment and requirement-classification layer.

More importantly, importing `skillNer.general_params` reads dictionary files from
the current working directory or fetches them from the mutable `first_release`
branch of AnasAito/SkillNER, then writes them into that directory. That default
behavior does not provide a pinned, checksum-verified offline asset boundary.
The README attributes the database to EMSI, but no separate provider-version license
or attribution record was found in the reviewed source. The current official
EMSI/Lightcast terms URLs returned proxy HTTP 403, so their terms were not verified
and are not inferred from a README's description of the data as open source.

The pinned upstream tree lists `skill_db_relax_20.json` as **7,061,931 bytes**,
`token_dist.json` as **184,416 bytes**, and the intermediate processed dictionary
as **11,545,962 bytes**. The fork has one additional byte in each file. No dictionary
payload was imported into ResumeAnalysis. The upstream README describes its default
spaCy model download as approximately 800 MB; that is a historical download claim,
not a measured RAM requirement.

Unbounded dependencies also deserve a fresh compatibility test before import.
For example, the original source's fallback calls `jellyfish.jaro_distance`, while
the fork changes it to `jaro_similarity` and modifies similarity/scoring behavior.
The fork is therefore not an identical baseline. No blanket claim of current
Python, NumPy or spaCy compatibility is made from old packaging metadata.

## Model and asset candidates

Sizes below come from repository/LFS metadata, not downloads or runtime profiling.
None of these model weights was executed for this research. A model-card declaration
is recorded separately from an actual standalone license file. Missing files do
not prove that no grant exists elsewhere; they mean this review did not establish
the asset-specific grant needed for reuse.

| Model and exact revision | Artifact/dependency evidence | Language and rights findings | Selection |
|---|---|---|---|
| `jjzha/jobbert_skill_extraction`, `a6211b675e287f3a07ffd4834b8182a2b519589b` | Standard `BertForTokenClassification`; **430,915,468-byte** safetensors. Existing Transformers can represent the architecture without requiring the original training runner; execution compatibility not tested. | English SkillSpan task; no Russian validation. Model card has no license declaration and the repository has no LICENSE file; README gives citation/demo context only. | Do not download or enable until model-specific terms are verified. |
| `jjzha/jobbert_knowledge_extraction`, `9dea9759b7e290413cb4d92e8cae338affab9be4` | Same architecture and **430,915,468-byte** safetensors; separate model needed for KNOWLEDGE coverage. | Same English scope and missing asset-specific license evidence. Source/data MIT is not silently substituted for a model grant. | Excluded pending license clarification. |
| `jjzha/escoxlmr_skill_extraction`, `180ca84c8be549e5f6564f61afef452cc0333032` | XLM-R-large; **2,235,428,364-byte** safetensors. | Card declares Apache-2.0; no standalone LICENSE listed. Multilingual pretraining does not establish Russian skill/requirement quality after English task adaptation. | Too large for an unmeasured lightweight default; retain as a future measured option after rights/provenance review. |
| `jjzha/escoxlmr_knowledge_extraction`, `11ff79181beba00102094005d0408a2d894e3092` | Separate XLM-R-large KNOWLEDGE model; **2,235,428,364-byte** safetensors. | Same card-only Apache-2.0 declaration and unverified Russian task quality. | Same deferred decision. |
| `nestauk/en_skillner`, `2610a6de4a5764a167c2eceff7cfd0a3e89e7c40` | English spaCy **>=3.7.4,<3.8.0**; **587,689,424-byte** wheel, **616,988,528-byte** unpacked vectors. These alternative encodings must not be summed as RAM. | MIT card declaration, no standalone model LICENSE listed. Card explicitly says maintenance/development stopped in May 2025. Card reports skill F1 0.724 and all-entity F1 about 0.5838 on its own evaluation; not a SkillSpan result reproduced here. | Heavy, English-only and no longer maintained; not selected as the default. |
| `spacy/en_core_web_lg`, `557bf758b8b070251eef3a62dd5442e72e72e2ce` | Separately inspected supporting model snapshot: version **3.7.1**, spaCy **>=3.7.2,<3.8.0**, **587,689,537-byte** wheel, **616,988,528-byte** unpacked vectors. This does not assert that SkillNER pins this version. | English, actual MIT LICENSE plus `LICENSES_SOURCES`. Source notices cover separate corpus/vector/lookup provenance; do not replace them with the application MIT. Generic spaCy NER labels are not a skill-span model. | Supporting dependency research only; no asset imported. |
| `TechWolf/ConTeXT-Skill-Extraction-base`, `039a63443fa9b82b1f97d138de11646ec966016f` | MPNet sentence-transformer embeddings; **437,967,672-byte** safetensors. | No model license or LICENSE found; generated card does not establish an extraction-specific grant. Embeddings alone do not produce BIO spans or requirement categories. | Not a direct replacement; no integration. |

The two small JobBERT models would require roughly 862 MB of weight downloads
together, before tokenizer/runtime overhead. The two XLM-R-large models would
require roughly 4.47 GB of weights. These are artifact totals, not estimates of
inference RAM or latency. No external project's CPU throughput, peak RAM or Russian
accuracy is invented from architecture or download size.

## Benchmarks and comparability

The SkillSpan [paper](https://aclanthology.org/2022.naacl-main.366/) reports that
job-domain adaptation improves its English task and single-task models outperform
multitask alternatives in its experiments. That published conclusion is not a new
ResumeAnalysis measurement or a guarantee about current exported weights.
The existing project evaluation covers all SKILL and KNOWLEDGE annotations, with
exact token-span and token-overlap metrics. Preserve the same input, evaluator and
scope when comparing the legacy and selected v2 extractors. Do not tune a lexicon
against the held-out test labels and then describe it as an independent test.

JobBot's README reports the following on its own **700 manually annotated
sentences** (490/105/105 split). The relevant annotations and fine-tuned checkpoint
were not present in the inspected repository, so these are upstream claims only:

| JobBot component | Published strict F1 | Published lenient F1 |
|---|---:|---:|
| Dictionary extraction | 0.175 | 0.405 |
| DistilBERT extraction | 0.619 | 0.797 |
| End-to-end pipeline | 0.366 | 0.540 |

These values cannot be ranked against ResumeAnalysis's SkillSpan results: the
examples, annotations and scoring definitions differ. No comparable quantitative
SkillSpan or Russian benchmark was found in the inspected SkillNER README/source.
No external extractor has a newly measured score, latency or RAM entry here because
it was excluded from execution after the rights/dependency review. Publishing an
empty or explicitly unverified result is preferable to inventing a same-data run.

Role Radar's composite relevance values are not annotated skill spans, requirement
labels, calibrated probabilities or independent human relevance judgments. Its
locally prepared 106 pairs may be absent from the cloud workspace. An offline
runner can report skills, requirement categories, disagreements, timings and
checkpoint-safe downstream scores when those files are supplied; descriptive output
must not be labeled extraction accuracy without independent annotations.

English and Russian controlled fixtures establish regression behavior separately.
They are not an independent Russian extraction benchmark, and an English model's
quality or multilingual initialization must not be used as a substitute.

## Reuse, attribution and remaining work

No new third-party implementation, external skill dictionary or model was copied
into tracked source. The optional phrase artifact derives from the pinned MIT
SkillSpan training annotations only, and remains outside Git with its source terms.
The curated vocabulary/context rules are original project code; no new upstream
copyright holder is substituted into the project MIT.
Existing SkillSpan source/data MIT notices and the requested scholarly citation
remain in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The preserved
[license audit](license_audit.md) still applies to ESCO, data/model provenance and
native runtime redistribution. Investigation of a public repository does not
clear its code or assets for copying.

Future direct reuse of MIT SkillNER code would require preserving its full original
notice, pinning the adapted files and stating modifications. Future dictionary or
model adoption needs its own version, source hashes, actual terms and required
attribution, plus CPU/RAM, same-data precision/recall and Russian fallback checks.
A local optional NER provider must retain original evidence spans and pass through
the same requirement/negation/alternative policy. It must not silently download
weights, infer skills from titles or make Qwen inference a dependency of extraction.

Changing extraction semantics changes the trained MLP's feature distribution.
Keeping the original extractor/model pair operational and requiring a matching
extractor/pipeline identity for a retrained checkpoint is part of selection, not
an optional performance optimization. Any reported downstream ablation must use
compatible model/extractor pairs and identical held-out examples, with explicit
requirements separated as a controlled intervention.
