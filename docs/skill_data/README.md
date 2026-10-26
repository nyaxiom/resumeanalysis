# Skill Extractor data foundation

The validation evidence supports **limited vocabulary coverage as a major recall
bottleneck, but rejects naive vocabulary expansion as a sufficient fix**. Only
12.69% of validation gold mentions equal a V2 alias. A train-only phrase expansion
raises recall but lowers precision enough to reduce F1. Better real span data and
context-sensitive boundary learning are the next experiments; more synthetic
sentence labels cannot substitute for span supervision.

This work adds research and offline dataset tooling on a feature branch. The
legacy default, V2 implementation, existing checkpoints and downstream matching
model are preserved. No merge to `main`, downstream training, pretrained NER
weight download, or new SkillSpan test evaluation is part of this work.

## Deliverables and reproduction

- [Measured inventory](inventory.md), with [machine-readable provenance](inventory.json).
- [Unified JSON schema](schema.json) and independently authored adapters in
  [`src/resumeanalysis/skill_data`](../../src/resumeanalysis/skill_data).
- [Quality and leakage evidence](quality_leakage.json).
- [Vocabulary coverage and validation experiments](vocabulary_coverage.json),
  interpreted in [findings.md](findings.md).
- [Additional sources and license review](additional_sources.md), including
  JobBot, SkillSkape, Russian candidates and lightweight base models.
- [Training experiment plan](training_strategy.md) and
  [machine-readable experiment configuration](../../configs/skill_data_experiments.json).

From the repository root, install the existing locked development environment
using the project's setup instructions, activate it, then run:

```bash
bash scripts/skill_data_foundation.sh
```

The first command explicitly uses the network. Every original asset is pinned by
repository commit, byte length and SHA-256 in
[`configs/skill_datasets.json`](../../configs/skill_datasets.json). Repeated
downloads verify existing files and refuse to overwrite changed originals.
Subsequent commands run offline. Originals and their upstream README/license
snapshots stay under `data/skill-foundation/raw`; normalized data stays in a
separate directory. Fingerprint sketches, per-record dispositions, near-overlap
details, audited partitions and derived phrase vocabularies stay under ignored
`artifacts/skill-foundation`. Neither datasets nor phrase artifacts enter Git.

Preprocessing, auditing and coverage require fresh output directories. For a
second run, supply new `--output` paths to individual commands; do not delete
original datasets or checkpoints. The checked-in reports contain aggregates and
hashes only. `python -m resumeanalysis.skill_data --help` lists all commands,
including `report` for publishing aggregate evidence. No command silently trains
the resume/vacancy model or changes the production vocabulary.

## Selected datasets

Counts below distinguish original CSV rows from grouped text examples. All
downloaded original files remain unchanged. Complete revisions, asset hashes,
license URLs, attributions and known risks are in the catalog and inventory.

| Dataset and original source | Samples and annotated skills | Language and actual license evidence | Annotation/task and extractor compatibility |
|---|---|---|---|
| [SkillSpan](https://github.com/kris927b/SkillSpan/tree/2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b), [NAACL 2022](https://aclanthology.org/2022.naacl-main.366/) | Public splits: 4,800 train / 3,174 development / 3,569 test sentences. Train/development contain 5,190 / 2,163 SKILL+KNOWLEDGE annotations. Test annotation counts are not recomputed in this work. | English. Actual GitHub MIT LICENSE, Copyright (c) 2023 Mike Zhang; README releases data. The [author's HF mirror](https://huggingface.co/datasets/jjzha/skillspan/tree/33062e6bd0e03a5e01ae299ce0518f4613ef4298) separately declares **CC BY 4.0**. Preserve attribution and clarify data-license scope before redistribution. | Two expert BIO layers; span extraction, no normalization IDs or requirement labels. Directly compatible with extraction scoring after explicitly reconstructing text with one space between tokens. Separate layers and original tokens are retained. Original advertisement character offsets are unavailable. |
| [TechWolf TECH](https://huggingface.co/datasets/TechWolf/skill-extraction-tech/tree/5f0cb05df03cd1cd5048ff1d64639bf9bd4cb36f), [paper](https://arxiv.org/abs/2209.05987) | Validation: 293 annotation rows / 101 sentence groups, 166 mapped labels and 127 `LABEL NOT PRESENT` outcomes; 68 distinct mapped labels. Sealed test: 1,164 rows according to revision-matched HF metadata; grouped/skill counts unverified. | English; publisher card explicitly declares CC BY 4.0; no standalone LICENSE. ESCO v1.1.0 and original advertisement rights remain separate review items. | `sentence,span,sub_span,label`. SkillSpan-derived surface-to-ESCO normalization. No offsets in this release; preserve surface strings without inventing offsets. **Every validation group is quarantined against SkillSpan test text**, so it is not an available independent validation set. Test annotations remain sealed. |
| [TechWolf HOUSE](https://huggingface.co/datasets/TechWolf/skill-extraction-house/tree/dcb313cad615a8f5148ef5addc01ef090fc9f295), same paper | Validation: 231 rows / 77 sentence groups, 144 mapped labels and 87 unmapped outcomes; 110 distinct mapped labels. Sealed test: 891 rows from revision-matched HF metadata; grouped/skill counts unverified. | English; CC BY 4.0 dataset-card declaration, no standalone LICENSE; same ESCO/source-text review. | Same surface normalization format and compatibility limits as TECH. All validation groups are quarantined against SkillSpan test text. Its validation name does not establish independence from the original SkillSpan test. |
| [TechWolf TECHWOLF](https://huggingface.co/datasets/TechWolf/skill-extraction-techwolf/tree/ed83e50ff6bc7e44008e98a82f6143ddaf59a255), [paper](https://arxiv.org/abs/2307.10778) | 588 label rows / 326 sentences; 314 distinct skill labels. | English; CC BY 4.0 card declaration, no standalone LICENSE. | Human sentence-level ESCO labels, **no spans**. Reserved external multi-label/normalization evaluation. Current local taxonomy covers only a small subset; compare normalized outputs only after an explicit ESCO mapping and annotation-policy review. Never score these labels as exact spans. |
| [Synthetic ESCO Skill Sentences](https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences/tree/09fc487d1394c1fdfd224cb621a89b14af925716), same paper | 138,260 sentence/skill pairs; 138,257 grouped exact-source sentences; 138,252 unique normalized texts; 13,826 distinct target strings. | English; CC BY 4.0 card declaration, no standalone LICENSE. Card says ten generated sentences per target and approximately 99.5% of ESCO v1.1.0 skills. | Synthetic sentence-to-skill classification/matching, no offsets. Auxiliary pool only. The target is not an exhaustive annotation of every incidental skill in a sentence. Do not create BIO tags by locating the target string, or treat synthetic scale as real annotated coverage. No phrase fitting uses this pool. |
| [Skill-XL](https://huggingface.co/datasets/TechWolf/Skill-XL/tree/221cbec139621712633cbc2d27dcd2daa68e48ed) | 5,847 development / 6,295 test CSV rows; 2,281 / 2,643 groups by job ID, subset and exact sentence. 3,957 / 4,514 nonempty skill-label occurrences; 1,662 / 1,740 distinct labels. | English from inspected content; the sparse card does not declare language or annotation methodology. CC BY 4.0 card declaration, no standalone LICENSE. | `ID,subset,title,sentence,relevant,cluster,skill`; sentence multi-label/relevance data, no spans. Retain relevance, cluster and missing labels. Job UUIDs permit additional group leakage checks. Reserve test for later compatible external classification; uncertain label completeness blocks treating missing skills as negatives. |
| [Green, Maynard and Lin corpus](https://github.com/acp19tag/skill-extraction-dataset/tree/ca1eadd2885a277853b01f0ec96fc89094adf71e), [LREC 2022](https://aclanthology.org/2022.lrec-1.128/) | Pinned processed files: 9,634 train / 336 evaluation sentences; 12,987 / 676 Skill spans. Other entity spans are retained separately. Datasheet raw counts are 10,000 / 586: they are not the processed counts. | English. Actual standalone CC BY 4.0 LICENSE and explicit README data grant. Authors credit Kaggle `airiddha/trainrev1` job texts; original rights and recruiter emails require review before redistribution. | Token-level CSV `sentence_id,word,pos,tag`, BIO Skill/Experience/Occupation/Qualification/Domain. Real crowdsourced annotations, worker qualification >70%; disagreement and upstream preprocessing remain limitations. Compatible auxiliary span supervision **for Skill only**, not a semantic equivalence between all Green entities and SkillSpan KNOWLEDGE. Evaluation remains reserved. |
| [Russian IT title/skill lists](https://huggingface.co/datasets/MrAlexGov/it-job-title-skills-ru/tree/9863ee993040c43305627c72227fc50415e257ad) | 51,029 title/list records; 28,576 unique normalized titles; 274,518 target-label occurrences; 1,119 distinct labels. | Publisher declares Russian and CC BY 4.0. Upstream Kaggle CC BY 4.0/Apache-2.0 claims could not be independently verified (HTTP 403); hh.ru source rights remain unresolved. | JSONL title, skills, experience, source. Real vacancy metadata but no descriptions, mention offsets or exhaustive skill gold. Rare skills and 71 general skills removed upstream. **Research inventory only; no eligible training/evaluation partition.** A skill may be absent from the title, so extraction/recall and title-to-competence inference are inappropriate. |

Language totals use dataset metadata and the documented English assumption for
Skill-XL, not per-record language detection. English technical tags in Russian
vacancy records do not establish Russian-language mention coverage. The additional
source review records which promising assets were excluded because annotations,
label mappings, original terms or licenses could not be verified.

## Schema and loss handling

`SkillRecord` preserves source text, language, annotation type, all annotations,
optional source tokens, full original row objects, and provenance: repository,
revision, original file SHA-256, source row numbers, split, intended role, group
ID where meaningful, attribution, license and license-evidence URLs.

Offsets are zero-based half-open Unicode character positions in the **declared
text**, checked against the exact surface. For BIO sources that text is a declared
one-space token reconstruction; the original tokens and BIO arrays remain intact.
For surface normalization and sentence/title labels, offsets stay `null` even if
a label happens to appear verbatim in the text. `LABEL NOT PRESENT`, Skill-XL
relevance/cluster values, POS tags and non-skill entities are retained. No new
normalization IDs are invented from label strings.

Adapters require known fields, valid arrays, valid BIO transitions and literal
offsets. Unknown fields, malformed JSON/CSV, blank records, empty datasets and
invalid annotations fail the run with recorded rejections; failed preprocessing
cannot enter the audit. Every compatible original source row contributes to a
group with retained row numbers and original fields. No source labels are silently
flattened into resume/vacancy pair labels or requirement categories.

## Leakage policy and data quality

The only operation on the original SkillSpan test is checksum verification and
**text-only fingerprinting for contamination exclusion**. Its `tokens` array is
decoded in isolation; annotation arrays are never decoded, exported or scored by
this pipeline. The resulting 2,660 unique text fingerprints and shingle sketches
are integrity-pinned. The original 3,569 test rows remain unchanged. TECH/HOUSE
test files are checksum-verified and skipped by annotation preprocessing, and
their known hashes cannot be relabeled as training through this tooling.

Exact comparison casefolds and collapses whitespace. Near-duplicate detection uses
3-token shingle Jaccard at 0.85, with deterministic 32-hash candidate retrieval in
eight bands. Candidates receive exact Jaccard verification. Candidate retrieval
is approximate and does not detect all paraphrases; **a passed audit is not proof
of zero semantic leakage**. Oversized candidate buckets fail instead of silently
truncating. Same-role near duplicates are reported and retained; future fitting
should group or weight them. Exact duplicates within a dataset/split retain one
consistent annotation set; conflicting sets are quarantined in full.

Cross-role exact/near overlap removes the lower-priority role: auxiliary, train,
validation, external evaluation, with SkillSpan test text protected above all.
Skill-XL job UUID overlap also quarantines lower-priority groups. Sentence indices
in Green and split-scoped SkillSpan document indices cannot establish global job
identity; remaining document-level leakage is a limitation. Original datasets are
never rewritten. Per-record reasons and all detected near pairs remain in local
artifacts; checked-in reports contain aggregate counts and artifact hashes.

The official SkillSpan test was evaluated during earlier extractor development.
That historical exposure cannot be undone. It is frozen for this task and cannot
serve as a pristine future final test. A later independent licensed span corpus,
or new prospectively annotated job-level holdout, is needed for a clean final
claim. This work neither rescored that test nor harvested its gold phrases.

## License and release review

The repository's MIT license covers its original code. It **does not relicense
CC BY 4.0 datasets, source advertisements, ESCO labels or model weights**. See
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) and the retained original
[Green CC BY 4.0 text](../../LICENSES/datasets/green-LICENSE.txt).
Downloaded README/license files remain byte-for-byte intact. Converted records
retain attribution and license links; experimental phrase artifacts contain the
full applicable SkillSpan/Green notices and identify their train-only sources.

Before redistributing data, dictionaries or a future trained model, manually
review the SkillSpan GitHub MIT/HF CC BY discrepancy, advertisement/platform
rights, ESCO v1.1.0 provenance, Skill-XL's sparse documentation and Russian
upstream-license claims. JobBot and explicitly unknown/unverified assets remain
excluded. No original copyright notice was removed or overwritten. No dataset
payload, upstream implementation or model weight is included in Git by this work.
