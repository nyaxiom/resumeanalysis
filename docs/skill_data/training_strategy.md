# Controlled training experiments

The implemented data experiment compares frozen legacy/V2 extractors with two
train-only expanded-vocabulary baselines. No neural extractor or downstream
resume/vacancy matching model was retrained. The following plan keeps incompatible
supervision separate and uses validation for selection.

## Comparison matrix

| Experiment | Fitting data | Selection/evaluation | Purpose |
|---|---|---|---|
| L: legacy | Existing original curated vocabulary; no fitting | Same audited SkillSpan validation as all rule baselines | Preserve the operational default and quantify its coverage gap. |
| V: V2 | Existing original curated vocabulary/rules; no fitting | Same validation; authored EN/RU fixtures as regression constraints | Measure the existing replacement independently of new training labels. |
| D1: expanded vocabulary | Eligible SkillSpan training spans only; distinct-sentence frequency ≥2 | Same validation; no threshold sweep was performed in this task | Test whether literal lexical expansion raises recall and at what precision cost. Implemented and measured. |
| D2: real auxiliary vocabulary | Same SkillSpan training plus Green training **Skill** spans only | Same validation; external gold supplies no phrases | Test real-data expansion under a fixed matcher. Implemented; worse F1 despite higher recall. |
| N1: lightweight NER | New token-classification heads on pinned generic DistilBERT; audited SkillSpan train only | Train loss for fitting, SkillSpan validation exact F1 for checkpoint/hyperparameter selection | Test contextual boundaries and unseen surface generalization instead of adding isolated dictionary matches. Proposed. |
| N2: real auxiliary NER | N1 plus Green training Skill spans under a separate masked auxiliary objective | Same validation protocol and budget as N1 | Test whether additional real domain diversity helps after controlling for model, sampling and label scheme. Proposed. |

For N1/N2 use
[`distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be`](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be),
whose actual Apache-2.0 LICENSE was verified. It is a roughly 67M-parameter generic
English encoder, not a pretrained skill extractor. No latency, memory footprint,
accuracy gain or Russian support is claimed before measurement. Preserve its
license, attribution and modification notices in a later derivative release;
review pretraining provenance and inherited biases. The research report retains
the exact license hash. Exported SkillSpan-trained JobBERT models have unresolved
asset licenses and possible benchmark exposure, so they are not substituted.

## Label and token alignment

Use separate BIO SKILL and KNOWLEDGE heads on SkillSpan so overlapping layers are
not destroyed. Evaluate their span union for comparability and retain separate
layer metrics. Convert source tokens with `is_split_into_words`/word IDs, compute
loss on first subtokens only, and mask padding/special tokens. Map predictions
back through the preserved token offsets; never reuse offsets from a transformed
string. Use 256-token windows with stride 64 and deterministic overlap merging;
report all overflow coverage and boundary effects instead of silently truncating
long skills. Original source rows and document grouping remain traceable.

Green **Skill** can provide an auxiliary skill objective, but its Domain,
Occupation, Qualification and Experience entities remain distinct. They are not
automatically SkillSpan KNOWLEDGE. Mask Knowledge-head loss on all Green data;
mask non-Skill Green entity targets in the auxiliary skill loss rather than
turning them into false negatives for another annotation policy. Keep source
sampling balanced so the larger Green training set does not silently dominate.
Repeat N1/N2 with the same budget and report a SkillSpan-only size-matched control
or learning curve before attributing a gain to dataset diversity.

Sentence-level TECHWOLF, Synthetic ESCO and Skill-XL labels do not enter BIO
training. A later optional sentence-label head or skill-normalization objective
must use its own loss, vocabulary/version, splits and metrics. Unannotated
incidental skills cannot be assumed negative, and relevant-but-unlabeled Skill-XL
rows cannot be converted into empty-skill negatives. No blanket concatenation or
automatic gold string-position search is allowed.

## Training and model selection

The proposed bounded grid is in
[`configs/skill_data_experiments.json`](../../configs/skill_data_experiments.json):
learning rates 3e-5/5e-5, three epochs, seeds 42/43/44, weight decay 0.01. Log data
artifact hashes, source policies, model/tokenizer revision, all seeds, sampling,
loss masks, window merging and every attempted configuration. Select by mean
validation exact span F1; report precision/recall and seed spread, with a
predeclared tie-break favoring precision then smaller runtime. Do not select by
held-out test score. Vocabulary-frequency or context-threshold changes in future
work must likewise be selected on validation and logged as additional attempts.

Report exact token and character boundary metrics, per-layer results, known and
unseen training-surface recall, one/two-to-three/four-plus-token mentions, the
declared contextual proxy, and negative-sentence false positives. Known/unseen
must be defined from training only. Bootstrap by verified source document group
where possible; use sentence-level uncertainty explicitly where document IDs are
unavailable. Measure initialization, CPU latency and runtime memory only after
running the model under a documented environment. Do not infer them from download
size or parameter count.

## Independent evaluation and Russian coverage

Keep Green's evaluation partition reserved for **Skill-only** span transfer after
annotation-guideline review. TECHWOLF and Skill-XL can provide reserved external
classification/normalization checks after explicit ESCO mapping and completeness
review; they cannot provide exact span accuracy. TECH/HOUSE inherit SkillSpan text:
their validation is quarantined in full by the original SkillSpan test guard, and
their tests remain sealed. They are not independent extraction evaluation data.

The original SkillSpan test stays frozen. Historical prior use means a future
clean final claim needs a new independently licensed, job-level holdout. Freeze
the chosen model/configuration and evaluation protocol before any such final
evaluation. External evaluation gold must never become dictionary entries or a
second selection set after looking at its scores.

For Russian, first obtain or commission a licensed real-job corpus with human
skill spans, original text/offsets, job/source IDs, annotation guidelines and
double annotation/adjudication. Split by vacancy/source/time **before** model
selection, freeze a holdout, and audit cross-lingual/template duplication. Include
Russian procedural/soft skills and mixed-script technology names. The existing
title tags and model-extracted skill lists can inform collection strata after
rights review, but do not supply span gold or extractor accuracy. A Russian base
such as `cointegrated/rubert-tiny2` is a later candidate after its card-only MIT
notice/provenance and actual Russian task performance are clarified.

All neural artifacts must use a new experiment directory and extractor identity.
The existing checkpoints and legacy default remain untouched. A future extractor
release needs a separate decision about downstream feature compatibility; this
task does not retrain or replace the matching model.
