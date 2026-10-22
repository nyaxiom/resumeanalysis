# Completed evaluation

Date: 2026-10-22. This is an exploratory smoke experiment on synthetic texts, not a
validation of hiring decisions. Results come from executed commands; no LLM was used
as an evaluator. Machine-readable reports:
[cpu_smoke.json](experiments/cpu_smoke.json), [skillspan.json](experiments/skillspan.json).

## Data and training

Downloaded Candidate Matching Synthetic parquet contains 10,000 profiles, 2,500 jobs
and 2,500 positive lists. Sampling selected 250 jobs, up to two positives and two
hard negatives per job, yielding 994 pairs: 696 train, 150 validation and 148 test.
Classes are balanced; test has 74 positives and 74 negatives. Both entity IDs and
normalized profile/job texts are disjoint across splits. The JSON report records
hashes, seed, source revision and the negative policy.

Source texts are Qwen-generated and positive lists are heuristic. Additional negatives
are weak labels for less than 60% required-skill coverage after structured and text
evidence are combined. They are not employer rejections. The closest eligible
sub-threshold profiles become hard negatives; arbitrary unlisted pairs are not
assigned a negative label.

The pretrained `intfloat/multilingual-e5-small` encoder was frozen, revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, with profile/query and job/passage prefixes.
Only MLP and Logistic Regression were trained. MLP: 1,553 → 128 → 32 → 1, GELU,
Dropout 0.2, AdamW learning rate 0.001, weight decay 0.001, batch 64, seed 42,
BCEWithLogitsLoss. Best validation loss occurred at epoch 63; patience 10 stopped
training after epoch 73. Parameters changed. MLP and LR use the same representation;
the LR scaler is fitted only on training data.

Validation Macro-F1 selects thresholds: rule 0.5, logistic 0.325, neural 0.575.
The MLP checkpoint is selected by validation loss; test data is not used by the
training/early-stopping code. Development corrected contradictory negative labels
that ignored text evidence, so this is an exploratory pipeline check rather than
a sealed external model-selection benchmark.

## Final test results

| Method | Positive precision | Positive recall | Macro-F1 | ROC-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---|
| Rule-based | 0.9024 | 1.0000 | 0.9458 | 0.9956 | 66 / 8 / 0 / 74 |
| Logistic Regression | 0.8625 | 0.9324 | 0.8917 | 0.9680 | 63 / 11 / 5 / 69 |
| ResumeAnalysisNet | 0.9733 | 0.9865 | 0.9797 | 0.9996 | 72 / 2 / 1 / 73 |

The MLP performs better on this sample. Other seeds, independent real fit data and
confidence intervals have not been evaluated. High ROC-AUC is plausible for a synthetic
task constructed around skill coverage; it cannot establish hiring performance.
Calibration has not been checked, and a sigmoid score saturated at 1.0 does not mean
100% certainty.

Reloading saved MLP/LR weights reproduced every test metric exactly. CLI inference
on `examples/pair.json` ran with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`:
Python/SQL were confirmed, Airflow remained unknown, and the experience requirement
was satisfied. Public Python `predict()` was also checked. Reserved fields are null.

## Error analysis

The MLP has two false positives with required coverage 0.5 and semantic similarity
0.8580 / 0.8454. One covers an optional skill; both meet the specified seniority.
Their scores, 0.6849 / 0.6789, exceed the validation threshold of 0.575.
The false negative has required coverage 0.6, optional coverage 0, lower seniority,
semantic similarity 0.8509 and score 0.5256. The published list nevertheless marks it
positive. These cases expose differences between skill-rule labels and multi-feature ranking.

The rule baseline has eight false positives; LR has eleven false positives and five
false negatives. A high-dimensional linear representation did not improve on the
simple formula in this small experiment; an ablation would be needed to attribute
the difference to particular feature blocks.

`artifacts/run/errors_{rule,logistic,neural}.json` contains IDs, labels, scores,
provenance and feature values without full resume texts. Errors were not repaired
by tuning the threshold on test data or by generative explanations.

## Skill extraction

The complete SkillSpan test set has 3,569 rows. Gold SKILL and KNOWLEDGE spans are
combined, and the dictionary was not expanded from test annotations.

| Level | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---|
| Exact span | 0.6065 | 0.0830 | 0.1460 | 188 / 122 / 2077 |
| Token overlap | 0.7906 | 0.0427 | 0.0810 | 268 / 71 / 6012 |

The finite technology dictionary misses long professional/soft-skill spans,
languages and unknown technologies. Its low recall is a substantial MVP limitation;
it is not equivalent to Nesta's NER. Token and exact-span metrics have different
denominators. Russian rules have unit-test coverage, but no annotated Russian
benchmark has been evaluated.

## Development checks

* `uv sync --frozen --extra dev --extra data`: repeatable dependency installation.
* `RESUMEANALYSIS_RUN_E5_TESTS=1 pytest -q`: 47 tests, including pretrained E5.
* `ruff check .`, `ruff format --check .`: lint and formatting.
* `uv build`: wheel and sdist, containing all Python submodules.
* Repeated preparation produces identical hashes. Entity/text leakage checks pass;
  generated negatives do not reach 60% evidence coverage.
* Saved-model evaluation, offline CLI inference and public `predict()` pass.

GitHub Actions checks Python 3.11/3.12. Full-data training, NER training,
med2425 payload execution, calibration and restoration in a new cloud task have
not been verified. License notices and asset-rights limits are documented separately
in [the compliance audit](license_audit.md).
