# Skill extractor v2: measured validation

Date: 2026-10-24. Branch: `feature/skill-extractor-v2`, based on
`feature/qwen-instruct` at `ad45d07eb220dfa3e68083910d6aa0c47750fd21`.
Neither the parent branch nor `main` was modified. The original extractor remains
the application default. The feature commit and PR are supplied in the completion
message, rather than embedding a self-referential commit hash here.

## Selection and licensing

All four requested repositories were investigated before implementation. The
[research matrix](skill_extractor_research.md) and
[evidence index](skill_extractor_evidence.json) retain exact commits, actual terms,
maintenance/dependency findings, asset sizes and explicitly unmeasured claims.

The selected v2 implementation uses original curated aliases and contextual rules,
the existing normalizer, and no new dependencies or pretrained NER weights. No
SkillNER/JobBot implementation or EMSI dictionary was copied. SkillNER code is MIT,
but its separate dictionary terms remain unresolved. JobBot and the linked JobBERT
weight repositories lack verified applicable grants and were excluded from execution.
Their published figures were not compared as if measured on the same SkillSpan data.

The optional phrase artifact uses the actual MIT SkillSpan training annotations,
copyright 2023 Mike Zhang, with original notice embedded. It is generated locally,
not committed. The root MIT and existing notices are preserved; underlying
advertisement/data-redistribution rights remain manual review items.

## Full SkillSpan benchmark

The baseline was reproduced **before implementation**. The new evaluator also
reproduces its exact counts, on all **3,569** pinned English test sentences and all
SKILL/KNOWLEDGE annotations. No vocabulary was harvested from test errors or tuned
after observing v2 test scores.

Source: `kris927b/SkillSpan`, commit
`2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b`; test SHA256
`8438eda34939a5877b984ea5daf9e6ff39b2d5c7e3d73cca0499d3de3784d24a`.
Exact-span matching uses the historical character-to-token overlap mapping and
requires identical predicted/gold token boundaries. Token metrics compare unions
of covered token indices. Requirement categories are not SkillSpan annotations.

| Extractor | Exact precision | Exact recall | Exact F1 | TP / FP / FN |
|---|---:|---:|---:|---|
| Legacy | 0.606452 | 0.083002 | 0.146019 | 188 / 122 / 2077 |
| Plain v2 | 0.609489 | 0.147461 | 0.237469 | 334 / 214 / 1931 |
| V2 + training phrases | 0.225434 | 0.327152 | 0.266931 | 741 / 2546 / 1524 |

| Extractor | Token precision | Token recall | Token F1 |
|---|---:|---:|---:|
| Legacy | 0.790560 | 0.042675 | 0.080979 |
| Plain v2 | 0.804478 | 0.085828 | 0.155108 |
| V2 + training phrases | 0.442781 | 0.292038 | 0.351948 |

Plain v2 improves measured recall/F1 on this corpus while retaining approximately
the baseline precision. Its recall remains low: it still misses most annotated
skills. These results do not establish general recruitment or Russian accuracy.

The phrase variant has a **severe precision loss** and also fails some controlled
requirement fixtures. It remains explicitly experimental and is not a default or
a recommended production upgrade. Its higher F1 does not remove that risk. It uses
450 phrases from the exact pinned 4,800-row training split, with fixed min-frequency
2, max-length 12 and declared ambiguity/disjunction filters. No neural model or
SkillSpan test labels were used to construct it.

## Exact split overlap and separate diagnostic subset

An explicit audit found that the official splits contain exact normalized sentence
overlap. Normalization joins original tokens with one space and applies casefold;
it does not remove punctuation, stem or perform semantic similarity.

| Split | Rows | Unique sentence texts | Internal duplicate rows |
|---|---:|---:|---:|
| Train | 4800 | 3943 | 857 |
| Validation | 3174 | 2597 | 577 |
| Test | 3569 | 2660 | 909 |

Train/validation share 159 unique texts, train/test 177, and validation/test 169.
This is especially relevant to a training-derived lexical variant. A separate
diagnostic excludes **916** test rows whose text appears in train or validation,
retaining **2,653**. Every original test row has a recorded retained/excluded outcome;
all counts reconcile. Original sources, labels and the full benchmark are unchanged.

Filtered SHA256:
`d4ccd85cc6ffa2a252c39d943d83ee1d4b95634bff753220212336b6355d1282`.

| Extractor, same filtered rows | Exact precision | Exact recall | Exact F1 |
|---|---:|---:|---:|
| Legacy | 0.606667 | 0.084612 | 0.148511 |
| Plain v2 | 0.610902 | 0.151093 | 0.242266 |
| V2 + training phrases | 0.228680 | 0.309159 | 0.262898 |

This subset still has 249 internal duplicate rows (2,404 unique texts). Removing
exact overlaps does not establish source-job, cluster, paraphrase or semantic
independence. It is a separate diagnostic, not a replacement benchmark or a claim
of comprehensive leakage removal. The phrase precision problem persists.

```bash
resumeanalysis extractors split-audit \
  --train data/external/skillspan/data/json/train.json \
  --validation data/external/skillspan/data/json/dev.json \
  --test data/external/skillspan/data/json/test.json \
  --output-dir artifacts/experiments/skill_extractor/split-audit
resumeanalysis extractors compare \
  --input artifacts/experiments/skill_extractor/split-audit/filtered-test.jsonl \
  --output artifacts/experiments/skill_extractor/filtered.json
```

## Controlled requirements and Role Radar

Thirty manually authored fixtures separately grade mandatory, optional, mentioned,
negated and alternative requirements. They are regression fixtures, not an
independent real-world language benchmark.

| Extractor | English: all graded categories correct | Russian: all graded categories correct |
|---|---:|---:|
| Legacy | 7 / 19 | 3 / 11 |
| Plain v2 | 19 / 19 | 11 / 11 |
| V2 + training phrases | 17 / 19 | 10 / 11 |

The user-supplied Essential Skills excerpt produces zero independent mandatory
skills with legacy and all **15** named technologies with plain v2. This is one
provided diagnostic, not an estimate of Role Radar dataset accuracy. The local
106-pair `inference_pairs.jsonl` file is absent and was not reconstructed or claimed
as evaluated. Its runner supports auditable extraction and optional matching-model
scores with a compatible checkpoint; source composite scores are never gold labels.

## CPU performance

Measured on Python 3.12 with CPU PyTorch, sequential fresh subprocesses per extractor,
configured Torch threads 2 and no CUDA. Values are local observations, not hardware
requirements or guaranteed throughput.

| Extractor | Initialization | Mean extract call | p95 extract call | Process peak RSS |
|---|---:|---:|---:|---:|
| Legacy | 1.38 ms | 0.0620 ms | 0.1517 ms | 365.59 MB |
| Plain v2 | 12.17 ms | 0.1686 ms | 0.4267 ms | 365.42 MB |
| V2 + training phrases | 21.29 ms | 0.2762 ms | 0.7177 ms | 366.23 MB |

Plain v2 costs about 2.7 times the legacy extraction-call time in this run. All
measured paths used about one CPU core during evaluation. Process RSS includes
Python/PyTorch imports, input and report overhead; it is not the extractor's exclusive
allocation, and the small RSS differences do not show that v2 uses less memory.
Full path wall times including parsing/metrics were about 0.479 / 0.917 / 1.339 seconds.
There is no external NER/model download in these measurements. Model/dictionary
artifact sizes for excluded candidates are research metadata, not measured RAM.

## Downstream E5 + MLP ablation

The original `artifacts/run/model.pt` remains byte-for-byte unchanged. Plain v2 was
actually retrained with the original CPU configuration and seed 42, same original
696/150/148 train/validation/test partitions, and a separate output directory
`artifacts/skill-extractor-v2/run`. It saved format 2, pipeline version 2 and the
matching extractor identity; best epoch was 63. No threshold was fitted on test.

All four conditions use the same 148 ordered original test pairs and profiles.
Description-only clears only the two structured vacancy skill lists; removed values
and provenance are retained in the report. Original structured requirements are
used exactly as supplied for the explicit conditions. No new fit labels or requirements
are inferred. Checkpoints must match all original partition hashes.

| Condition | Source-label agreement accuracy | Source-label agreement macro-F1 |
|---|---:|---:|
| Legacy + original model, description only | 0.587838 | 0.560360 |
| Plain v2 + separately retrained model, description only | 0.587838 | 0.560360 |
| Legacy + original model, supplied requirements | 0.979730 | 0.979729 |
| Plain v2 + separately retrained model, supplied requirements | 0.979730 | 0.979729 |

**No downstream score improvement was measured on this synthetic dataset.** Its
structured fields and heuristic labels differ from realistic vacancy extraction.
Agreement under information-removal interventions is descriptive, not independently
validated classification quality. Synthetic source labels do not establish real
recruitment accuracy. Per-record scores, decisions, evidence, label disagreements,
wall/process CPU and RSS are saved in the local ablation report. Shared-process
ordering and embedding caches affect ablation latency; it is not a speed ranking.

The original saved E5 neural/logistic/rule metrics reproduce exactly. Original
extractor, normalizer, taxonomy, network, model checkpoint and baseline evaluation
implementation are retained; supporting compatibility/training/feature plumbing
was extended without changing the default legacy vector or decision semantics.

## Reproducibility, tests and artifacts

The full automated suite passed **219 tests**, with downloaded E5 integration
enabled. It covers original regressions, thirty EN/RU fixtures,
span/alias/context boundaries, alternatives as feature units, authoritative explicit
skills, malformed/long input, versioned training/cache/checkpoint failures, phrase
source/notice integrity, four-way ablation, split-audit reconciliation, notebook
structure, optional Qwen integration mocks and successful console exit behavior.

All notebook code cells were executed on the actual 3,569-row test file, and in an
explicit two-row authored smoke mode. Committed notebook outputs remain empty.
No pretrained Qwen/NER quality or real Role Radar performance was verified.

Local ignored artifacts include:

- `artifacts/experiments/skill_extractor/final-curated.{json,html}`
- `artifacts/experiments/skill_extractor/final-train-phrases.{json,html}`
- `artifacts/experiments/skill_extractor/final-fixtures-{legacy,v2,train-phrases}.json`
- `artifacts/experiments/skill_extractor/train-phrases-final.json` and metadata
- `artifacts/experiments/skill_extractor/split-audit/`
- `artifacts/experiments/skill_extractor/filtered-{curated,train-phrases}.{json,html}`
- `artifacts/experiments/skill_extractor/ablation-final.json`
- `artifacts/skill-extractor-v2/run/` for the separate retrained checkpoint/metrics

Plain v2 extractor identity:
`rules-v2:413645e1ead9f301e90be135e26529b98382174fb0245b878b5123cac7dea123`.
Phrase variant identity:
`rules-v2:5e452f218ede7b6def0b218571e6691df1355ca3a9e5816ac75ee714f81ddb18`.
Original extractor source SHA256:
`ee89e4804cc7871da02afe422b21b0a25e71a161726b948cc96bfeb08217000d`.

New files: `extraction/{registry,vocabulary,improved,phrase_lexicon,cli}.py`,
`experiments/{extractor_evaluation,extractor_ablation,skillspan_splits}.py`,
the comparison notebook, six extraction/evaluation test modules and controlled
fixture JSON, and four `docs/skill_extractor_*.{md,json}` documentation/evidence files.
Modified: main CLI, feature builder, predictor, training compatibility, postprocessor,
Qwen predictor/CLI injection, README, third-party notices and license-audit addendum.
The original extractor and dataset schemas are unchanged; dependencies/lock are unchanged.

See [Arch Linux commands and implementation limits](skill_extractor_v2.md).
Remaining work includes licensed independent EN/RU labels, semantic/job-cluster
duplicate auditing, harder linguistic scope cases and any future licensed NER
comparison. The optional phrase variant needs substantial precision work before
production consideration. Existing data/model/taxonomy redistribution review items
remain in the preserved license audit.
