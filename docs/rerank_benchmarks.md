# Retrieve → Re-rank: measured experiment

The pretrained cross-encoder did **not** improve agreement with the current synthetic matching labels. The existing MLP achieved test Macro-F1 **0.9797**; the frozen cross-encoder achieved **0.5397**. These results support keeping the new backend experimental. They do not establish comparative hiring quality.

The checked-in [machine-readable results](rerank_benchmark_results.json) contain the exact thresholds, source hashes, identities, counts and measurements. Full per-pair diagnostics remain in ignored runtime artifacts. The final evidence policy was remeasured after software correctness fixes; model selection and thresholds use validation only.

## Records and threshold protocol

- Original partitions: **696 training / 150 validation / 148 test pairs**; the test set has 74 positive and 74 negative source labels.
- All five methods use identical ordered records. Candidate/vacancy IDs and normalized text are checked for cross-split leakage, and hashes are checked again after evaluation. The existing MLP checkpoint must identify these same partitions.
- The MLP retains its existing validation-selected threshold. The existing rule formula uses its original validation threshold grid. Cosine and cross-encoder thresholds maximize validation Macro-F1, breaking ties by precision and then higher cutoff. Each threshold artifact is written before test inference and cannot be overwritten.
- The MLP was already trained on the synthetic training partition. The cross-encoder remains frozen and was trained upstream for document relevance; it receives no resume–vacancy fine-tuning. This is a transfer baseline comparison, not evidence that one architecture is inherently better.
- Test labels do not select model weights, prompts, thresholds, policy rules, or document-window settings. Policy safeguards come from explicit input invariants and authored software fixtures. Original SkillSpan test annotations are not used.

## Binary classification on the original test pairs

Precision and recall refer to the positive class. ROC-AUC uses the continuous raw score. The pair experiment scores every supplied pair, so retrieval cannot remove a pair here; preselection is measured separately on genuine candidate pools.

| Method | Validation cutoff | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Existing E5 + MLP | 0.575000 | 0.9733 | 0.9865 | 0.9797 | 0.9996 |
| Existing weighted rule baseline | 0.500000 | 0.9024 | 1.0000 | 0.9458 | 0.9956 |
| E5 cosine only | 0.863694 | 0.6889 | 0.4189 | 0.5995 | 0.6569 |
| Cross-encoder only | -0.724527 | 0.5376 | 0.6757 | 0.5397 | 0.5720 |
| Retrieve + cross-encoder, raw score | -0.724529 | 0.5376 | 0.6757 | 0.5397 | 0.5720 |
| Full backend with structured decision policy | -0.724529 | 1.0000 | 0.1351 | 0.4681 | See note |

The full backend uses the same semantic score, so its stored score ROC-AUC remains the raw cross-encoder AUC; it is not a separate AUC for policy decisions. Policy precision/recall/F1 use the final boolean decision after mandatory evidence and preference gates. This conservative policy intentionally abstains when qualifications are unverified. Its high precision on these labels does not validate suitability or fairness.

Final evidence statuses across 148 test pairs: **{'needs_review': 97, 'blocked': 39, 'eligible': 12}**. Unknown requirements are not asserted to be missing abilities. Confusion matrices and exact thresholds are in the JSON report. Tiny raw threshold differences between batched CE-only and pairwise pipeline inference reflect floating-point batching differences; the classifiers have identical test metrics.

The CLI default raw cutoff remains `0.0`. The validation-fitted cutoffs above are experimental artifacts, not promoted defaults and not calibrated job-fit probabilities.

## Ranking on actual judged pools

The existing test records contain **138 distinct candidate profiles: 128 singleton pools and 10 pools of two judged vacancies**. Singletons are excluded from ranking metrics. Seven two-vacancy pools contain at least one positive; three are all-negative. There are no invented cross-product labels and no assumed negatives for unjudged vacancies.

Precision averages across all 10 pools. Recall, NDCG and MRR average across the seven positive-containing pools; they are undefined for all-negative pools and reported as null per query. Precision divides by `min(K, complete pool size)`, including slots lost to filtering. Recall and ideal DCG retain the complete judged pool, even when retrieval or policy removes an item. MRR@2 is full-pool MRR here.

| Method (full two-item preselection) | Precision@1 | Recall@1 | NDCG@2 | MRR@2 | Precision@2 | Recall@2 |
|---|---:|---:|---:|---:|---:|---:|
| Existing E5 + MLP | 0.7000 | 0.8571 | 1.0000 | 1.0000 | 0.4500 | 1.0000 |
| Existing weighted rule baseline | 0.7000 | 0.8571 | 1.0000 | 1.0000 | 0.4500 | 1.0000 |
| E5 cosine only | 0.5000 | 0.5714 | 0.8946 | 0.8571 | 0.4500 | 1.0000 |
| Cross-encoder only | 0.4000 | 0.4286 | 0.8418 | 0.7857 | 0.4500 | 1.0000 |
| Retrieve + cross-encoder, raw score | 0.4000 | 0.4286 | 0.8418 | 0.7857 | 0.4500 | 1.0000 |
| Full backend returned list with policy | 0.4000 | 0.5000 | 0.7517 | 0.7143 | 0.3500 | 0.8571 |

The policy ranking evaluates the actual returned list, including flagged `needs_review` items; it does not count every returned item as an automatic recommendation. The binary policy row above measures automatic positive decisions. For these two-item pools, all @5 and @10 metrics equal @2 because the denominator is clipped to the complete pool size. The JSON retains those values explicitly.

## Preselection sensitivity

| `preselect_k` | E5 preselection recall | Raw rerank NDCG@2 | Raw rerank MRR@2 | Policy-list recall@2 | Policy-list NDCG@2 | Policy-list MRR@2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5714 | 0.6038 | 0.7143 | 0.5000 | 0.5162 | 0.5714 |
| 2 | 1.0000 | 0.8418 | 0.7857 | 0.8571 | 0.7517 | 0.7143 |
| 5 | 1.0000 | 0.8418 | 0.7857 | 0.8571 | 0.7517 | 0.7143 |
| 10 | 1.0000 | 0.8418 | 0.7857 | 0.8571 | 0.7517 | 0.7143 |
| 20 | 1.0000 | 0.8418 | 0.7857 | 0.8571 | 0.7517 | 0.7143 |

The semantic preselection column isolates E5 retrieval without policy filtering. The production-policy columns execute the public `rank` method, including known hard-preference filtering before Top-K and subsequent evidence checks. Their denominators remain the original judged pools. With only two vacancies per pool, `preselect_k >= 2` exhausts every pool; this cannot establish recall at 20 in a wider vacancy catalog. The apparent quality change at K=1 versus K=2 is based on seven positive queries and is not a basis for selecting a production K.

## CPU measurements

Each method runs in a fresh process with two Torch/OMP/MKL threads, CPU-only models, batch size 8 and maximum pair length 512. No hardware or personal computer inventory is recorded. A validation-only warmup precedes scoring. Each E5 method starts with a fresh private disk cache and private retrieval memory cache. Entity-disjoint test inputs are cold on first use; repeated profiles inside a split can reuse vectors.

| Method | Initialization (s) | 148-pair test total (s) | Test total / pair (ms) | Process peak RSS (MiB) |
|---|---:|---:|---:|---:|
| Existing E5 + MLP | 3.022 | 5.141 | 34.73 | 744.6 |
| Existing weighted rule baseline | 1.043 | 5.380 | 36.35 | 746.6 |
| E5 cosine only | 2.574 | 5.761 | 38.92 | 746.7 |
| Cross-encoder only | 1.478 | 5.399 | 36.48 | 839.4 |
| Retrieve + cross-encoder, raw score | 2.871 | 14.361 | 97.03 | 1193.3 |

Initialization includes method-specific model construction and asset checks; it excludes initial common imports and source loading. RSS is the process-lifetime high-water mark, including Python, imported libraries, data and models. It is not a standalone tensor-allocation measurement.

CE-only scores all pairs in one call with internal batches; MLP/rule and full-pipeline classification score pairs individually, while cosine processes candidate groups. Total/pair is therefore an amortized cost, not a common per-request latency distribution. The JSON includes exact call counts and percentiles; CE-only has one measured batch, so its percentiles are not per-pair percentiles. These differing workloads and single-run measurements do not establish an architecture speedup.

The actual `rank` calls run after pair scoring, so their E5 cache is warm. They use the 10 true two-vacancy test pools:

| `preselect_k` | Mean rank call (ms) | p50 (ms) | p95 (ms) | Calls |
|---:|---:|---:|---:|---:|
| 1 | 52.46 | 51.97 | 56.62 | 10 |
| 2 | 90.52 | 90.15 | 95.99 | 10 |
| 5 | 87.85 | 87.99 | 91.55 | 10 |
| 10 | 89.94 | 89.38 | 95.55 | 10 |
| 20 | 99.84 | 92.42 | 146.75 | 10 |

The initial exploratory run reused a pre-existing legacy E5 cache and was excluded from published timing. All five published measurements come from the final fresh-cache run after identity/safeguard fixes. Original reports are preserved, with source paths and SHA-256 hashes in the compact report. Runtime/load variation means small timing differences should not be interpreted as stable performance differences.

## Reproduce

After the explicit model downloads documented in [the architecture guide](retrieve_rerank.md):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/benchmark_rerank.py --data data/processed \
  --output artifacts/rerank-benchmark-reproduction \
  --e5-path artifacts/e5 --model-path artifacts/reranker \
  --checkpoint artifacts/run/model.pt --threads 2 --batch-size 8 --max-length 512
```

The output directory must be new. The runner emits per-method immutable validation thresholds, full reports, `comparison.json`, and a compact `summary.json`. No training command is run and original dataset/checkpoint files are not rewritten.

## Claims still unverified and next experiments

No legally admitted independent human resume–vacancy relevance benchmark was found in the [source review](rerank_relevance_data.md). The repository includes a [human evaluation protocol, blank templates and runner](../evaluation/README.md). Those templates are software scaffolding, not labeled quality evidence.

Obtain consented/licensed independently adjudicated English and Russian pools with realistic candidate counts. Freeze validation cutoffs before testing and report raw semantic ranking separately from eligibility decisions. Then evaluate larger preselection budgets, false-positive/abstention costs, long-document aggregation, multilingual behavior, calibration and subgroup errors. Domain fine-tuning or a different reranker should be selected using training/validation data only; the existing test set must not become a tuning loop.

The current results justify preserving the MLP default and all existing checkpoints. They demonstrate a functioning experimental retrieval/reranking implementation and measurable transfer limitations, not improved hiring outcomes.
