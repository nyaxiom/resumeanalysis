# Online recommendation measurements

The implementation runs locally and restores preferences after restart. **Real
recommendation-quality improvement is not yet demonstrated.** No historical user
feedback was available. The following results measure actual E5/River execution on
explicitly authored fictional profiles, jobs and interest labels; they are not
customer outcomes or evidence of calibrated interest probabilities.

## Reproduce and inspect

```bash
uv sync --frozen --group research --extra dev --extra qwen --extra profile-llm --extra documents
source .venv/bin/activate
resumeanalysis download-e5 --output artifacts/e5
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python legacy/scripts/benchmark_online_recommender.py \
  --output artifacts/online-recommender-new-run
```

The output directory must be new. Downloads are explicit; the benchmark uses local
E5 files. The [scenario](../evaluation/online_recommendations/scenario.json),
[measured report](../evaluation/online_recommendations/report.json),
[pre-update trace](../evaluation/online_recommendations/prequential.json) and
[validation record](online_recommender_validation.md) are retained in Git. SQLite
databases, embeddings and model weights remain outside Git.

## Chronological evaluation protocol

Two fictional users, with English and Russian profile text, receive six authored
vacancy templates covering three roles and remote/office work. Each round assigns
fresh vacancy IDs to those templates, respecting the engine's one-active-label
policy. There are 32 rounds per user in each of two phases: 128 observed labels
total. The users initially prefer opposite work arrangements; the second phase
reverses those interests. This is a repeated-template behavior check, not a diverse
job-market dataset or an independent language-quality comparison.

All methods predict and rank before the next explicit label is learned. A fixed
cyclic exposure rule selects one observed vacancy per round, independently of
model rank. Only that observed label updates the corresponding user's model; the
other pool labels are used solely to compute NDCG@3. Timestamp order, user identity,
hard-constraint eligibility and duplicate user/vacancy feedback are checked.
Vacancy-ID tie-breaking matches the engine. Pools without a positive label have
undefined NDCG and are excluded from its average, not scored as zero.

The non-learning baseline assigns 0.5 to every vacancy. The content baseline uses
the engine's cold-start score. River variants blend the same content score with
their learned score using the production warm-up rule. The benchmark fixes
learning rate 0.05, L2 0.001, warm-up 10 and maximum preference weight 0.7. Its
32-event history window makes preference change observable in this short stream;
the production default remains 512. No hyperparameter search or learned projection
was performed. This evaluation measures a fixed exposure stream, not the causal
effect of the engine's exploration policy.

| Method | NDCG@3 ↑ | Log loss ↓ | Brier score ↓ | Accuracy at 0.5 ↑ |
|---|---:|---:|---:|---:|
| Non-learning, constant 0.5 | 0.5000 | 0.6931 | 0.2500 | 0.5000 |
| Content only | 0.5000 | 1.1550 | 0.3548 | 0.5000 |
| River, 47 compact features | 0.6316 | 0.7070 | 0.2519 | 0.5547 |
| River, 64 features including projection | 0.6477 | 0.7660 | 0.2784 | 0.4844 |

Loss and accuracy here use the final blended ranking score, not an independently
calibrated probability. Raw River predictions are also retained in the trace.
The projected variant ranks better on this fixture but has worse loss, Brier score
and threshold accuracy than the compact variant. Neither improves all metrics
over the constant baseline. These mixed results do not justify selecting the
projected variant as the default.

| Phase, 64 pools each | Content | River compact | River projected |
|---|---:|---:|---:|
| Initial preferences, NDCG@3 | 0.5000 | 0.7302 | 0.7624 |
| Reversed preferences, NDCG@3 | 0.5000 | 0.5330 | 0.5330 |

Performance drops after the preference reversal. A recent-history window and
constant learning rate permit adaptation but do not eliminate lag. The data do
not establish that the default window or learning rate is appropriate for real
users. Collect consented explicit feedback and evaluate future events
chronologically before tuning these settings or claiming recommendation uplift.

## CPU time and memory

Measured on 2026-10-27 with the locked environment, CPU inference, two Torch
threads and Torch 2.14.1+cpu. E5 initialization took **1.152 seconds**. Peak process
RSS was **749.87 MiB**; VRAM use was zero. RSS includes Python, E5, caches and both
feature variants. It is not isolated per-user model memory.

| Operation | Compact mean | Projected mean | Observations per variant |
|---|---:|---:|---:|
| Build features with warm E5 cache | 1.081 ms | 1.596 ms | 12 |
| Recommend one vacancy, warm cache, including snapshot write | 3.151 ms | 3.470 ms | 20 |
| Record feedback and commit transaction | 2.560 ms | 2.807 ms | 20 |
| River prediction only | 0.00643 ms | 0.00784 ms | 200 |
| River update with prequential metrics | 0.03810 ms | 0.04665 ms | 200 |
| Reopen SQLite, replay 20 labels and recommend | 7.427 ms | 7.391 ms | 1 |

The compact feature first pass averaged 12.754 ms, with shared profile/template
caching already occurring within that pass. The projected first pass reused the
compact pass's E5 cache and therefore is not an independent cold-inference timing.
Restore timings exclude E5 initialization. The report retains p95 values and
counts; these small, single-run samples are observations, not latency guarantees.

The compact representation has 47 coordinates. The enhanced representation has
64, including 16 fixed projected E5 coordinates and an availability indicator.
The projection matrix occupies 49,152 bytes per feature builder. No full
384-coordinate embedding is passed to River. Linear prediction/update cost grows
with the feature count; additional coordinates also increase the number of weights
to estimate from sparse personal feedback. The synthetic comparison supplies no
reliable statistical justification for the larger representation.

SQLite occupied 172,032 bytes for compact features and 212,992 bytes for projected
features after 20 events, including schema and snapshots. These are small-example
file sizes, not a per-event storage guarantee. Replay is bounded by the training
window, but current repository history reads and retained snapshots grow with
total history. Large-history performance and retention need separate validation.

## End-to-end behavior and remaining evidence

The [separate-process CLI demonstration](../evaluation/online_recommendations/cli_demo.json)
used actual local E5, recorded a dislike, then reopened the application. The
rejected vacancy remained excluded and the learned preference score was restored.
No personalized pretrained checkpoint was required. Automated tests also cover
corrections, reset/replay, transaction failures and independent users.

Live Ollama extraction, real ApplyPilot database/API deployment, real-user ranking
quality, probability calibration and large-scale throughput are not validated by
this benchmark. Existing qualification and cover-letter quality limitations remain.
