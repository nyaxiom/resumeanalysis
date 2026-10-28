# Reranking delivery validation

Validated on 2026-10-26 before creating the separate cover-letter feature.

- Full suite with both downloaded-model flags enabled: **509 passed**, no skipped tests.
- Ruff checks and formatting passed; Git whitespace checks passed.
- Source distribution and wheel built successfully. The wheel contains 75 license
  files, including byte-identical retained reranker notices, and no model weights.
  Setuptools reports an existing deprecation warning for the `+` in the Torch
  license filename; that original notice was preserved.
- Real offline CLI single-pair prediction and batch ranking completed. The authored
  batch example filters the incompatible remote preference and marks the unknown
  Kubernetes requirement for review.
- All 16 baseline checksums remained unchanged: legacy model/feature/extractor
  implementation, both saved MLP checkpoints, original matching partitions and
  the frozen SkillSpan test file. No matching model was retrained.
- All five final benchmark methods ran in fresh processes with private caches.
  Exact results and limitations are in [the benchmark report](rerank_benchmarks.md).

The full verification command was:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RESUMEANALYSIS_RUN_E5_TESTS=1 RESUMEANALYSIS_RUN_RERANK_TESTS=1 pytest -q
ruff check .
ruff format --check .
python -m build --no-isolation
```

Integration tests required local asyncio socket notifications. The managed
sandbox blocks those notifications; the minimal `asyncio.to_thread` reproduction
also stalled there. Tests passed with that local restriction lifted. Actual model
smoke tests additionally monkeypatch network connections to fail, verifying offline
inference independently of the sandbox.

## Created and modified files

| Area | Files |
|---|---|
| Runtime | `src/resumeanalysis/rerank/{__init__,assets,config,evidence,model,predictor,retrieval}.py` |
| Integration changes | `src/resumeanalysis/inference/backends.py`, `src/resumeanalysis/cli.py` |
| Evaluation | `src/resumeanalysis/experiments/rerank_evaluation.py`, `rerank_human_evaluation.py`; `scripts/benchmark_rerank.py`, `scripts/evaluate_rerank_human.py` |
| Configuration/example | `configs/rerank.json`, `examples/rank.json` |
| Tests | `tests/test_rerank_{cli,evaluation,evidence,human_evaluation,model,predictor,retrieval}.py` |
| Human evaluation | `evaluation/README.md`, grouped schema, blank JSONL and assessor CSV under `evaluation/rerank/` |
| Documentation | `docs/retrieve_rerank.md`, `docs/rerank_{model_selection,benchmarks,relevance_data,validation}.md`, model/data evidence JSON and measured benchmark JSON |
| Notices | Additions to `THIRD_PARTY_NOTICES.md`, original card and Apache license in `LICENSES/references/reranker-*` |
| Overview | Additive README links; MLP remains default |

No changes were merged into main. Existing PRs #4 and #5 remain separate.
