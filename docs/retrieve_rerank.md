# Experimental Retrieve → Re-rank backend

The opt-in `rerank` backend combines exact multilingual E5 retrieval, joint
cross-encoder scoring and explicit evidence checks. The existing public
`predict(profile, vacancy) -> VacancyAnalysis` contract and default `mlp` backend
remain available, as does `qwen`. Existing checkpoints, extractors, original
dataset splits and the legacy E5 implementation are preserved.

The branch began at the data-foundation commit. The inspected remote
`feature/retrieve-rerank-backend` contained no preliminary `reranker.py` to reuse.
The implementation builds on the existing interfaces rather than changing the
MLP's feature identity or retraining its downstream model.

## Pipeline and interfaces

```mermaid
flowchart LR
    A[Profile and vacancies] --> B[Known hard preference checks]
    B --> C[Exact E5 cosine Top-K]
    C --> D[Multilingual cross-encoder batches]
    D --> E[Skill and requirement evidence]
    E --> F[Ranked analyses and explanations]
```

The original pipeline encodes each pair with E5, concatenates embeddings and
structured features, and evaluates the saved MLP. The experimental pipeline
uses those same E5 content serializers and preserves its query/passage prefixes,
overlapping chunks and weighted mean pooling. `ExactRetriever` batches vacancy
chunks, reuses profile vectors, normalizes cosine scores and breaks ties by
original input index. Exact search avoids additional indexing infrastructure.

`CrossEncoder.score_pairs` jointly encodes each resume and vacancy. The selected
model is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` at
`1427fd652930e4ba29e8149678df786c240d8825`. Its pinned publisher card declares
Apache-2.0; the E5 publisher declares MIT. See
[selection, alternative comparison and provenance](rerank_model_selection.md)
for the actual architecture, checksums and unresolved upstream rights.

`build_evidence` reuses the existing extraction/normalization helper. The new
Python backend defaults to V2; the existing single-pair CLI keeps its historical
`legacy` extractor default, so pass `--extractor v2` explicitly for this experiment.
`rank` defaults to V2. A future extractor can implement the existing
`extract(text, vacancy=...)` and `normalizer` interface and provide a stable
versioned `identity`; no retrieval or reranker rewrite is required.

`RankedVacancy` includes the original input index, vacancy, public analysis,
raw cross-encoder logit, E5 cosine, evidence status, model/window diagnostics,
and component identities. `analyze_pair` returns this richer record for one pair;
`predict` returns only its `analysis`. `LocalInferenceClient` can use the backend
without an ApplyPilot API change.

## Decision policy

- Confirmed skills require supplied evidence. Job titles never establish skills.
- Unknown mandatory qualifications receive `needs_review`; absence of a mention
  is not evidence of incapacity. Optional unknowns do not block recommendations.
- Explicit mandatory denials or conflicting positive/negative claims receive
  `blocked`. One confirmed member satisfies a required alternative group; every
  option must be denied/conflicted before that group is blocked.
- Experience and seniority gates use explicit structured input fields. Bare
  years/level strings extracted from prose remain unverified diagnostic metadata;
  they can refer to unrelated facts. Clients should populate verified fields.
  Narrow, explicit textual experience/seniority requirements trigger review when
  structured fields are missing or conflict. Clear remote/office text conflicts
  also trigger review. These cues never invent candidate qualifications.
- Known remote/location conflicts are filtered before retrieval so they do not
  consume `preselect_k`. Unknown required preferences need review. Location
  comparison is exact, case-insensitive text, without geographic inference.
- Hard-blocked candidates are omitted from batch results. Remaining results sort
  by positive eligible decision first, descending raw logit, then original index.
  Review items can remain in the list with `analysis.relevant=false`; callers must
  inspect that field and `evidence_status` before treating a result as recommended.

`analysis.relevance` is the sigmoid of the raw logit to preserve the public
bounded score field. It is **not a calibrated job-fit probability**. Decisions
compare raw logits against `decision_logit_threshold` (default `0.0`) and enforce
the evidence policy independently. A score cannot override a hard conflict.
Empty positive candidate evidence or empty vacancy competency content produces
no model score and no positive decision.
Batch ranking with an empty positive candidate profile returns an empty list
without running either encoder.

## Install, download and run

Use the repository's existing CPU environment; no new framework is needed:

```bash
uv sync --frozen --extra dev --extra data
source .venv/bin/activate
resumeanalysis download-e5 --output artifacts/e5
resumeanalysis download-reranker --output artifacts/reranker
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  resumeanalysis --threads 2 predict --backend rerank --extractor v2 \
  --rerank-config configs/rerank.json --input examples/pair.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  resumeanalysis --threads 2 rank --input examples/rank.json \
  --rerank-config configs/rerank.json --preselect-k 20 --top-k 10
```

Downloads are explicit and pinned. Inference validates local manifests/checksums,
uses safetensors, disables remote code and cannot fetch missing assets. Large
weights and runtime caches remain outside Git. The full Apache license and
original model card are included with the software notices.

```python
from resumeanalysis.embeddings.encoder import TextEncoder
from resumeanalysis.inference.backends import create_predictor
from resumeanalysis.rerank.config import RerankConfig

predictor = create_predictor(
    "rerank",
    encoder=TextEncoder("artifacts/e5"),
    config=RerankConfig(model_path="artifacts/reranker"),
    cache_dir="artifacts/retrieval-cache",
)
analysis = predictor.predict(profile, vacancy)
results = predictor.rank(profile, vacancies, preselect_k=20, top_k=10)
```

The optional persistent retrieval cache has a separate namespace. Its identity
includes E5 asset hashes, tokenizer, dimensions and chunk settings; it never
rewrites legacy checkpoint/cache identities. The in-memory cache is bounded.
Caches contain derived resume/vacancy vectors: store them with the same access
controls as the underlying user data. A predictor serializes its requests for
consistent diagnostics; create separate instances for independent workers.

`configs/rerank.json` controls cross-encoder batching and length. Default
`window` covers both documents with overlapping paired windows and maximum-logit
aggregation. If the complete Cartesian window count exceeds `max_windows`,
the request fails explicitly. `error` rejects oversized pairs; explicit
`truncate` reports discarded tokens. Maximum-window aggregation is a heuristic
and can overvalue a locally relevant passage. These settings are part of model
identity and must be fixed before evaluation.

## Evaluation and validation

[Measured benchmark results](rerank_benchmarks.md) compare the existing MLP,
rule baseline, cosine-only retrieval, cross-encoder alone and the full backend.
The runner freezes validation-selected thresholds before test scoring, checks
split leakage, and uses the original records. It reports classifier metrics,
true judged-pool ranking, preselection sensitivity, initialization, CPU latency
and process peak RSS. The existing synthetic labels cannot establish hiring
quality. Its true multi-vacancy pools have only two candidates each; higher-K
measurements cannot establish wide-catalog retrieval recall.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/benchmark_rerank.py --data data/processed \
  --output artifacts/rerank-benchmark-reproduction \
  --e5-path artifacts/e5 --model-path artifacts/reranker \
  --checkpoint artifacts/run/model.pt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RESUMEANALYSIS_RUN_E5_TESTS=1 RESUMEANALYSIS_RUN_RERANK_TESTS=1 \
  pytest -q tests/test_encoder.py tests/test_rerank_model.py \
  tests/test_rerank_retrieval.py tests/test_rerank_predictor.py
ruff check .
ruff format --check .
uv build
```

The [public relevance-data review](rerank_relevance_data.md) and
[human annotation protocol](../evaluation/README.md) document the lack of an
admitted independent hiring benchmark. The provided templates have null labels;
the human evaluation runner rejects incomplete/unadjudicated data. No generated
judgments are passed off as independent annotations. Original SkillSpan test
annotations are not used for this backend.

Before promotion, collect licensed, independently adjudicated English/Russian
candidate pools; freeze a validation threshold and document policy tradeoffs;
evaluate wider pools, multilingual negation, long-document aggregation and
calibration. Preserve independent test sets through those choices. The default
backend remains the MLP until independent evidence supports a change.
