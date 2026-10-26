# Prospective human evaluation of resume–vacancy ranking

No independent human relevance benchmark was admitted during the
[source review](../docs/rerank_relevance_data.md). This protocol and the
[blank grouped template](rerank/annotation_template.jsonl) are deliverables for a
future real evaluation. The checked-in examples are independently authored
format examples with **null judgments**, not human hiring ground truth.

## Freeze the collection and splits before looking at model results

Recruit consenting candidates or obtain a dataset-specific grant permitting the
intended use of résumé and vacancy text. Record source URLs, source dates,
license/permission evidence, withdrawal policy and content hashes. Retain raw
permissions and personal data outside Git under appropriate access controls.
Remove direct identifiers from evaluator views and do not use protected attributes
as relevance criteria. A publicly visible résumé is not by itself permission.

Start with a prospective pilot of at least 100 English and 100 Russian profiles,
with 20–50 vacancies per profile. These are **collection targets**, not existing
sample counts or statistical power claims. Include mixed-language inputs,
different occupations/seniority, sparse résumés, long text, explicit skill denials,
alternative requirements and hard remote/location preferences. Report the final
sampling frame and counts rather than presenting it as representative of all jobs.

Build each pool using a model-independent vacancy sample plus a blinded union of
retrieval candidates from all compared systems and independent hard negatives.
Include plausible zero-relevance pools. Freeze the sampled pool, original source
IDs and the pool-construction seed before annotation/model selection. Every system
receives exactly the same profile and pool. Do not invent a labeled cross product
by assuming that unobserved jobs are irrelevant. Recall measured on a sampled
pool is **pool recall**, not full labor-market recall.

Assign connected components of candidate, vacancy, normalized text, source job,
template/augmentation family and near-duplicate links to train/validation/test
before fitting thresholds or choosing prompts. Use source/time boundaries where
available. For a strict cold-entity evaluation, no profile or job may cross a
boundary; if a shared job catalog is intentional, label that separate evaluation
explicitly. Freeze a test manifest and hashes. Do not view test labels to choose
the Cross-Encoder, length policy, `preselect_k`, decision threshold or hybrid weight.

## Blinded annotation and adjudication

Two qualified, language-fluent assessors independently read each profile/vacancy
pair in randomized order without model scores, model names, generated explanations
or another assessor's label. Provide the same task rubric and short pilot training
examples. They judge **document-supported relevance for considering an application**,
not employability, protected traits or an ultimate hiring decision:

| Grade | Meaning |
|---|---|
| `2` | Strong document-supported alignment with the job's main responsibilities and explicit mandatory requirements; no established hard-preference conflict. |
| `1` | Plausible/partial alignment, with unresolved requirements or meaningful uncertainty that warrants clarification. An unmentioned skill is unknown, not proven absent. |
| `0` | Clear document-supported mismatch, explicit unsatisfied hard requirement or an established hard-preference conflict. Record the evidence; do not use mere absence of a skill as a contradiction. |
| `null` | Not judged or abstained: insufficient information, language/occupation expertise gap, unresolved source issue, or irreconcilable ambiguity. Supply `abstain_reason`; never coerce this to `0`. |

Record a separate semantic rationale and hard-preference compatibility
(`compatible`, `conflict`, `unknown`), required/optional/alternative skill evidence,
explicit denials, unknown experience/seniority and remote/location uncertainty.
Do not infer skill possession from a title or count optional/alternative terms as
independent mandatory requirements. A relevant semantic topic does not override
an explicit contradiction or a candidate's hard preference.

Retain both original labels and literal evidence references in private annotation
logs. A third qualified assessor adjudicates disagreements using the same source
text; mark `adjudicated=true` only after completion. `assessor_ids` are pseudonyms,
not names/emails. Report agreement before adjudication (weighted Cohen's kappa
where two assessors have overlapping judgments), grade distributions, abstention
rates/reasons and adjudication rates by language and scenario. Agreement is a
quality check, not proof that the rubric measures actual hiring success.

## Machine-readable records

One JSONL record represents a profile query and its fixed vacancy pool. Validate
it against [grouped_annotations.schema.json](rerank/grouped_annotations.schema.json).
The top-level `profile` and each `vacancy` use the existing public schemas. Store
one canonical adjudicated `grade` per candidate, retaining raw assessor logs
separately; do not replace them with the aggregate value. Source provenance must
identify both profile and vacancy text. `judged_pool_complete=true` means all
members in the declared pool have a non-null, adjudicated grade; it makes no claim
about jobs outside that pool. `data_origin=authored_fixture` can only support
software/format tests, never an independent human relevance claim.

The evaluation runner must reject null, abstained or non-adjudicated judgments as
gold instead of silently skipping them. An intentionally partial collection may
be inspected for coverage, but final ranking scores require a fully adjudicated
fixed pool or a separately documented incomplete-qrels metric. Unjudged records
must remain in the annotation ledger so all counts reconcile.

## Evaluation and promotion criteria

Select thresholds and all model/configuration choices on validation only. Preserve
the selected artifact and input/config/model identities before the final test.
Evaluate all systems on identical ordered queries, with deterministic input-order
ties. Binary positives use grade ≥1; also report grade-2-only sensitivity and
graded NDCG with gain `2**grade - 1`. Report Precision/Recall/Macro-F1 and ROC-AUC
where both classes exist, plus Precision@K, Recall@K, NDCG@K and MRR. State whether
all-negative queries receive zero ranking credit or are excluded from each
denominator, and report the counts either way.

Measure retrieval candidate recall across predeclared `preselect_k` values and
score final rankings against **all judged relevant jobs in the full frozen pool**,
not only the retrieved subset. Report preference/contradiction violations and
unknown-evidence rates separately. Measure CPU initialization, cold/warm latency
and isolated-process peak memory without claiming probability calibration. Bootstrap
by independent query/source component, and report per-language/occupation results
with confidence intervals and sample sizes. Do not tune on language subgroups of
the final test after seeing their outcomes.

No promotion is justified by synthetic-label accuracy alone. A prospective result
should demonstrate improvement under the frozen rubric without increased hard
preference or contradiction violations, with uncertainty and operational costs
reported. Until then, the reranking backend remains opt-in and actual hiring
quality, fairness, calibration and English-to-Russian transfer remain unverified.

## Reproducible runner for completed human annotations

The runner is ready; the repository does **not** yet contain admissible human
judgments. It deliberately rejects `rerank/annotation_template.jsonl`. Do not
replace its null labels with model judgments. Collect and audit real, permitted,
independently assessed examples first, retaining all splits in one private
`human-pools.jsonl` so identity, source-document and normalized-text leakage can be
checked together.

Export a separate prediction JSONL for each model and split. Each line uses this
format (IDs below illustrate the format and are not a real evaluated record):

```json
{
  "schema_version": 1,
  "query_id": "query-id-from-human-pools",
  "model_identity": "pinned-model-and-weight-checksum",
  "config_identity": "hash-of-frozen-serialization-extractor-policy-and-model-options",
  "candidates": [
    {"vacancy_id": "job-1", "score": 2.4, "retrieval_score": 0.82, "decision": true, "evidence_status": "eligible"},
    {"vacancy_id": "job-2", "score": -1.2, "retrieval_score": 0.41, "decision": false, "evidence_status": "blocked"}
  ],
  "retrieved_vacancy_ids": ["job-1", "job-2"],
  "ranked_vacancy_ids": ["job-1"]
}
```

Write one compact object per line. `candidates` must contain **every** judged
vacancy, in the original pool order, with a finite numeric semantic score. For
pair classification, score the complete pool in a separate diagnostic pass;
record the actual retrieval/ranking subsets in `retrieved_vacancy_ids` and
`ranked_vacancy_ids`. Never invent scores for unretrieved pairs. The diagnostic
pass does not establish the runtime cost of Top-K ranking. If a model cannot
score the complete pool, this runner cannot report its complete-pool pair
classification/AUC; use a separately documented ranking-only evaluation.

`score` may be a raw Cross-Encoder logit, cosine similarity or another declared
model score; it is not assumed to be a calibrated probability. Optional
`retrieval_score` and `decision` must each be present for either every member of a
query or none. If actual rank IDs are omitted, the runner evaluates the complete
pool ordered by descending score, with original input order resolving ties.
Supplying `retrieved_vacancy_ids` requires an explicit `ranked_vacancy_ids` list;
the runner will not substitute a complete-pool ordering for the actual pipeline.
An explicitly empty rank list means no results and receives zero retrieval
credit. Blocked and unretrieved relevant jobs remain in metric denominators.

Freeze the model/configuration and predeclared ranking cutoffs using validation
only, before generating or inspecting the final test results:

```bash
python scripts/evaluate_rerank_human.py freeze-validation \
  --annotations /private/human-pools.jsonl \
  --predictions /private/model-validation-scores.jsonl \
  --k 1 5 10 --preselect-k 1 5 10 20 \
  --output /private/model-frozen-validation.json

python scripts/evaluate_rerank_human.py evaluate-test \
  --annotations /private/human-pools.jsonl \
  --predictions /private/model-test-scores.jsonl \
  --validation-predictions /private/model-validation-scores.jsonl \
  --frozen /private/model-frozen-validation.json \
  --output /private/model-human-test-report.json
```

The commands create new files exclusively and refuse overwrites. The validation
artifact stores the selected cutoff, ordered validation-content hash, exact
prediction-file hash, model/config identities and declared K values. Test
execution rechecks the validation inputs and cutoff, requires unchanged model and
config identities, and never selects a new threshold. File hashes record supplied
inputs; they cannot prove that an operator did not inspect labels before choosing
a model. Enforce the prospective freeze and assessor blinding operationally.

The report contains overall and per-language pair metrics, graded ranking
metrics, grade-2-only sensitivity with the same frozen threshold, actual retrieved
pool recall, and score-based preselection recall where complete retrieval scores
were supplied. All-negative query recall/NDCG/MRR are null and excluded from their
macro means; precision includes those queries. Counts are always reported.
Each profile has one canonical fixed query pool; duplicated profile queries are
rejected to prevent double-weighting. The runner detects exact normalized
duplicates and source/ID conflicts; audit
near-duplicates, job families, consent, assessor independence, agreement and source
permissions separately. Query/source-component confidence intervals and occupation
slices require the final collection metadata and remain a prospective analysis,
not measurements on the unjudged templates.
