# Vocabulary coverage hypothesis: validation evidence

Limited lexical coverage explains much of the current extractor's low recall.
It does **not** explain away the need for phrase boundaries and contextual
recognition. Adding training phrases improves recall while degrading precision
and F1 in this controlled intervention.

## Scope

The experiment uses **2,402 eligible SkillSpan development sentences**, with
**2,136 distinct gold token spans** across the SKILL/KNOWLEDGE layers. It excludes
exact/near SkillSpan test text, conflicts and repeated development text identified
by the audit. Vocabulary fitting uses audited training partitions only: **3,665
SkillSpan sentences**, optionally plus **9,608 Green training sentences**.

These are new **validation** results. They are not a rerun of the user's quoted
whole-test numbers (precision 0.6095, recall 0.1475, F1 0.2375), and differences
from those historical numbers do not measure a new test improvement. All model
comparisons below use the same eligible development examples. No test annotations,
external evaluation labels or synthetic sentence labels supplied vocabulary entries.

The two vocabulary interventions use a fixed minimum frequency of two distinct
training sentences, maximum twelve tokens, and declared filters for ambiguous
single terms/disjunctions. They contribute **430** and **1,129** train-derived
phrases, respectively, before merging overlapping V2 aliases. No frequency sweep
or phrase harvesting from development/test was performed.

## Measured comparison

| Extractor | Precision | Recall | Exact token F1 | Gold full-alias coverage |
|---|---:|---:|---:|---:|
| Legacy | 0.5114 | 0.0843 | 0.1447 | 7.40% |
| V2 | 0.5339 | 0.1437 | **0.2265** | 12.69% |
| V2 + audited SkillSpan training phrases | 0.1736 | 0.2767 | 0.2134 | 25.37% |
| V2 + audited SkillSpan and Green training phrases | 0.1190 | 0.3141 | 0.1726 | 29.21% |

Coverage means the complete gold surface, after declared case/Unicode/whitespace
normalization, equals a current vocabulary alias. It does not mean that a short
known word occurs somewhere inside a longer skill. The slightly different lexical
regex coverage of V2 is **13.06%** because punctuation/guard candidates differ
from literal alias lookup. Exact token scoring projects literal predictions onto
the original token intervals, matching the historical evaluator's unit. Character
exact V2 recall is **12.78%**; token projection can count a partial-token match as
token exact. Neither coverage statistic is claimed as an upper bound on that
more permissive token score.

## Known names, boundaries and contextual mentions

For V2, **271 of 2,136 mentions** are full-alias known. Recall is **99.26%** on
these known surfaces and **2.04%** on the remaining **1,865** surfaces. The known
definition is computed from the fixed vocabulary independently of predictions.
The nonzero unknown recall can result from punctuation/regex variants and token
projection; it does not establish semantic inference.

| V2 diagnostic group | Gold mentions | Exact token recall |
|---|---:|---:|
| Single-token mentions | 705 | 36.74% |
| Multi-token mentions | 1,431 | 3.35% |
| Two or three tokens | 745 | 6.44% |
| Four or more tokens | 686 | 0.00% |
| KNOWLEDGE layer | 1,076 | 25.46% |
| SKILL layer | 1,060 | 3.11% |
| Action/ability surface-cue proxy | 284 | 4.23% |

The action/ability group is an explicit lexical diagnostic (for example,
action/ability cue patterns), not an independently annotated contextual-skill or
requirement category. SkillSpan has no mandatory/optional/negated requirement gold;
this experiment makes no requirement-classification accuracy claim. Its procedural
SKILL spans and longer phrases nevertheless expose a substantial mismatch with a
mostly short, named-technology vocabulary.

Of V2's **1,829** missed spans, **1,719** have no overlapping prediction and
**110** have an overlapping prediction with the wrong boundary. Six gold surfaces
match its complete lexical regex but are missed in sentence context. These
diagnostics overlap: they are not independent additive causes. After adding Green
and SkillSpan phrases, wrong-boundary misses rise to **875**, and false positives
rise from **268** to **4,967**. More isolated phrases activate many short/generic
matches and nested fragments. The expanded dictionaries therefore remain
experimental artifacts and are not installed in the default extractor.

## Russian evidence

There is **no comparable Russian span recall estimate** from this work. The
51,029 Russian vacancy-title examples have skill-list targets rather than human
mention offsets. Many labels need not occur in their titles. Reporting a Russian
NER recall from them would change the task. Existing EN/RU authored regression
fixtures can check behavior, but cannot replace an independent Russian benchmark.
Language distribution in the inventory measures source metadata, not proven
Russian skill extraction coverage.

## Implication for more data

More **compatible, real, varied span supervision** is likely to help a learned
extractor recognize long/procedural mentions and choose boundaries; this is a
testable next experiment, not an improvement established by the dictionary run.
The intervention shows a strong lexical bottleneck and also demonstrates that
simply growing the dictionary can lower overall quality. It does not isolate
training-set size from annotation policy/domain shift, and provides no causal
evidence that synthetic label volume improves NER.

Prioritize a lightweight token classifier trained on audited SkillSpan training
data, then a controlled Green Skill-only auxiliary experiment. Keep sentence-label
datasets for separate normalization/classification objectives. Obtain licensed
Russian job-level human span annotations before making Russian comparisons.
Use development data to select a model, and a new independent holdout for the
final claim. The full counts and definitions are preserved in
[vocabulary_coverage.json](vocabulary_coverage.json).
