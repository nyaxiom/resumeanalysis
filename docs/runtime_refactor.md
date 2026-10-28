# Runtime package refactor

Base: `52bd68f` (the merged online recommender). Work is isolated on
`feature/runtime-cleanup`; main is not modified.

## Dependency findings and plan

The package root currently imports the MLP predictor. Recommendation features
import `rerank.evidence`, which imports the Qwen classification module only to
reuse skill evidence. The root CLI registers training, dataset preparation,
experiments and three matching backends. Cover-letter analysis selects between
MLP and cross-encoder scoring, while integration exposes multiple local/Kafka
clients. These are implementation couplings rather than requirements of the
River recommendation product.

1. Extract shared text serialization, skill evidence and constraint checks into
   `matching/`, preserving recommendation feature values and behavior.
2. Move training, dataset preparation, evaluation experiments, MLP/Qwen/cross-encoder
   matching and historical transport adapters to an optional `legacy/` distribution
   with the `resumeanalysis_legacy` namespace. Keep its regression tests runnable.
3. Retain the existing E5 implementation and local Qwen generation runtime. Keep
   Instructor profile extraction local by default. Production commands focus on
   recommendations, explicit feedback, profiles and drafting a letter for a selected
   vacancy; generation does not classify qualification or apply for a job.
4. Expose one local `ApplyPilotClient`, composing profile/vacancy providers, the
   recommendation engine and an optional letter generator. No transport layer is
   needed inside this package.
5. Update installation, packaging and current documentation. Verify the runtime
   wheel without installing legacy, and run both runtime and archived tests.

Before edits, authored examples were evaluated into
`tests/fixtures/runtime_refactor_baseline.json`. These preserve exact compact and
projected features, identities, qualification evidence and constraint outcomes.
Source-based extractor identities need a narrow, hash-verified compatibility map
when moving the training-only lexicon builder: unchanged inference must not reset
saved user models. Any unrecognized source change must still change the identity.

Saved checkpoints, model weights, historical evaluation artifacts, original
copyright notices and licenses are preserved. New paths and commands will be
documented; historical reports retain their original provenance and results.
