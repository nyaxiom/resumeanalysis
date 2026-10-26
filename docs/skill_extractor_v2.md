# Skill extractor v2

The default extractor remains the original `SkillExtractor`, also exposed as
`LegacyRuleBasedExtractor` and `create_extractor("legacy")`. Its implementation is
preserved byte-for-byte. The E5 + MLP and Qwen backends remain independently usable.
V2 is an explicit option, not an automatic change to existing saved predictions.

The [research matrix](skill_extractor_research.md) covers all four requested
repositories, exact source commits, code/data/model terms, maintenance, compatibility
and measured versus unmeasured claims. Neither SkillNER fork adds requirement
classification or original-character alignment; their default dictionary licensing
and mutable downloads need review. JobBot has no verified source license, and the
linked JobBERT assets have no verified model-specific grant. No implementation or
dictionary was copied from these projects. No pretrained NER model was installed.

## Available implementations

| Selection | Inputs/assets | Scope |
|---|---|---|
| `legacy` | Existing small vocabulary and original rules | Reproduce original extraction/checkpoint behavior |
| `v2` | Original curated aliases, existing normalizer/taxonomy, new context rules | Broader technical/professional vocabulary and EN/RU requirements |
| `v2 --lexicon-path ...` | Above plus an explicitly built SkillSpan training-only phrase artifact | Additional English phrase coverage; an experimental lexical variant |

All three run on CPU and make no network calls. No dependencies were added for
extraction or evaluation. The optional phrase artifact learns exact phrases from
licensed training annotations; it is not a semantic neural NER model. Future NER
providers can implement the extractor interface, but must register their identity,
retain original offsets, establish model rights and pass the same requirement tests.
No generic unverified-model loader or silent fallback is supplied.

The curated additions were authored from the task's competency categories and the
supplied diagnostic, before evaluating v2 on the held-out SkillSpan test. They were
not copied from EMSI, ESCO exports or test-set false negatives. Existing ESCO IDs,
relations and legacy aliases are unchanged. V2 creates a separate extended taxonomy
instance, using `SkillNormalizer` rather than a duplicate normalization system.

## Semantics

Recognition covers languages, web frameworks, databases, cloud/DevOps tools, ML/data
engineering, testing, and explicitly named professional/soft competencies. It
preserves literal substrings, Python character offsets, canonical names and mention
provenance. Longest lexical matches and word boundaries avoid substring matches;
tokenized punctuation such as `Node . js` retains its original offsets.
JS/TS and Postgres use existing normalization. Bare Node needs backend/JavaScript
context; short C/R/Go names need language context and additional ambiguity guards.
No related skill is substituted as proof of possessing another skill.

Vacancy descriptions distinguish mandatory, optional, mentioned, negated and
alternative requirements. Headings such as Essential Skills, Qualifications,
Nice to Have, Обязательно and Желательно carry context to following lines;
responsibility/benefit headings reset it. Local preference, negation and contrast
clauses override section context. A job title is never passed as skill evidence by
the feature builder. Explicit structured skill lists remain authoritative even for
terms outside the dictionary.

`EnhancedExtraction` extends the legacy result with `alternative_groups` and
`negated_requirements`; Profile, Vacancy and VacancyAnalysis remain unchanged.
For `Required: Python or Node.js, Docker`, Docker is independent, while Python/Node
remain one disjunction. V2 feature ratios count that group as one unit: one confirmed
option satisfies it, all explicitly denied options contradict it, otherwise its
status is unknown. Unconfirmed alternatives are not flattened into independent
missing skills. Internal feature evidence and explanation retain their group and
status. An optional or negated group never becomes mandatory.

Candidate negation and learning intentions remain separate from confirmed skills.
Absent information remains unknown. These lexical rules cannot verify the truth of
resume claims or resolve every linguistic ambiguity. Complex nested expressions,
mixed comma lists with postfix qualifiers, slash-based alternatives and subtle
negation require human review; slash-separated technologies are not automatically
treated as alternatives. Unknown terms are retained when supplied explicitly,
but open-ended unseen terms in free text may be missed.

## Checkpoints and compatibility

The original `artifacts/run/model.pt` is retained. Format-1 checkpoints accept only
the original extractor and compatible taxonomy/encoder/features. Registry guards
check the baseline implementation and normalizer, including mutated pattern/lookup
state. Unknown extractor implementations do not inherit legacy compatibility.

V2 uses pipeline version 2 and format-2 checkpoints recording extractor identity.
Its identity includes implementation files, normalization/taxonomy, matching state
and the optional lexicon identity. Changed semantics, vocabulary, model pairing or
pipeline version fail before inference. Its feature-cache keys are separate; legacy
cache keys remain unchanged. Formatting the identity-bearing source files also
invalidates v2 artifacts intentionally. Rebuild/retrain after such edits.

Training with v2 requires a separate output directory and refuses to overwrite an
existing checkpoint or metrics file. There is no switch to force an old MLP to use
new extraction as a production-valid combination. Qwen can select either extractor
for its grounded skill catalog without requiring MLP weights or fine-tuning; its
semantic score remains uncalibrated and pretrained Qwen quality remains unverified.

## Reproduce on Arch Linux

Use a project virtual environment; do not install project packages into Arch's
system-managed Python. The project supports Python 3.11/3.12 in CI. If the system
Python is newer, let uv select a supported interpreter.

```bash
sudo pacman -S --needed git uv
git clone https://github.com/nyaxiom/resumeanalysis.git
cd resumeanalysis
git switch feature/skill-extractor-v2
uv sync --frozen --python 3.12 --extra dev --extra data --extra qwen

uv run --frozen resumeanalysis extractors extract --extractor v2 --vacancy \
  --text 'Essential Skills: React, Next.js, Python/FastAPI, Docker'
uv run --frozen resumeanalysis extractors fixtures --extractor v2 \
  --input tests/fixtures/skill_extraction_v2.json \
  --output artifacts/experiments/skill_extractor/controlled-v2.json
```

Download the already-reviewed SkillSpan source explicitly and pin its revision:

```bash
git clone https://github.com/kris927b/SkillSpan data/external/skillspan
git -C data/external/skillspan checkout 2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b
uv run --frozen resumeanalysis extractors compare \
  --input data/external/skillspan/data/json/test.json \
  --output artifacts/experiments/skill_extractor/curated.json
uv run --frozen resumeanalysis extractors build-lexicon \
  --input data/external/skillspan/data/json/train.json \
  --output artifacts/experiments/skill_extractor/train-phrases.json
uv run --frozen resumeanalysis extractors compare \
  --input data/external/skillspan/data/json/test.json \
  --lexicon-path artifacts/experiments/skill_extractor/train-phrases.json \
  --output artifacts/experiments/skill_extractor/train-phrase-comparison.json
```

The builder accepts only the exact SHA256-pinned training split, so passing dev/test
or renamed/modified data fails. Fixed defaults retain phrases occurring at least
twice, at most twelve tokens, excluding short ambiguous terms and disjunction
phrases. These are declared engineering filters, not universally optimal thresholds.
No test labels are used to construct the artifact. The artifact includes source
commit/hash, parameters, counts and the actual upstream MIT notice; weights/data
and generated phrases stay under ignored artifacts. Changing its identity requires
a separately matching v2 checkpoint. Broader data redistribution rights remain a
manual review item, as in the existing audit.

The unchanged original evaluator remains available as `resumeanalysis skillspan`.
New comparisons score all SKILL/KNOWLEDGE annotations on identical hashed input,
report exact spans and token overlap, deterministic false-positive/negative
examples, initialization, extraction latency, wall/process CPU and peak RSS.
Fresh subprocesses isolate extractor measurements. RSS includes interpreter and
imports; it does not represent the vocabulary's exclusive allocation. Shared host
load and caches affect latency, so these are local measurements, not guarantees.

For the uncommitted local Role Radar pairs:

```bash
uv run --frozen resumeanalysis extractors role-radar --extractor v2 \
  --input data/external/role-radar/processed/inference_pairs.jsonl \
  --output artifacts/experiments/skill_extractor/role-radar-v2.json
```

The file is absent in this cloud workspace. The runner reports every line outcome,
unmapped field paths, hashes, extracted spans/categories and timing. Source scores
remain metadata; without independent annotations it reports no extraction accuracy.
Outputs may contain local text and identifiers: keep them under ignored artifacts
and follow the source retention/privacy policy. Missing files fail clearly rather
than creating a fabricated 106-record report.

For optional local downstream scores, add `--checkpoint` pointing to a checkpoint
that matches the selected extractor and `--e5-path artifacts/e5`. The runner uses
each supplied profile rather than inventing context, enforces normal checkpoint
compatibility, and retains score/analysis as uncalibrated diagnostics. Without
those flags it runs extraction alone. No fit labels are inferred from Role Radar scores.

With the existing E5 and original prepared dataset/checkpoint available:

```bash
uv run --frozen resumeanalysis train --extractor v2 --data data/processed \
  --config configs/cpu.json --output artifacts/skill-extractor-v2/run
uv run --frozen resumeanalysis evaluate --extractor v2 --data data/processed \
  --output artifacts/skill-extractor-v2/run
uv run --frozen resumeanalysis extractors ablate --data data/processed \
  --v2-checkpoint artifacts/skill-extractor-v2/run/model.pt \
  --output artifacts/experiments/skill_extractor/ablation.json
```

The four conditions use the same original test pairs: legacy/original model and
v2/retrained model, each with description-only versus original structured vacancy
requirements. Description-only clears exactly the two structured skill lists and
records what was removed. No requirement labels are invented. Both checkpoints
must match all original train/validation/test hashes. Thresholds come from their
respective validation splits. Source-label metrics under these interventions are
diagnostics; synthetic/heuristic labels do not establish hiring accuracy or labels
valid for changed observations. Shared-process ablation timing is cache-sensitive
and is not an isolated extractor speed comparison.

## Notebook and tests

`notebooks/skill_extractor_comparison.ipynb` keeps reusable logic in Python modules.
It shows side-by-side spans/categories, inline SVG precision/recall/F1 and latency
charts, false positives/negatives, independent EN/RU fixture results and explicit
Role Radar availability. It writes JSON/HTML under
`artifacts/experiments/skill_extractor/`; committed cells contain no runtime output.

Install Jupyter in a separate notebook environment or with `uv tool`; it is optional
and outside the locked extraction runtime. Select the project environment as its
kernel. Set `SKILLSPAN_PATH` to the pinned test file and optionally `ROLE_RADAR_PATH`
and `EXTRACTOR_LEXICON_PATH`. `SKILL_EXTRACTOR_SMOKE=1` explicitly selects two authored
mechanics rows and does not measure real SkillSpan quality. Equivalent commands
above work without Jupyter or IPython.

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
RESUMEANALYSIS_RUN_E5_TESTS=1 uv run --frozen pytest -q
```

The last command needs the separately downloaded E5. Tests cover the original
pipeline, requirement regressions, aliases, negation/uncertainty, alternatives,
offsets, explicit unknown skills, input failures, long text, source provenance,
training-only lexicon integrity, Qwen mocks and checkpoint/cache compatibility.
Controlled Russian fixtures measure regression conformance only, not general
Russian extraction quality. Read the measured results in
[the validation record](skill_extractor_validation.md) before making quality claims.
