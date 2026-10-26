# Local cover-letter comparison

Measured on 2026-10-26 with actual local Qwen inference, using eight authored,
fictional resume/vacancy pairs: four validation and four held-out test cases,
with two English and two Russian cases in each split. No training was performed.
No personal resumes, paid APIs or external judge services were used.

**ApplyKit is selected for the draft integration, but the strict factuality
criterion is not met.** The final adapter passes the bounded automatic checks
on all four held-out cases while still adding unsupported details to the prose.
It must not be treated as ready for unattended application submission.

The repository audit found no prior cover-letter implementation. The old Qwen
backend predicts relevance JSON, which is a different task; there is no old
cover-letter baseline to invent or remove.

## Controlled comparison and selection

Both shortlisted implementations received identical source pairs, the same model
revision, dtype, local runtime, greedy decoding, seed 42, 4,096-token input limit,
512-token output budget and 180-word target within each comparison. The runtime
used two inference threads. The two prompts differ because they are the component
being evaluated. Resume-Matcher received the explicit language name `Russian`
because its original language mapping otherwise falls back to English.

The initial Qwen2.5-1.5B-Instruct experiment used float32. Both candidates exceeded
the word limit on two of four cases. ApplyKit produced eight skill flags and one
numeric flag; Resume-Matcher produced four skill flags. These are lexical flags,
not verified false-claim counts: for example, aliases can cause false positives.
Actual outputs also included placeholders, an unsupported 70% improvement,
denied Docker experience and, in one Russian output, Chinese characters. That
experiment did not justify a quality promotion. The original report predates the
final screen and does not contain a screen-pass metric.

The subsequent comparison used Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, unquantized CPU bfloat16. Model and dtype
both differ from the earlier diagnostic, so cross-model changes cannot be
attributed solely to prompt design.

| 4B run | Cases | Automatic screen passes | Numeric / skill flags | Placeholders | Mean generation time |
|---|---:|---:|---:|---:|---:|
| Initial ApplyKit adaptation, validation | 4 | 3/4 | 1 / 1 | 0 | 244.57 s |
| Resume-Matcher adaptation, same validation | 4 | 1/4 | 2 / 1 | 3 | 212.53 s |
| Revised ApplyKit, validation | 4 | 4/4 | 0 / 0 | 0 | 178.72 s |
| Frozen revised ApplyKit, held-out test | 4 | 4/4 | 0 / 0 | 0 | 156.31 s |

The selection rule prioritizes validation screen passes, then fewer lexical
flags/placeholders, then the smaller MIT adapter for a tie. ApplyKit won the
common-model comparison. Its original STAR/result instruction still encouraged
an invented 40% improvement. Using validation only, the adapter removed that
instruction, shortened the evidence paragraph and explicitly prohibited inferred
benefits, scale and implementation details. This improved measured screening
results, but did not solve unsupported elaboration.

The final validation run was frozen before any test generation. A serialization
correction made benchmark inputs and production Pydantic inputs identical
(omitting default empty fields and normalizing integral floats). Before testing,
all four rendered validation prompt hashes and screen results were checked against
the already measured run; no regeneration or test observation was used to update
the source fingerprint. The [frozen selection](../evaluation/cover_letters/selection.json)
records that replay and the validation report digest. The script rejects changed
prompt/model/data fingerprints or an unfrozen test invocation. No test-dependent
prompt, threshold or model selection followed the held-out run.

## What the outputs establish

All four final validation and four test outputs meet the word limit, requested
language-script majority and placeholder/script checks. The final validation
letters have 103–139 words; test letters have 81–123. The screen found no
unsupported digit-based numbers or recognized skill names in those final outputs.
It cannot verify achievements, causality, responsibilities, implementation details,
spelled-out numbers or arbitrary skills outside its vocabulary.

The following is **assistant qualitative inspection**, not independent human
annotation, a separate LLM-as-a-judge experiment or factuality accuracy. Each row
identifies a concrete source gap in a letter that passed the screen. Counts of
these observations would depend on how claims are segmented and are not presented
as a hallucination rate.

| Case | Vacancy/source alignment | Detail requiring correction or verification |
|---|---|---|
| Validation EN backend | Mentions the employer, Python APIs, SQL imports and tests. | Claims the tests ensured data integrity and process stability; those outcomes are absent from the resume. |
| Validation RU data | Uses SQL/Python, the sales-reporting role and the supplied 12 weekly reports. | Adds a specific extraction/processing/formatting workflow not explicitly supplied. |
| Validation EN student | Correctly links the CSV missing-value project to the internship. | Invents logging, use of built-in libraries and clear error reports. |
| Validation RU coordinator | Retains scheduling, participant questions and 10 volunteers. | Links spreadsheets to volunteer onboarding and claims provision of schedule access without source evidence. |
| Test EN support | Addresses email support, documentation and the supplied colleague training. | Says training ensured consistent request handling; that outcome was not provided. |
| Test RU junior | Retains the Python book project and does not claim Java experience. | Adds an interface with add/delete/view operations absent from the project description. |
| Test EN data | Uses SQL stock reporting and the supplied four-to-two-hour reduction. | Attributes the reduction to SQL refinements and streamlined extraction/validation, which the source does not establish. |
| Test RU project | Uses tasks, meeting minutes and the six-person calendar. | Adds timely completion and synchronized workflow benefits absent from candidate facts. |

English prose is readable but repeats openings such as “I’m excited to apply” and
confident closing formulas. Russian prose is readable in these examples but
includes generic claims and some awkward phrasing. These observations do not
establish native-language fluency, hiring effectiveness or generalization to
longer real resumes. The internship outputs show why plausible extra detail is
still a factuality failure even when every recognized technology is supported.

No exact sentence repeats within any measured letter. Across the four final
letters in each language, mean pairwise five-token Jaccard overlap is 0.0236 for
English and 0.0016 for Russian (six pairs per language). Low literal overlap does
not rule out a shared three-paragraph template or repeated ideas. One additional
held-out English generation with the same input was an exact text match. This is
a single repeatability probe, not a guarantee across runtimes/devices/dtypes.

## Runtime measurements

| Run | Model initialization | Process peak RSS | VRAM | Generation range |
|---|---:|---:|---:|---:|
| Initial 1.5B comparison, eight letters | 7.80 s | 9.013 GiB | 0 | 64.22–168.98 s |
| Initial 4B comparison, eight letters | 9.31 s | 8.165 GiB | 0 | 158.80–299.40 s |
| Revised 4B validation, four letters | 8.81 s | 8.146 GiB | 0 | 140.73–207.97 s |
| Frozen 4B test, four letters plus one repeat | 7.42 s | 8.151 GiB | 0 | 107.26–206.16 s |

Generation timings exclude initialization and subsequent lexical screening. RSS
is the process high-water mark, including Python, model and tokenization; it is
not isolated model allocation or a deployment capacity guarantee. These are
single-process observations in a shared execution environment, with occasional
concurrent verification work, not controlled latency percentiles. No GPU was used.
The repeat took 148.62 seconds. Local CPU inference is too slow for an interactive
request deadline of only a few seconds; the Kafka client defaults to a 900-second
timeout and deployment must configure its own queue/deadline policy.

## Evidence and reproduction

The [machine-readable metrics](cover_letter_benchmark_metrics.json) contain
aggregates and source-report hashes. Full inputs, exact rendered prompts, prompt
hashes, generated text, configurations and timings are preserved in:

- [Authored split cases](../evaluation/cover_letters/cases.jsonl).
- [Initial 1.5B comparison](../evaluation/cover_letters/initial_1_5b_validation.json).
- [Initial 4B comparison](../evaluation/cover_letters/initial_4b_validation.json).
- [Revised validation](../evaluation/cover_letters/final_validation.json).
- [Held-out test and repeat](../evaluation/cover_letters/held_out_test.json).

Install and download assets as described in [local setup](cover_letters.md). To
reproduce the final comparison and frozen evaluation with fresh output paths:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/benchmark_cover_letters.py --split validation --candidate both \
  --output artifacts/cover-letter-replay/validation.json \
  --freeze-selection artifacts/cover-letter-replay/selection.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/benchmark_cover_letters.py --split test \
  --selection evaluation/cover_letters/selection.json --repeat-first 1 \
  --output artifacts/cover-letter-replay/test.json
```

The first command uses the current revised ApplyKit prompt. To reproduce an
earlier prompt version exactly, use its retained messages instead of reconstructing
them with today's builder:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/replay_cover_letter_prompts.py \
  --report evaluation/cover_letters/initial_4b_validation.json \
  --output artifacts/cover-letter-replay/initial-4b.json
```

The replay verifies saved prompt hashes and uses that report's exact model config.
For the 1.5B report, first download the diagnostic configuration's assets. Replays
are reproduction, not a new independent evaluation. Any further tuning needs new
validation examples and a new untouched test set. Independent human review and a
larger evaluation of source entailment remain prerequisites for an unattended
release; the current automatic-pass numbers do not satisfy that requirement.
