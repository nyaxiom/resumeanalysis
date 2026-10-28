# Cover-letter integration validation

Executed on 2026-10-26, on `feature/local-cover-letter-generator`, based on
`feature/retrieve-rerank-backend` at `573e8d059c4e6bbf79abba2c2a1df3edb4bc6a0e`.
The draft PR is stacked on that feature branch; no main commit or merge is part
of this work. The inspected remote main remained
`a9632b2cd4b1077d5e88ec4fd126bd04dc694ea1` before publication.

## Executed checks

| Check | Result and scope |
|---|---|
| Full regression suite | **547 passed in 70.18 seconds**, with both local E5 and reranker opt-in checks enabled; no skips. |
| Ruff lint and formatting | Passed across the checkout. |
| Git whitespace validation | `git diff --check` passed. |
| Source distribution and wheel | Both built successfully with `python -m build --no-isolation`. |
| Packaged notices | Wheel contains 79 license/notice files, including the four new cover-letter/model licenses; vendored adapter included and model weights excluded. |
| Original assets and code | All 16 recorded baseline SHA-256 digests matched, including both saved MLP checkpoints, matching/extraction code, processed data and the original SkillSpan test source. |
| Attribution integrity | Existing `THIRD_PARTY_NOTICES.md` content remains an unchanged prefix; all four added full licenses are byte-identical to pinned upstream files. |
| Actual local CLI | The authored two-resume example selects resume 1, preserves relevance 1.0 and returns a nonempty Qwen draft. Full actual output is archived. |
| Actual local generation comparison | Two upstream candidates on the same four validation cases per model; final validation, four frozen held-out cases and one exact-repeat probe completed. See the separate benchmark report. |

Commands for the full regression and build checks:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RESUMEANALYSIS_RUN_E5_TESTS=1 RESUMEANALYSIS_RUN_RERANK_TESTS=1 \
  python -m pytest -q
ruff check .
ruff format --check .
git diff --check
python -m build --no-isolation
```

These model-enabled regression flags require the existing locally downloaded E5
and cross-encoder assets. Ordinary CI does not download those models and retains
the existing explicit skip conditions. The tests for the new orchestration and
transport use injected backends; their passes are contract evidence, not writing
quality measurements. Async checks were executed with working local socket support
because the execution sandbox prevented asyncio thread-completion wakeups.

The build retains an existing setuptools warning about `+` in an unrelated Torch
license filename. Renaming or removing that existing notice was not necessary.

## Implementation changes

| Area | Change |
|---|---|
| `cover_letters/vendor/applykit.py` | Licensed upstream prompt and input-boundary helpers, with original attribution. |
| `cover_letters/prompts.py` | Adapted upstream builder: native structured inputs, EN/RU, bounded length, data-only context and validation-driven grounding constraints. |
| `cover_letters/generator.py` | Thin async adapter to the existing local Qwen runtime, lazy initialization, bounded output screening and redacted failures. |
| `cover_letters/schema.py`, `service.py` | Strict request/resume records; existing relevance scoring followed by deterministic positive-resume selection and optional draft generation. |
| `integration/cover_letters.py` | Direct client, correlated Kafka request/reply client, model worker and publish-before-commit transport adapter. |
| Main CLI and `cover_letters/cli.py` | Add `resumeanalysis analyze`; keep the existing MLP default and opt-in reranker. |
| `schemas.py` | Enable optional summary/letter strings and positive resume ID; relevance-only results keep their existing null defaults. |
| Shared Qwen config/runtime | Permit explicitly configured CPU bfloat16 alongside float32; retain original relevance config, JSON decision schema and no-fallback behavior. |
| Configurations, examples, scripts and evaluation | Pinned primary and diagnostic Qwen configs, actual request/result, fresh-output benchmark and historical-prompt replay, frozen reports and fingerprints. |
| Documentation and licenses | Six-project audit, exact source evidence, comparison, setup, validation and complete original third-party licenses. |

Deleted old generative components: **none**. No cover-letter generator existed in
the audited branches, and the existing Qwen relevance backend remains needed.
Removed dependencies: **none**. No upstream UI, database, PDF, cloud-provider or
Kafka package was introduced. Relevance checkpoints, matching code and extractor
defaults were preserved; no downstream matching model was retrained.

## Remaining verification and release limits

The strict factuality acceptance condition is **unmet**: actual letters add
unsupported details despite passing the lexical screen. The benchmark report
documents concrete examples. Every returned letter is marked as a draft in the
analysis explanation and needs factual review. This draft PR does not claim an
unattended production release or verified hiring benefit.

No independent human panel, separate LLM-as-a-judge run, live Kafka broker,
ApplyPilot database integration, end-to-end email submission, CUDA execution or
production load test was performed. Kafka tests validate the injected-client
contract and publish-before-commit order; broker lifecycle, consumer isolation,
deduplication, poison-message handling and deployment authentication still belong
to the host application. The synchronous generator is serialized per instance;
request cancellation does not interrupt an already running inference thread.

The eight authored cases are too small for broad quality claims. Test results
were reported without further tuning. Future factuality improvements require
fresh validation data, independently checked source claims and a new held-out set.
