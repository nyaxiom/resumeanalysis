# Local Qwen cover-letter drafts

The runtime drafts a letter from one confirmed profile and a selected vacancy.
Recommendation ranking remains in River; generation does not choose between
matching backends or reinterpret a dislike as a qualification decision.

```bash
uv sync --frozen --extra qwen
source .venv/bin/activate
resumeanalysis download-qwen --config configs/cover_letter_qwen_cpu.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 resumeanalysis cover-letter \
  --profile examples/recommendation_profile.json \
  --vacancy examples/recommendation_vacancy.json --language en
```

The configured local Qwen weights, tokenizer, license and checksums must exist.
The downloader is explicit. There is no cloud fallback or automatic replacement
model. `GenerationConfig` lives in `resumeanalysis.cover_letters.qwen_config`;
classification thresholds and retries belong to the archived Qwen scorer.

`CoverLetterService(generator).draft(profile, vacancy, language="en")` is the
Python interface. `ApplyPilotClient.draft_cover_letter(user_id, vacancy_id)` uses
the same service after checking profile ownership and vacancy availability.
Showing jobs and recording feedback never generate letters implicitly. A request
with an unreviewed extracted profile fails before the generator is called.

The generator reuses the existing licensed prompt adapter, local Transformers
runtime, evidence checks and output screening. Skill-specific experience is
preserved as explicit context and is never summed into overall experience.
The CLI returns `vacancy_id`, `cover_letter` and `needs_review: true`. Drafting does
not apply for a job, send a message or certify that requirements are met.

Screening catches some unsupported skills, numbers and invalid output, but cannot
prove arbitrary factual entailment. Existing [benchmark limitations](cover_letter_benchmarks.md)
remain: strict factuality has not been established. The [source audit](cover_letter_audit.md)
and original notices still apply; no new model was trained for this refactor.

The previous multi-resume `AnalysisService`, MLP/cross-encoder selection and
Direct/Kafka request/reply adapters are archived under
[`resumeanalysis_legacy`](../legacy/README.md). They are not imported by the runtime.
