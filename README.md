# ResumeAnalysis

Local job recommendations that learn each user's interests from explicit likes and
dislikes. **River is the main recommendation component.** Frozen multilingual E5,
skill evidence and explicit preferences provide cold-start ranking; SQLite retains
feedback across restarts. No pretrained personalized recommendation model is needed.

## Runtime structure

| Package | Responsibility |
|---|---|
| `recommendation/` | River learning, ranking, exploration, SQLite and feedback |
| `extraction/` | Skill mentions, requirement evidence and phrase-artifact loading |
| `normalization/` | Skill aliases, canonical names and related concepts |
| `embeddings/` | One frozen multilingual E5 encoder and its explicit downloader |
| `profiles/` | Text/document ingestion, optional local Instructor extraction, corrections |
| `matching/` | Shared qualification evidence, text serialization and constraint checks |
| `cover_letters/` | Explicit Qwen letter drafting from confirmed profile facts |
| `integration/` | One local `ApplyPilotClient` and source-data adapters |

Qualification, hard constraints and personal interest remain separate. A dislike
is not evidence that the candidate is unqualified. Unknown facts stay unknown.
No Kafka, HTTP worker or alternative matching backend is in the runtime package.

## Install and run

Python 3.11+ is required. From the repository root:

```bash
uv sync --frozen --extra documents --extra profile-llm --extra qwen
source .venv/bin/activate
resumeanalysis download-e5 --output artifacts/e5

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 resumeanalysis recommend \
  --profile examples/recommendation_profile.json \
  --vacancies examples/recommendation_vacancies.json \
  --database artifacts/preferences.sqlite --exploration-fraction 0

resumeanalysis feedback --database artifacts/preferences.sqlite \
  --user-id demo-user --vacancy-id authored:data-remote --action dislike \
  --event-id first-dislike
```

Running `recommend` again restores learned state and excludes the rejected vacancy.
The document, profile-LLM and Qwen extras are optional when the host supplies
structured profiles and does not request letter generation. Model downloads are
explicit. Recommendation inference uses local files and does not call a paid API.

Profile extraction defaults to a separately provisioned local Ollama-compatible
server. Extracted profiles require review and can be corrected without overwriting
the source document. See [profile and recommendation usage](docs/online_recommender.md).

To draft a letter for a selected vacancy:

```bash
resumeanalysis download-qwen --config configs/cover_letter_qwen_cpu.json
resumeanalysis cover-letter \
  --profile examples/recommendation_profile.json \
  --vacancy examples/recommendation_vacancy.json --language en
```

Letter generation is explicit and uses Qwen; it does not classify qualification,
choose a matching backend or apply for a job. Drafts need factual review. See
[cover-letter usage](docs/cover_letters.md).

## ApplyPilot

`resumeanalysis.integration.ApplyPilotClient` provides `recommend`,
`record_feedback` and `draft_cover_letter` through local Python calls. The host
supplies authenticated profiles and available vacancies. No changes to ApplyPilot
are required inside this repository; deployment still needs host integration.

See the [integration guide](docs/online_recommender_integration.md),
[versioned schemas](docs/online_schemas/), [architecture and migration](docs/runtime_refactor.md)
and [runtime validation](docs/runtime_validation.md).

## Tests and historical research

```bash
uv sync --frozen --extra dev --extra documents --extra profile-llm --extra qwen
uv run --frozen pytest -q
uv run --frozen ruff check .
uv build
uv run --frozen python scripts/check_runtime_wheel.py dist/resumeanalysis-*.whl
```

The old MLP, Qwen classification and cross-encoder backends, training, dataset
preparation, experiments and transport adapters live in the optional
[`legacy/` workspace package](legacy/README.md). They are excluded from the
runtime wheel and normal installation. Install the `research` group to use them;
existing checkpoints and historical evaluation artifacts are preserved.

Real-user recommendation improvement and calibrated probabilities have not been
established. The [existing synthetic evaluation](docs/online_recommender_benchmarks.md)
has mixed metrics and preference-change lag; this structural refactor makes no new
quality claim. Live Ollama and deployed ApplyPilot integration remain unverified.

The project MIT license applies to original code. Original third-party notices are
retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/](LICENSES/).
Downloaded datasets, model weights, databases, caches and personal documents stay
outside Git. See the [license audit](docs/license_audit.md) for unresolved upstream
rights questions.
