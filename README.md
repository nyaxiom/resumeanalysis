# ResumeAnalysis

A small ML service for ApplyPilot: rank vacancies using each user's own likes and
dislikes, and draft a cover letter from supplied resume text. One FastAPI process,
one frozen multilingual E5 encoder, River logistic regression, SQLite, and one
Qwen Instruct inference backend. Neither pretrained model is fine-tuned.

## Install and run

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are supported.

```bash
uv sync --frozen --extra dev
cp .env.example .env
# Edit .env as needed; load it into the shell explicitly.
set -a
. ./.env
set +a
uv run --frozen uvicorn resumeanalysis.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`.env` is a local shell configuration file and is not read automatically. Do not
commit it. Interactive API documentation is at `http://127.0.0.1:8000/docs`.
Starting the application creates its SQLite database but does not load E5 or Qwen.
The first request that needs a model loads it. With default settings, Transformers
downloads the pinned revision from Hugging Face into its cache. Subsequent starts
reuse that cache. Weights are never included in Git or the Python package.

Existing model directories can be reused by setting
`RESUMEANALYSIS_E5_LOCAL_PATH=artifacts/e5` and
`RESUMEANALYSIS_QWEN_LOCAL_PATH=artifacts/qwen-cover-letter-4b` when those directories
contain complete compatible Hugging Face snapshots. Local paths use offline
loading and safetensors. Otherwise leave the paths empty. Set `HF_HOME` to choose
the Hugging Face cache directory and `HF_HUB_OFFLINE=1` after downloading if the
service must operate offline.

The lockfile selects CPU PyTorch. CPU inference is supported; a 4B Qwen model can
consume substantial memory and generate slowly. CUDA is optional: install a
compatible CUDA PyTorch build in your deployment environment and set the relevant
`*_DEVICE=cuda` (or `auto`). No quantization backend or automatic model fallback is
included. Float32 is the conservative CPU default; bfloat16 is configurable.

## API

### `POST /rank`

Each vacancy needs a unique `vacancy_id`, a nonempty `title`, and a nonempty
`description`. Up to 100 vacancies are accepted per request.

```bash
curl -s http://127.0.0.1:8000/rank -H 'Content-Type: application/json' -d '{
  "user_id": "user-1",
  "vacancies": [
    {"vacancy_id": "sales-1", "title": "Sales specialist", "description": "Develop customer accounts."},
    {"vacancy_id": "dev-1", "title": "Python developer", "description": "Build and maintain Python APIs."}
  ]
}'
```

Response: `{"personalized": false, "vacancies": [...]}`. Returned vacancies include
the original fields plus `score`. Until the user has at least one current like and
one current dislike, all scores are `0.5` and input order is preserved. Afterwards,
vacancies are sorted by descending River `predict_proba_one` output; ties preserve
input order. **Scores are uncalibrated ranking scores, not reliable probabilities
of a like.** Ranking never trains a model. No resume comparison, qualification
filter, exploration policy, or handcrafted scoring formula is applied.

### `POST /feedback`

A vacancy previously ranked for this user can be referenced by ID:

```bash
curl -s http://127.0.0.1:8000/feedback -H 'Content-Type: application/json' -d '{
  "user_id": "user-1", "vacancy_id": "dev-1", "liked": true
}'
curl -s http://127.0.0.1:8000/feedback -H 'Content-Type: application/json' -d '{
  "user_id": "user-1", "vacancy_id": "sales-1", "liked": false
}'
```

Alternatively send data directly, without calling `/rank` first:

```bash
curl -s http://127.0.0.1:8000/feedback -H 'Content-Type: application/json' -d '{
  "user_id": "user-1", "vacancy_id": "other-1", "liked": true,
  "vacancy": {"title": "Backend developer", "description": "Develop Python services."}
}'
```

Response fields: `updated`, `revision`, `feedback_count`, and `personalized`.
`liked` must be a JSON boolean. Repeating the current label and text is a no-op,
including concurrent retries. Sending the opposite label corrects the rating;
explicitly sending different vacancy text also replaces that feedback's text.
Corrections rebuild the model using each vacancy's current label in original
first-feedback order. There is only one active feedback record per user/vacancy.
If a correction leaves only one class, ranking returns to neutral.

An ID-only correction uses the text originally reacted to, even if a later rank
request included a changed listing. References are scoped to the user: another
user's stored vacancy is not available. Unknown references return `404`.
The latest received rating is authoritative; ApplyPilot should serialize a user's
changes and avoid retrying superseded ratings after a newer correction.

### `POST /generate-cover-letter`

```bash
curl -s http://127.0.0.1:8000/generate-cover-letter -H 'Content-Type: application/json' -d '{
  "resume_text": "I build Python APIs and maintain PostgreSQL databases.",
  "vacancy_title": "Backend developer",
  "vacancy_description": "Develop Python APIs for our scheduling product.",
  "instructions": "Keep the tone direct and concise."
}'
```

Response: `{"cover_letter": "...finished letter text..."}`. The optional `language`
field overrides the vacancy language; otherwise Qwen is instructed to use the
predominant language of its description. `instructions` can guide style and
emphasis but cannot supply candidate facts. Resume text is the only source of
candidate qualifications. There is no matching prerequisite or automatic sending.
Resumes and generated letters are not persisted by this service.

The prompt forbids invented skills, experience, education, achievements, contacts,
and personal information. Basic checks reject empty, overlong, unfinished, or
malformed output and some unsupported numeric/contact claims. **These checks do
not prove factual correctness or language compliance.** ApplyPilot should present
the letter for user review before sending. Sparse resumes may produce short letters.
Inputs over the configured token budget are rejected, never silently truncated.

All endpoints reject unknown fields and malformed bodies with `422`. Model or
storage unavailability returns `503`; rejected generation returns `502`. Request
bodies are omitted from validation error details. The service does not log them.

## Learning and persistence

E5 encodes exactly `title + "\n\n" + description` with the `passage:` prefix in
both feedback and ranking. The existing masked mean pooling and overlapping token
chunks are reused. Normalized chunk vectors are combined into one normalized
vacancy embedding. Inference batches chunks across vacancies. Optional `.npy`
cache files contain only numeric arrays and are read with pickle disabled.

Each user has an independent River logistic regression with constant-rate SGD.
Embedding coordinates are its only features. Each new explicit reaction calls
`learn_one`; there are no inferred labels, synthetic feedback, shared user models,
or downstream training pipeline. Receiving both classes enables sorting, but does
not establish model quality or calibrated probabilities.

SQLite stores user-scoped vacancies, current feedback text/labels, and numeric JSON
model snapshots. Feedback and model updates commit in the same transaction.
Snapshots contain weights, intercept, iteration count, revision, and feature/model
identity, protected by a checksum. No pickle or executable serialization is used.
A missing, malformed, or stale snapshot is rebuilt from stored feedback. The
pinned River version's constant-rate SGD has no momentum state to serialize.

Embedding identities include the remote commit or local asset content hashes,
chunking configuration, and preprocessing version. Changing identity automatically
invalidates embedding caches and rebuilds each user's model on their next request,
re-embedding the saved texts. Changing River version or learning parameters also
rebuilds models. Restart all workers together after configuration changes. Do not
mutate model files while the service runs. Old cache files can be removed manually
when no process uses them.

SQLite transactions coordinate writers, including separate processes. A single
worker is recommended: multiple workers duplicate heavyweight models in memory.
Corrections/recovery replay all current feedback, and version changes can require
expensive re-embedding while holding the database transaction. There is no bounded
history policy, background queue, or claim of large-scale throughput. Back up the
SQLite database using SQLite's backup API or while the service is stopped; account
for WAL files when making filesystem backups. Protect its permissions: it stores
user IDs, vacancy text, feedback, and learned preferences.

## Configuration

Every setting has a `RESUMEANALYSIS_` prefix. The complete editable example is
[.env.example](.env.example). Model revisions must be immutable 40-character commit
SHAs when using a remote model. Empty local model paths select Hugging Face loading;
an empty cache directory disables embedding caching.

| Variable suffix | Default | Purpose |
|---|---|---|
| `DATABASE_PATH` | `data/ml-service.sqlite3` | New service database |
| `EMBEDDING_CACHE_DIR` | `artifacts/service-embeddings` | E5 vector cache; empty disables |
| `E5_MODEL` | `intfloat/multilingual-e5-small` | Frozen encoder |
| `E5_REVISION` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Pinned E5 revision |
| `E5_LOCAL_PATH` | empty | Optional offline E5 directory |
| `E5_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `E5_BATCH_SIZE` | `16` | Chunks per inference batch |
| `E5_MAX_LENGTH` | `512` | Maximum tokens per chunk |
| `E5_OVERLAP` | `32` | Overlapping content tokens |
| `LEARNING_RATE` | `0.05` | River constant SGD rate |
| `L2` | `0.001` | River regularization |
| `QWEN_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Instruct generator |
| `QWEN_REVISION` | `cdbee75f17c01a7cc42f958dc650907174af0554` | Pinned Qwen revision |
| `QWEN_LOCAL_PATH` | empty | Optional offline Qwen directory |
| `QWEN_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `QWEN_DTYPE` | `float32` | `float32`, `bfloat16`, or GPU `float16` |
| `QWEN_MAX_INPUT_TOKENS` | `4096` | Prompt budget including system prompt |
| `QWEN_MAX_NEW_TOKENS` | `512` | Generation budget |
| `LETTER_MAX_WORDS` | `180` | Output word limit |

## ApplyPilot integration

ApplyPilot calls these HTTP endpoints directly. It authenticates users, derives
`user_id` from the authenticated session, supplies stable vacancy IDs, collects
explicit reactions, and displays letters for review. Sending applications remains
ApplyPilot's responsibility. This MVP has no built-in authentication: run it on a
private interface behind the trusted backend, never as an unauthenticated public
endpoint accepting arbitrary user IDs.

## Layout and development

```text
src/resumeanalysis/
  main.py           # FastAPI app and three routes
  schemas.py        # Request and response contracts
  config.py         # Environment settings
  embeddings.py     # Lazy E5, chunking, batching, cache
  recommender.py    # Per-user River training, ranking, replay
  storage.py        # SQLite schema and transactions
  cover_letter.py   # Lazy Qwen inference and output checks
  prompts.py        # Resume-grounded prompt and input encoding
```

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen pytest -q
uv build
```

Tests use deterministic embeddings, a tiny locally created transformer, and mocked
Qwen generation. They do not download pretrained weights or measure real-model
ranking/generation quality. They cover cold start, both feedback classes, isolation,
restart/continued learning, retries, corrections, atomic rollback, cross-connection
concurrency, corrupt snapshots, changed embeddings, API validation, and lazy startup.

The former extraction, qualification, handcrafted feature, MLP, profile parsing,
legacy/research, and offline training modules and their documentation/tests have
been removed. There is no compatibility layer. Use a new database; the service
refuses to repurpose an old database. Existing local data, environments, secrets,
and model artifacts are left untouched. Git history retains the old implementation.

The source is MIT. Original upstream notices remain in `LICENSES/`; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for current dependencies and model
license limitations. Model weights and third-party dependencies have their own terms.
