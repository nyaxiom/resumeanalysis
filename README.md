# ResumeAnalysis

A small ML service for ApplyPilot: rank vacancies using each user's own likes and
dislikes, and draft a cover letter from supplied resume text. One FastAPI process,
one frozen multilingual E5 encoder, River logistic regression, PostgreSQL 16, and one
Qwen Instruct inference backend. Neither pretrained model is fine-tuned.

## Install and run

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are supported. A PostgreSQL
16 database is required. For a fresh local development database:

```bash
read -r -s -p 'Local PostgreSQL password: ' POSTGRES_PASSWORD
export POSTGRES_PASSWORD
docker run --detach --name resumeanalysis-postgres \
  --publish 127.0.0.1:5432:5432 \
  --env POSTGRES_USER=resumeanalysis --env POSTGRES_DB=resumeanalysis \
  --env POSTGRES_PASSWORD --volume resumeanalysis-pg-data:/var/lib/postgresql/data \
  postgres:16
docker exec resumeanalysis-postgres pg_isready -U resumeanalysis -d resumeanalysis
```

Wait for `accepting connections`. This example creates a new named volume and
leaves existing databases alone. If port 5432 is already occupied, change the
published host port and the URL accordingly. Reuse an existing PostgreSQL 16
instance if available; a separate ML database on that instance is recommended.
Copy `.env.example` only when no `.env` exists, then edit it. Set
`RESUMEANALYSIS_DATABASE_URL` to the database you created, including its username
and password. Percent-encode special characters in URL credentials. Never commit
credentials; production should use your existing secret-delivery mechanism.

```bash
uv sync --frozen --extra dev
test -e .env || cp .env.example .env
# Edit .env as needed; load it into the shell explicitly.
set -a
. ./.env
set +a
uv run --frozen alembic upgrade head
uv run --frozen alembic current
uv run --frozen uvicorn resumeanalysis.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`.env` is a local shell configuration file and is not read automatically. Do not
commit it. Interactive API documentation is at `http://127.0.0.1:8000/docs`.
Apply migrations once before starting application workers. FastAPI never creates
tables or applies migrations; it initializes an async SQLAlchemy pool and disposes
it on shutdown. The initial migration creates only `ml_users`, `ml_vacancies`, and
`ml_feedback`, with a separate `ml_alembic_version` revision table. Repeating
`alembic upgrade head` is safe. No pretrained model is loaded at startup.
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
`description`. Up to 100 vacancies are accepted per request. Use a stable opaque
string `user_id` supplied by the trusted backend. For source IDs, namespace each
`vacancy_id`, for example `hh.ru:12345678`; existing globally unique ApplyPilot
string IDs remain valid and are not rewritten. Both IDs are limited to 200
characters. Text fields cannot contain NUL characters, which PostgreSQL rejects.
Send complete extracted descriptions, not just URLs or email subjects.

```bash
curl -s http://127.0.0.1:8000/rank -H 'Content-Type: application/json' -d '{
  "user_id": "user-1",
  "vacancies": [
    {"vacancy_id": "hh.ru:12345678", "title": "Sales specialist", "description": "Develop customer accounts."},
    {"vacancy_id": "career.habr.com:87654321", "title": "Python developer", "description": "Build and maintain Python APIs."}
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
  "user_id": "user-1", "vacancy_id": "career.habr.com:87654321", "liked": true
}'
curl -s http://127.0.0.1:8000/feedback -H 'Content-Type: application/json' -d '{
  "user_id": "user-1", "vacancy_id": "hh.ru:12345678", "liked": false
}'
```

Alternatively send data directly, without calling `/rank` first:

```bash
curl -s http://127.0.0.1:8000/feedback -H 'Content-Type: application/json' -d '{
  "user_id": "user-1", "vacancy_id": "hh.ru:99999999", "liked": true,
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
The last serialized update is authoritative; ApplyPilot should serialize a user's
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

PostgreSQL stores user-scoped vacancies (`ml_vacancies`), current feedback
text/labels (`ml_feedback`), and numeric JSON model snapshots and revisions
(`ml_users`). `ml_users` is preference state keyed by an external user ID, not an
authentication or account system. There are no foreign keys into ApplyPilot's
schema. The feedback key `(user_id, vacancy_id)` is unique; corrections replace
the current label and preserve original feedback order, rather than appending a
second active training example. This is current training history, not a full
immutable audit log of every superseded reaction.

Feedback and model updates commit in the same transaction. A first-use upsert and
`SELECT ... FOR UPDATE` on the corresponding `ml_users` row serialize requests
for that user across workers. Other users can use independent transactions. No
in-memory mutable classifier is shared between requests.
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

All database access uses SQLAlchemy 2.x `AsyncEngine`, `async_sessionmaker`, and
asyncpg. SQL and ORM operations live in `storage.py`; the recommender consumes
user-scoped storage operations. E5 inference, River learning/replay, and scoring
run in the threadpool, while database awaits remain on the event loop. Qwen runs
in a worker thread and never opens or holds a database transaction.

A single worker is recommended to avoid duplicating heavyweight models in memory.
Restoring/re-embedding a user's history or resolving an ID-only feedback cache
miss may hold that user's row lock during E5 computation. Same-user requests wait;
PostgreSQL lock and command waits are bounded at 30 seconds and connection startup
at 10 seconds. Database failures produce a redacted `503`, and the transaction is
rolled back. A caller can retry the current rating, but should not retry superseded
ratings. There is no queue or automatic replay worker.

Use PostgreSQL backups and appropriate database permissions: the database stores
user IDs, vacancy text, feedback, and preferences. The old SQLite databases are
not read, migrated, or deleted. `RESUMEANALYSIS_DATABASE_PATH` is no longer used.

## Configuration

Every setting has a `RESUMEANALYSIS_` prefix. The complete editable example is
[.env.example](.env.example). Model revisions must be immutable 40-character commit
SHAs when using a remote model. Empty local model paths select Hugging Face loading;
an empty cache directory disables embedding caching.

| Variable suffix | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://localhost/resumeanalysis` | PostgreSQL URL; set explicit deployment credentials |
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

The reference [zvictor12/pet_proj](https://github.com/zvictor12/pet_proj) was inspected
at `65d400cd4a1851e98f3c19dc44b391c444147447`. Its real implementation currently
extracts vacancy links from email and models mail/link records. It does not yet
provide application user records, full vacancy descriptions, or an ML HTTP client.
The source-by-source findings and mapping are in
[docs/applypilot_integration.md](docs/applypilot_integration.md).

The calling backend must provide a trusted stable `user_id`, fetch/extract complete
vacancy titles and descriptions, and send source-qualified IDs or stable internal
IDs. It collects explicit reactions and calls `/feedback`, and displays generated
letters for review. Sending applications remains the backend's responsibility.
This service does not parse mail, fetch job sites, or send applications.

There is no interservice authentication in this service or the inspected reference
contract. Deploy on loopback or a private network accessible only to the trusted
backend; do not expose these endpoints directly to the internet. ApplyPilot must
authenticate users and must not forward arbitrary client-supplied user IDs.
For untrusted network hops, put authenticated TLS/mTLS at your existing gateway.
The two repositories remain independent, communicating over HTTP.

## Layout and development

```text
src/resumeanalysis/
  main.py           # FastAPI app and three routes
  schemas.py        # Request and response contracts
  config.py         # Environment settings
  embeddings.py     # Lazy E5, chunking, batching, cache
  recommender.py    # Per-user River training, ranking, replay
  storage.py        # Async PostgreSQL operations and user row locks
  models.py         # ML-only SQLAlchemy ORM tables
  cover_letter.py   # Lazy Qwen inference and output checks
  prompts.py        # Resume-grounded prompt and input encoding
```

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen pytest -q -m "not integration"
uv build
```

Fast unit tests require no running database. All model tests use deterministic
embeddings, a tiny locally created transformer, or mocked Qwen, without downloading
pretrained weights or measuring real-model quality.

Run real PostgreSQL integration tests against a disposable test instance:

```bash
docker run --detach --name resumeanalysis-pg-test \
  --publish 127.0.0.1:55432:5432 \
  --env POSTGRES_USER=resumeanalysis_test --env POSTGRES_DB=resumeanalysis_test \
  --env POSTGRES_HOST_AUTH_METHOD=trust postgres:16
docker exec resumeanalysis-pg-test pg_isready -U resumeanalysis_test -d resumeanalysis_test
export RESUMEANALYSIS_TEST_DATABASE_URL=postgresql+asyncpg://resumeanalysis_test@127.0.0.1:55432/resumeanalysis_test
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen pytest -q
```

The trust-auth example is exclusively for an isolated local test container with
no real user data. Test credentials need permission to create/drop databases.
The fixture accepts only a `resumeanalysis_test...` database on PostgreSQL 16,
creates a fresh randomly named database for each integration test, applies real
Alembic migrations, and drops only that generated test database afterwards. It
never clears the database supplied in the URL. With no test URL, integration
tests are explicitly skipped; SQLite is never used as a substitute. CI supplies
PostgreSQL 16 and runs the full suite on Python 3.11 and 3.12.

Tests cover repeatable migrations and preservation of unrelated tables, cold
start, both classes, user isolation, persistence and recovery, duplicates,
corrections, atomic rollback, multi-pool concurrency and row locks, namespaced
IDs, version changes, API validation, generation, lazy startup, and pool disposal.
All old local datasets, environments, secrets, and pretrained artifacts remain
untouched. There is no automatic migration from the previous SQLite schema.

The source is MIT. Original upstream notices remain in `LICENSES/`; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for current dependencies and model
license limitations. Model weights and third-party dependencies have their own terms.
