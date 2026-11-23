# ApplyPilot integration contract

Reference inspected: `zvictor12/pet_proj` commit
[`65d400cd4a1851e98f3c19dc44b391c444147447`](https://github.com/zvictor12/pet_proj/tree/65d400cd4a1851e98f3c19dc44b391c444147447).
The source was read directly; the repository was not modified.

| Inspected source | Confirmed behavior | Integration consequence |
|---|---|---|
| `app/models/mail.py` | `Mail` has mailbox, UID, message metadata and processing status. `Link` has an integer row ID, mail FK, URL, domain, and source vacancy ID. | No full vacancy title/description or user-account model here. Do not infer a user ID from mail UID or `Link.id`. |
| `app/parsers/vacancy_link_parser.py` | Returns `VacancyLink(url, source_domain, vacancy_id)`. Source vacancy ID is a numeric path suffix. Deduplicates `(domain, vacancy_id)` within a message and strips query/fragment from URLs. | Use source domain plus source ID for stable job identity across emails. |
| `app/services/email_ingestion_service.py` | Filters senders, parses mail and vacancy links, returns an `EmailProcessingResult` with mail metadata, status and links. | This supplies links, not complete job bodies. Fetching/extraction belongs to the caller. |
| `app/core/settings.py` | Pydantic settings read `.env`; DB URL uses `postgresql+asyncpg`, DB credentials, and hostname `db`. | Same driver stack; host `db` resolves inside that Docker network. An independently launched ML process needs its reachable database host. |
| `app/db.py` | `create_async_engine` and `async_sessionmaker(..., expire_on_commit=False)`. | ResumeAnalysis uses the same async SQLAlchemy approach, with pool disposal on shutdown. |
| `compose.yml` | `postgres:16`, persistent DB volume, health check and a separate Alembic migration command. | PostgreSQL 16 is the supported integration target. Apply migrations before serving requests. |
| `migrations/env.py` and `migrations/versions/9d5d52f32e4f_create_mail_tables.py` | Async Alembic uses `DATABASE_URL`, default `alembic_version`, and creates `mails` and `links`. | ResumeAnalysis uses its own `RESUMEANALYSIS_DATABASE_URL`, `ml_alembic_version`, and `ml_*` tables. No cross-project FKs. |
| `app/main.py`, `app/api/v1/router.py`, `app/api/v1/health.py` | The mounted API currently exposes a health endpoint. | No existing ML HTTP client or user-authentication API contract was found; do not claim an end-to-end integration already exists. |
| `pyproject.toml` | Python >=3.12, SQLAlchemy 2.x, asyncpg, Alembic, FastAPI. | Python 3.12 is supported and tested, alongside 3.11. |

## HTTP mapping

All existing endpoint paths and JSON fields are preserved. String IDs remain opaque
and are not automatically rewritten. The caller should choose one stable namespace
for each identity and consistently use it in both `/rank` and `/feedback`:

| Caller value | ML field | Rule |
|---|---|---|
| Authenticated application identity | `user_id` | Stable string up to 200 characters, supplied by the trusted backend. User identity is not implemented in the inspected mail/link models; it must come from the caller's authentication boundary. |
| Parser `VacancyLink.source_domain` + `.vacancy_id` | `vacancy_id` | Example `hh.ru:12345678`. For persisted `Link` rows, use `.domain` + `.vacancy_id`. A globally unique stable internal ApplyPilot string ID is also accepted. |
| Extracted job title | `title` | Required nonempty text, up to 500 characters. Email `subject` is not automatically a job title. |
| Full extracted job body | `description` | Required nonempty text, up to 20,000 characters. A URL or mail excerpt is not a replacement. |
| Explicit user's reaction | `liked` | JSON boolean only. No inferred or synthetic feedback. |
| Supplied resume and selected job | Cover-letter fields | `resume_text`, `vacancy_title`, `vacancy_description`, optional `instructions`/`language`; no database lookup or matching prerequisite. |

For example, parser values `source_domain="hh.ru"`, `vacancy_id="12345678"` map
to `"vacancy_id": "hh.ru:12345678"`. The same numeric ID from a different domain
has a different key. Repeated emails about that vacancy should reuse the same key.
Do not use per-email `Link.id` if that would give the same job multiple identities.
Already-stable internal IDs need not be changed. Applications control any alias
mapping between equivalent source domains; this service does not guess it.

ApplyPilot calls `POST /rank` with user ID and complete jobs, displays returned
scores/order, then sends `POST /feedback` for explicit reactions. It can reference
a job already ranked for that same user, or include `vacancy: {title, description}`.
`POST /generate-cover-letter` returns only `{cover_letter: "..."}`. Full HTTP
examples and response semantics are in the root README.

## Database and deployment boundary

Prefer a dedicated ML database on the existing PostgreSQL 16 instance. If sharing
a database, this service's migrations touch only its three `ml_*` tables and own
version table; Alembic autogeneration filters out unrelated tables. The reference
backend's unfiltered Alembic autogeneration could independently propose removing
unknown tables, so its generated migrations require review before deployment.
No modifications to its migrations or configuration were made here.

The ORM's `ml_users` table only stores preference model state and revision, keyed
by the supplied external ID. It contains no login, password, mailbox, or account
profile. FKs exist only between this service's ML tables. Feedback text is kept
as the reacted-to snapshot, independent of later changes to ranked job text.
Row locks and transactions cover both active feedback and numeric model state.
A database failure rolls back the transaction and produces a redacted `503`.

There is no authentication token verification in the current ML service. Keep it
on loopback or a private service network and have the trusted backend derive the
user ID after authenticating its caller. Do not expose it directly to the internet.
Use an existing authenticated TLS/mTLS gateway for untrusted network hops.
ResumeAnalysis does not implement IMAP, crawling, mail parsing or application
submission. These repositories stay independently deployable.

## Remaining integration work outside this repository

The reference backend still needs its own authenticated user identity, full job
text acquisition, and HTTP calls to these endpoints. This task does not implement
those missing components in `pet_proj`. Real recommendation quality, calibration,
and Qwen factuality have not been established by the mocked integration tests.
Old SQLite data is left in place and is not imported automatically.
