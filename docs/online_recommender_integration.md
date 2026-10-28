# ApplyPilot integration contract

This integration supplies tested local Python adapters in ResumeAnalysis. It does
not modify ApplyPilot or claim that new HTTP endpoints/tables are already deployed.
Pinned source links/hashes are in [source evidence](online_source_evidence.json).
Generated [JSON schemas](online_schemas/) describe the profile, vacancy,
recommendation, feedback event and source document contracts. Regenerate them with
`python scripts/export_online_schemas.py` after an intentional schema change.

## Actual upstream state

| Inspected source | What exists | What is absent or incomplete |
|---|---|---|
| `main`, `65d400cd4a1851e98f3c19dc44b391c444147447` | `mails`/`links` ORM and migration, email ingestion, vacancy-link parser, health route. | No structured vacancy, profile, recommendation or feedback API/table. |
| `mail-repo/ingestion-service`, `1c464cf3784a2835c332098729a66895df013c30` | `VacancyDescDTO`, `Vacancy` ORM/repository, HTML fetching/enrichment, mail/link/sync-state routes. | No vacancy-table migration or recommendation/profile/feedback endpoints. DTO module has a misplaced `from __future__` import; mapper uses `**dto.model_dump` without calling it. These source-level issues prevent treating the branch as a working deployed contract. |
| Other public feature/infrastructure branches | Earlier mail, link and database work. | No recommendation/profile/feedback implementation found. |

ApplyPilot declares no repository license. No implementation was copied. The
adapters independently map the observed public field shapes; host fixes and
database migrations remain outside this repository's scope.

## Host responsibilities and local interfaces

- `RecommendationProfileProvider.get_profile(user_id) -> StructuredProfile` supplies
  the authenticated user's current profile and immutable version. Store user edits
  as a new version; do not change content under an already recorded version.
- `RecommendationVacancyProvider.list_vacancies(user_id) -> list[StructuredVacancy]`
  supplies deduplicated candidate content, with stable source-qualified IDs.
- `LocalRecommendationClient.recommend(user_id, limit=20)` returns recommendations,
  their IDs, qualification evidence, interest scores and independent hard-constraint
  status. The new path uses River; it does not load the legacy MLP.
- `LocalRecommendationClient.record_feedback(...)` forwards explicit labels and
  client event IDs. Pass `recommendation_id` for the exact shown snapshot and
  `corrects_event_id` for an intentional correction.
- `LocalRecommendationClient.cover_letter_input(...)` returns the existing
  `AnalysisRequest` with a selected resume ID, confirmed facts and skill-specific
  experience context. Interest is not converted into qualification evidence or
  candidate experience. Cover-letter generation remains a separate explicit step.

The synchronous interfaces can be placed in a host worker thread if the web
application is asynchronous. No Kafka, Redis, Kubernetes or cloud service is added.
The host controls database lifecycle, authentication, candidate visibility, profile
storage and any external page fetching. No user is inferred from an email address
or mailbox by this library.

## Vacancy mapping

`adapt_applypilot_link(link, resolver)` accepts the current parser's
`source_domain` or the ORM/DTO's `domain`, plus `url` and `vacancy_id`. The ID becomes
`domain:vacancy_id`, preventing collisions between job sources. URL/domain mismatch
fails. A host `VacancyContentResolver.resolve(link) -> ResolvedVacancy` must supply
actual text and optional verified salary; a link alone is not a parsed vacancy.

For the richer feature branch, use
`adapt_applypilot_description(link, description, verified_salary=None)`. It maps
`title`, `description_plain` and `location`. Missing plain text fails rather than
treating HTML as extracted content. Free-text salary and untyped enrichment do not
establish numeric pay, remote work, skills or experience requirements. The host may
provide a validated `Salary` separately. Organization, dates and employment type
remain host metadata; this MVP does not silently treat them as preference evidence.

## Migration and operational behavior

`from_legacy_profile` preserves qualification fields and moves legacy remote/location
constraints into the new explicit preferences object. `from_legacy_vacancy` wraps
existing vacancy fields with a stable ID. These are additive replacements; old
`InferenceClient.process(vacancy) -> VacancyAnalysis`, checkpoint loading and
cover-letter interfaces remain available.

The feedback repository owns its SQLite schema (`user_version=1`) and does not
alter ApplyPilot's PostgreSQL tables. To use the host database later, implement
`FeedbackRepository`/`FeedbackSession` with equivalent transaction, user isolation,
idempotency and immutable-snapshot guarantees. Persistence commits must precede
publishing learned cache state.

A preference reset keeps audit history and rejection suppression. An explicit
correction can restore a previously rejected vacancy. Profile confirmation and
hard-constraint uncertainty should remain visible in the host UI; an interest score
does not authorize applying or emailing. No email is sent by these adapters.

Tests cover both observed link shapes, the unmerged vacancy DTO shape, unknown
salary/enrichment, cross-user snapshot rejection, the cover-letter contract,
restart behavior and offline calls. Live host endpoints/database operation were
not exercised because the required integration does not yet exist.
