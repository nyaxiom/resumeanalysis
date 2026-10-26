# Loss-aware dataset adapters

The new `datasets` commands operate on UTF-8 **JSONL**. They do not change the
existing `prepare`, `prepare-med2425`, canonical schema or E5 training behavior.
Those older preparation commands retain their historical behavior and are not
certified by these new adapter guarantees. Use this interface to audit new imports;
do not assume prior datasets were originally converted without loss.

## Source inspection and conversion

```bash
resumeanalysis datasets inspect --input data/source --dataset example --revision export-v1
resumeanalysis datasets convert --adapter canonical --input data/source \
  --dataset example --revision export-v1 --provenance 'Local licensed export' \
  --output data/converted --strict
resumeanalysis datasets audit --input data/converted
```

`--strict` is the default. Every nonempty or blank source line receives an outcome:
`converted`, `converted_with_loss`, `excluded` by a named policy, or `rejected`.
Malformed JSON, duplicate JSON keys, nonfinite numbers and nonobject records are
rejected. An invalid line is not skipped. Diagnostics use opaque source identifiers
and field paths; source text and unexpected label values are not printed.

An input directory's `.jsonl` files are read in sorted order. Files named
`train.jsonl`, `validation.jsonl` and `test.jsonl` preserve their partitions.
Other files produce `records.jsonl`; there is **no automatic splitting** and no
invented label. Use the existing entity-aware splitter only after labels, licensing
and enough independent components have been established. All original partitions
are required by the shared comparison/augmentation commands.

Non-JSONL entries and subdirectories are listed as excluded by the declared file
format policy; they are not recursively imported. Empty JSONL files in an otherwise
readable source cause a reported quality failure. Source file hashes are checked
again before publishing to detect changes during conversion.

Strict mode rejects unknown fields, including nested keys and fields inside
arrays of objects. The canonical adapter directly maps declared canonical fields
and validates types without automatic string-to-number coercion. Label values must
be actual integers `0` or `1`, never booleans or guessed fit categories. Numeric
experience bounds and other canonical constraints are validated on every output.
The canonical schema supplies its existing defaults; source missingness is reported
separately so defaults are not confused with originally supplied information.
Pydantic's accepted integer-to-float normalization is explicitly recorded as a
canonical type conversion. JSON decimal literals that cannot round-trip through
the supported float representation are rejected with a numeric-precision diagnostic.
Original bytes or their hashes remain available under the source retention policy.

Any rejected row fails the strict run after diagnostics have been written. No
normalized files are published from a failed run, even if some rows were valid.
Config, filesystem or unreadable-source failures may prevent report creation.
Output directories must be empty and separate from the source; inputs are never
rewritten. There is no force/overwrite option.

`--permissive` is an explicit opt-in. It can omit unknown fields and retain valid
rows after rejections, with per-row warnings, coverage and reconciliation. It does
not coerce unexpected types without a declared transform, guess missing labels,
disable provenance, bypass version checks or permit cross-partition leakage.
Every excluded field has a recorded policy or an unmapped-field diagnostic.

## Custom mappings and versions

Use `--adapter mapped --mapping configs/example_adapter.json`. The included example
is a small hypothetical schema, not a claim about a public dataset's current schema.
Rules use JSON Pointer source/target paths and one disposition:

| Disposition | Meaning |
|---|---|
| `direct` | Exact typed value copied to a canonical field |
| `transformed` | Named conversion, accepted source types and target recorded |
| `auxiliary` | Not an ML feature; retained through the source policy |
| `excluded` | Deliberately omitted for the stated reason |
| `unsupported` | No rule; strict failure, explicit loss in permissive mode |

Renames are recorded independently from type conversions. `float`, `integer`,
`string` and exact typed `label` mappings are supported. Transforms declare
`source_types`; unexpected types and failed conversions reject the row. Integer
conversion never rounds fractional values. Label mappings must be unique and
explicit; an unmapped category fails. Version fields can require an exact typed
version, and changed or missing versions reject the record in both modes.

Nested scalar fields and primitive lists can map directly. Whole dictionaries or
arrays of objects cannot silently flatten into model fields. Declare their leaf
rules or preserve/exclude the subtree with an explicit reason. Auxiliary/excluded
subtree rules deliberately allow all nested fields underneath them: every observed
nested path is still enumerated in coverage. Review these broad allowlists carefully
if the upstream metadata schema changes. Overlapping source rules or duplicate
canonical targets are rejected when loading a specification.

Exclusion policies match one explicit source value and record a reason for each
row. They do not rescue unknown fields from strict schema validation. Excluding
rows can alter class balance; configure quality expectations for the intended
population rather than assuming that an explicit exclusion is statistically safe.

## Retention, provenance and reports

Source files remain unchanged. By default, the adapter follows a metadata-only
retention policy: source filename/line, dataset/revision/provenance, deterministic
row identifier, raw-line SHA256 and normalized hash/index are retained. Full
reversibility is explicitly marked unavailable; the custodian must retain originals
if subsequent reconstruction is needed. Identifiers include dataset, revision,
filename, file hash, line number and row hash, so they are stable for an identical
source snapshot and change when that snapshot changes.

If privacy, licensing and storage policy permit, explicitly add `--retain-source`
to keep each original raw line **byte-for-byte** under `sources/<source_id>.jsonl`.
This also preserves unmapped/auxiliary information separately from `PairRecord`.
Even rejected and excluded rows are retained in that mode. Original container
files are not copied as a second full archive, and non-JSONL sidecars are outside
the format's input scope. No source fields are injected into canonical objects.
The repository ignores `data/` and `artifacts/`; keep exports and local reports there.
Inspect any alternate output path before committing because raw snapshots and
even linkage metadata can be personal information. Licenses/consent remain external
requirements; passing this technical adapter does not establish rights to the data.

Successful conversion of a prepartitioned input produces:

```text
data/converted/
  train.jsonl
  validation.jsonl
  test.jsonl
  manifest.json
  conversion_report.json
  conversion_summary.md
  quality_report.json
  sources/                 # only with --retain-source
```

The conversion report includes every row outcome, source links, schema coverage,
unknown paths, missing/null values, renames, transformations, label mappings and
counts. The Markdown summary lists counts, field dispositions and representative
opaque problem identifiers. The quality report includes preconversion mapped
skill counts/text lengths and source labels; postconversion missing rates,
skill/text-length distributions, label balance, numerical validation, exact
source/normalized duplicates and leakage checks. The manifest hashes all produced
files. `datasets audit` verifies file integrity, row hashes/indexes and lineage,
reconciles counts and reruns normalized quality checks. These hashes detect
accidental modification; they are not signatures against an adversary rewriting
the entire manifest and reports.

Every field's disposition is measured. An auxiliary field being preserved, or a
Pydantic object validating successfully, does **not** mean all source information
is available to the ML model. Raw-only relationships and source schema meaning
can still be unavailable; their paths/disposition and reversal limit remain visible.

## Configurable quality policy

```bash
resumeanalysis datasets convert --adapter canonical --input data/source \
  --dataset example --revision export-v1 --provenance 'Local licensed export' \
  --output data/converted --quality-policy configs/example_quality_policy.json
```

The included policy is illustrative, for a curated balanced synthetic export only.
Choose thresholds and explain their rationale for the actual dataset. No universal
missingness, class-balance or fuzzy-similarity threshold is imposed by default.
Quality violations fail the run in both modes. Configurable checks include maximum
missing rates, positive-rate bounds and drift against an explicit reference rate.
Exact duplicate counts are always reported rather than automatically deduplicated.

Near duplicates use optional `SequenceMatcher` similarity of normalized combined
resume/vacancy text. The threshold and maximum pair comparisons are configurable;
the report declares complete or partial coverage and representative source IDs.
It is a diagnostic, not a semantic duplicate detector or proof of independence.
The mandatory leakage check uses existing candidate/vacancy IDs and normalized
content across all supplied partitions. Changed IDs plus heavily rewritten text
can evade that check; human review or a dataset-specific semantic detector is still
needed. Source partitions, source duplicates and text equivalence need particular
care before claiming independent evaluation.

See [Qwen experiments](qwen.md) for label-free augmentation lineage. Removing
candidate information invalidates automatic reuse of original fit labels. Shared
evaluation consumes the normalized canonical schema; partial observations remain
separate and never become the official labeled test set.
