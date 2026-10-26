# Measured dataset inventory

Generated from checksum-verified originals. Sealed SkillSpan-derived test annotations are excluded from these measured totals. See [research and policy](README.md) for license and compatibility decisions.

| Dataset / split | Source rows | Samples | Unique texts | Skill annotations | Span mentions | Eligible samples |
|---|---:|---:|---:|---:|---:|---:|
| skillspan / train | 4,800 | 4,800 | 3,943 | 5,190 | 5,190 | 3,665 |
| skillspan / validation | 3,174 | 3,174 | 2,597 | 2,163 | 2,163 | 2,402 |
| tech / validation | 293 | 101 | 101 | 166 | 0 | 0 |
| house / validation | 231 | 77 | 77 | 144 | 0 | 0 |
| techwolf / test | 588 | 326 | 326 | 588 | 0 | 326 |
| synthetic_esco / train | 138,260 | 138,257 | 138,252 | 138,260 | 0 | 138,248 |
| skill_xl / validation | 5,847 | 2,281 | 2,157 | 3,957 | 0 | 1,969 |
| skill_xl / test | 6,295 | 2,643 | 2,529 | 4,514 | 0 | 2,493 |
| green / train | 233,628 | 9,634 | 9,630 | 12,987 | 12,987 | 9,608 |
| green / test | 8,050 | 336 | 336 | 676 | 676 | 336 |
| ru_titles / unsplit | 51,029 | 51,029 | 28,576 | 274,518 | 0 | 0 |

Total: **212,658 samples**, **188,090 unique texts** before quarantine. English: **161,629**; Russian title/list records: **51,029**. Eligible samples include auxiliary classification data and must not be treated as one NER training corpus.

Skill annotations count span/surface annotations and sentence/title target labels. BIO spans are measured against reconstructed tokenized text, not unavailable original advertisement offsets. Source rows mean sentences for SkillSpan/JSONL, token rows for Green, and label rows for TechWolf CSVs. Differences in these units prevent blind summation.

The machine-readable [inventory](inventory.json), [quality/leakage report](quality_leakage.json) and [validation coverage results](vocabulary_coverage.json) retain counts, hashes and definitions. Original data, source text, fingerprint sketches, per-record outcomes and vocabulary artifacts stay outside Git.
