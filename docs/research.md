# Source and dataset research

Reviewed on 2026-10-22. Temporary reference clones are outside the project checkout,
in `/workspace/reference-repos/`. The local implementation uses architectural ideas
and public libraries; no reference implementation was vendored. The later
[license audit](license_audit.md) records evidence, limits and unresolved rights.

## Reference projects

| Project | Reviewed commit | Files and decisions |
|---|---|---|
| [ResuMatch](https://github.com/JananiV07/resumatch) | `6eb8d49b6a5de28222d538f5c9350c639caa380b` | `backend/resumatch/features.py`, `skill_matcher.py`, `ranker.py`, `explain.py`, `api.py`, training script, requirements and tests. Reference for extraction, features, ranking and explanation. Its random beta-feature training scheme was not adopted. |
| [Nesta OJD DAPS Skills](https://github.com/nestauk/ojd_daps_skills) | `e73c2b5045793d891678ad387b2c7fda1756d236` | Extraction/config utilities, `multiskill_rules.py`, taxonomy mapper, ESCO config/table. Reference for separating extraction, normalization and mapping, including compound skill spans and hierarchical thresholds. |
| [SkillAlign](https://github.com/Y4SSERk/SkillAlign) | `cf0681df995838b14823825f0a44ec18c3903108` | Data processing, embedding generation, Neo4j ETL, occupation skill-gap service/routes, requirements and LICENSE/NOTICE. Reference for concept URIs, essential/optional requirements and skill gaps. Neo4j, FAISS, frontend and occupation-to-skill inference were not adopted. |
| [SkillSpan](https://github.com/kris927b/SkillSpan) | `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b` | README, LICENSE, JSONL train/dev/test, BIO skill/knowledge annotations and evaluation scripts. The full test split evaluates the dictionary without expanding it from test examples. |

## Compatibility and licenses

The three reference projects have MIT source licenses, but their full dependency
stacks are not installed. ResuMatch pins sentence-transformers 3.1.1 and scikit-learn
1.5.2. Nesta declares spaCy ~3.7.4, torch ^1.13.1 and transformers 4.41.2;
its pinned torch is incompatible with the selected Python 3.12 environment.
Its automatic shell model installation and pickle loading are not used.
The MVP uses independently implemented English/Russian dictionary rules. Nesta NER
has not been installed, run or trained.

| Model | Declared card license | Use |
|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | Downloaded and SHA256-checked frozen encoder, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. No separate upstream license file was found; see the audit. |
| `nestauk/en_skillner` | MIT | Metadata/reference only; not installed. |
| `nestauk/multiskill-classifier` | MIT | Metadata/reference only; not installed. |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | ResuMatch reference, not the local encoder. |
| `sentence-transformers/all-mpnet-base-v2` | Apache-2.0 | SkillAlign reference, not used. |

Four concept URIs were checked against Nesta's formatted ESCO v1.1.1 table.
Aliases and weighted relationships are local additions, not official ESCO relations.
SkillAlign's NOTICE describes ESCO as CC BY 4.0; the Commission's official legal
notice confirms a general CC BY 4.0 default for EU-owned website content, subject
to exceptions. The ESCO download page returned HTTP 403, so export-specific terms
remain unverified. `Taxonomy.import_esco()` accepts official CSV columns
`conceptUri`, `preferredLabel`, and `altLabels`. Attribution is retained in
`THIRD_PARTY_NOTICES.md`; this does not relicense ESCO material to MIT.

## Candidate Matching Synthetic

Source: `michaelozon/candidate-matching-synthetic`, MIT declaration, revision
`178ab864dcad9910c5670d43e4bdbbb901a11f18`. Three downloaded parquet tables contain
10,000 profiles, 2,500 vacancies and 2,500 positive-match lists. Checksums were verified.
The matches table contains only positive resume IDs, approximately 30 per job.

* `resumes`: `resume_id`, `role`, `seniority`, `years_experience`, `industry`,
  `education`, `skills`, `summary`, `experience_bullets`.
* `jobs`: `job_id`, `job_title`, `seniority`, `industry`, `must_have_skills`,
  `nice_to_have_skills`, `description`, `responsibilities`, `requirements`.
* `matches`: `job_id`, `relevant_resume_ids`.

Skills, bullets and relevant IDs are actual lists in the inspected files.
Profiles/jobs were generated upstream with Qwen, and matching labels are heuristic.
These are downloaded text records, not random numeric features, but they are not
employer decisions. Upstream precomputed embeddings are not used; local E5 computes
its own representations.

Additional negatives are explicitly weak synthetic labels: a candidate must be absent
from the published positive list and cover less than 60% of required skills after
combining structured and text evidence. The closest eligible candidates become hard
negatives. Absence from the positive list alone is not a negative label or evidence
that a real candidate lacks a skill.

## Resume-Job Fit Merged

Source: `med2425/resume-job-fit-merged-v1`, revision
`f46f355acddab1368a7386cbe94d42fd42767d88`. Only card/schema metadata was inspected;
its payload was not downloaded or used for training.

Fields are `resume`, `jd`, `label`, `source`, `resume_domain`, `jd_domain`.
Labels are `Good Fit`, `Potential Fit`, and `No Fit`, not binary integers.
The card describes Qwen2.5-32B labels and merging two other datasets. No license is
declared in the inspected card/metadata, including its named upstream inputs.

The separate `prepare_med2425` adapter requires reviewed license evidence and an
explicit Potential Fit policy. Good Fit maps to 1, No Fit to 0, and Potential Fit
is excluded by default. The manifest-field check is not a legal permission grant.
Refusal without evidence has been tested; full upstream-data execution has not.

## SkillSpan

The README releases code and data; the repository LICENSE is MIT, copyright
(c) 2023 Mike Zhang. No separate data license was found. Retained notices and the
requested paper citation are in `THIRD_PARTY_NOTICES.md`.

Files named `.json` are JSONL with `idx`, `tokens`, `tags_skill`, `tags_knowledge`,
and `source`. This is extraction annotation, not hiring-fit labels. It was not
converted into hiring decisions. Evaluation combines skill/knowledge gold spans
across all 3,569 test sentences. Aggregate results and short missed-span phrases
are committed; complete dataset records remain outside Git.

## Leakage prevention

Candidate Matching entities are partitioned before pair sampling using a seeded hash
of normalized profile summaries and vacancy descriptions, with 70/15/15 proportions.
Identical text gets the same split despite different IDs or supplemental fields.
Cross-partition positive edges are discarded; negatives are drawn only within a split.
Final checks reject candidate/vacancy ID or normalized-text overlap across splits.
This yields fewer pairs than all published positive edges.

For existing arbitrary pairs, connected components of the candidate/vacancy graph
also join duplicate-text identities. Fewer than three independent components cause
an error rather than a random row split. Near-duplicates and paraphrases remain a
risk, and template-based synthetic data does not independently validate real-world fit.
