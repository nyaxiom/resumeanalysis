# Additional dataset and base-model review

Reviewed on **2026-10-23 UTC**. The downloadable, pinned dataset inventory is in
[the catalog](../../configs/skill_datasets.json). This report covers additional
candidates and reasons for deferral. A publisher's license declaration, a copied
license text, and verified rights in an original job advertisement are different
pieces of evidence. Counts below are labeled by their source; an unknown count
does not mean zero.

## Additional dataset candidates

| Dataset and pinned source | Samples and annotated skills | Language and format | Supported task and decision |
|---|---|---|---|
| [JobBot](https://github.com/SwathyAnand100/JobBot/tree/e2f38a1a8cf0714e973bf9d4b713cacb5a4cafc1) | README reports **700** manually annotated sentences, split **490/105/105**. Total skill mentions and unique skill names are not reported. The described annotation files and trained checkpoint were absent from the inspected tree. | English LinkedIn job sentences; README describes span annotations converted to BIO and a separate manually normalized ESCO evaluation. The annotation payload/schema could not be independently checked. | Potential span extraction and normalization, **unavailable for this pipeline**. The README's academic-purpose statement is not an open-source license; no LICENSE/NOTICE or separate annotation grant was found. Do not copy notebooks, annotations or checkpoints without permission. The reported F1 is not a reproduced SkillSpan result. |
| [SkillSkape author mirror](https://huggingface.co/datasets/jjzha/skillskape/tree/d3a3c6e49a08157068b6b258acf4659269ab76cd), associated with [JobSkape](https://github.com/magantoine/JobSkape/tree/2dec11084441240edbb69611ee848d3c54aeb02e) and the [NLP4HR 2024 paper](https://aclanthology.org/2024.nlp4hr-1.4/) | HF viewer metadata reports **22,340 sentence-label rows**, split **16,345/2,751/3,244**. These are not verified unique sentence counts or span counts. Distinct skill labels are unknown. The viewer's source checksums reference **`aa53ffbe45d0a8d4647e416111f633fe7d47dc41`**, an older revision than the inspected mirror: counts must not be presented as verified counts of the pinned current files. | English synthetic job sentences. Viewer schema is `sentence: string`, `label: string`; no offsets. Generated combinations of ESCO skills and job sentences, rather than human span annotations. | Skill-to-taxonomy matching or sentence classification. **Deferred**: the mirror has no license declaration or LICENSE. The original GitHub repository has a root MIT license, but the data-specific grant, mirror lineage, generator terms and ESCO provenance should be clarified before importing a selected dataset release. The GitHub and HF layouts are not assumed identical. Synthetic labels can omit incidental skills. |
| [Sayfullina soft-skill mirror](https://huggingface.co/datasets/jjzha/sayfullina/tree/624b4f8ff58be7de4c5df1565b998cd8c8fd92f7) | Card and viewer agree on **7,411 sentences**, split **3,705/1,855/1,851**. Total annotated mentions and unique skill surfaces were not verified because annotation files were not downloaded. | English, Adzuna UK source. `idx`, token lists, `tags_skill` with `B-SOFT/I-SOFT/O`; no document delimiters. The mirror's author created the split. | Soft-skill span extraction; potentially useful after legal clearance. **Excluded now:** card explicitly declares `license: unknown`, with no standalone license. Original work: Sayfullina, Malmi and Kannala, *Learning representations for soft skill matching* (AIST 2018). Soft-skill-only annotation is not exhaustive general-skill gold, and missing document groups complicate leakage checks. |
| [sunk999/ner_job_skill](https://huggingface.co/datasets/sunk999/ner_job_skill/tree/afdfcdf0fe5d832e64108eef92f348802c59acc2) | Card and viewer show **843 examples**, split **674/169**. Mention count and label vocabulary are unknown. | No language declaration or collection description. Parquet columns contain `requirement: list[string]` and `label: list[int64]`. A tag-ID-to-BIO mapping is absent from the inspected card. | Potential token classification, **not safely convertible yet**. Card declares MIT, but no LICENSE/NOTICE, source attribution, annotation instructions or integer-label meanings were found. Do not guess the language or the tag mapping. |
| [AlekseyScorpi/vacancies_prompts](https://huggingface.co/datasets/AlekseyScorpi/vacancies_prompts/tree/13669ea61b357bd855834f07d80f4b95229dc309) | Card and viewer report **48,564** rows in one training split. Skill mention and distinct-label counts are not supplied. | Russian, real hh.ru-derived vacancies repackaged as `prompt` and `description` strings. Prompts were constructed from employer, role, experience, schedule and keys. | Vacancy-text generation or weak structured supervision, **not human NER gold**. Card-only MIT declaration; no standalone license. Original Kaggle/hh.ru terms and prompt transformations need review. Do not mine supposed gold offsets out of generated prompts. |
| [trewwxsav/IT_vacancies_from_hh.ru](https://huggingface.co/datasets/trewwxsav/IT_vacancies_from_hh.ru/tree/268402c604e97c9bcc68ce518b4bd12f3f930f22) | Count and annotated-skill count **unverified**. The viewer failed to build; its displayed zero rows is not an empty-dataset finding. | One `all_vacancies.csv` file. The README only declares `license: apache-2.0`; language, columns, annotation process and label meanings are undocumented. Russian provenance is suggested by the title, not verified language metadata. | **Deferred** until schema, original terms and any skill annotations are established. A bare Apache declaration does not establish hh.ru's underlying advertisement rights. No payload imported for this review. |
| [Dashkins/russian-job-retrieval-dataset](https://huggingface.co/datasets/Dashkins/russian-job-retrieval-dataset/tree/de61951ca8fe5f56c6ca9f8c65f91a73349ee0ac) | Card describes generator parameters of **250,000 resumes, 15,000 vacancies and 2,500,000 relevance pairs** over 12 IT directions. The card itself says actual artifact counts require verification. The viewer returned HTTP 500; these are **reported generation parameters**, not measured counts. Skill annotation counts are unknown. | Russian synthetic `resumes.csv`, `vacancies.csv`, `qrels.csv`; resume/vacancy IDs, text and pair relevance. | Matching/retrieval, **not span extraction**. Card declares MIT but no separate LICENSE was listed. Synthetic relevance does not provide human skill or requirement gold. No downstream matching retraining is warranted for this task. |
| [alekseevpavel04/ru-it-vacancies](https://huggingface.co/datasets/alekseevpavel04/ru-it-vacancies/tree/c522c3654cdf9dbef49b8095bc5344447dc3234f) | Card and viewer show **25,818** Rostrud rows, **764** SuperJob field-only rows, **51,170** Telegram aggregate rows and **426** Habr aggregate rows. The card reports **18,864** Rostrud records with text. These units must not be summed as annotated vacancy sentences. Counts of skill mentions/unique labels were not verified. | Mainly Russian job data. Per-vacancy skill lists are **model-extracted**, without offsets; some configurations contain only aggregated statistics. Card states that the texts of SuperJob, Habr and Telegram sources are withheld. | Useful candidate for later weak supervision and domain-shift inspection, **not independent human extraction accuracy**. Actual `LICENSE.md` separates **CC BY 4.0 for derived fields/aggregates** from Rostrud open-data terms for source text and portal fields. Rostrud attribution and a link to trudvsem.ru are required by the publisher's instructions. The primary portal terms need independent review before importing text; this work downloads metadata only. |

The already selected
[MrAlexGov/it-job-title-skills-ru](https://huggingface.co/datasets/MrAlexGov/it-job-title-skills-ru/tree/9863ee993040c43305627c72227fc50415e257ad)
contains **51,029** real vacancy-title/skill-list examples and **1,119** normalized
skill labels, according to its card/stats and the local inventory. It deliberately
omits full job descriptions, removes 71 general skills and filters rare skills.
Its CC BY 4.0 card credits two Kaggle releases, with upstream CC BY 4.0 and
Apache-2.0 terms claimed by the publisher. Both Kaggle verification requests were
blocked with HTTP 403. The catalog therefore restricts it to research inventory.
Title-to-skill labels do not establish mention spans, and technical English tags
inside Russian titles do not establish Russian-language NER coverage.

## Metadata-only counts for sealed TECH/HOUSE splits

Hugging Face dataset-server `/info` and `/size` metadata agree on the following
annotation-row counts. The `/info` source-file references match the exact pinned
revisions and catalog byte sizes: TECH **`5f0cb05df03cd1cd5048ff1d64639bf9bd4cb36f`**
and HOUSE **`dcb313cad615a8f5148ef5addc01ef090fc9f295`**.

| Dataset | Validation annotation rows | Sealed test annotation rows | Total rows |
|---|---:|---:|---:|
| [TECH metadata](https://datasets-server.huggingface.co/info?dataset=TechWolf%2Fskill-extraction-tech) | 293 | 1,164 | 1,457 |
| [HOUSE metadata](https://datasets-server.huggingface.co/info?dataset=TechWolf%2Fskill-extraction-house) | 231 | 891 | 1,122 |

These are metadata-verified counts of CSV annotation rows, not unique sentence
counts or verified positive-skill counts. A row can carry an unmapped outcome.
The sealed test CSVs were not parsed and their labels, unique skill types and
sentence-group counts were not inspected. Metadata evidence and snapshot hashes
are retained in the evidence JSON. This operation does not evaluate either test
split or make them available for dictionary fitting or model selection.

## Skill-XL and graded retrieval variants

[Skill-XL's pinned card](https://huggingface.co/datasets/TechWolf/Skill-XL/blob/221cbec139621712633cbc2d27dcd2daa68e48ed/README.md)
contains only a CC BY 4.0 declaration. Its commit history supplies no additional
annotation instructions. CSV inspection established **5,847 development rows**
and **6,295 test rows**, with **3,957/4,514 nonempty skill-label rows** and
**1,662/1,740 distinct nonempty labels**, respectively. The row layout repeats
sentences across labels and subsets; those row counts are not unique examples.
There are **530/295 rows with `relevant=True` but no skill label**. Missing skill
labels must not become exhaustive negative skill annotations. No span offsets are
provided, and `cluster` is not an annotated text boundary. These checks describe
annotation structure and inventory; no test performance or model selection used
the test file.

Two newer TechWolf datasets were also reviewed through their cards:

- [Skill-extraction-SkillSkape-graded](https://huggingface.co/datasets/TechWolf/Skill-extraction-SkillSkape-graded/tree/af3632993ede994bb36166f84c914e388fd1a579)
  declares CC BY 4.0 and describes **100 validation / 500 test sentence queries**,
  a **13,891-skill ESCO v1.1.0 corpus**, and **1,389,100 / 6,945,500** query-skill
  relevance judgments. These counts are card-reported. The card attributes
  SkillSkape sentences and ESCO, but the reviewed source mirror's missing license
  remains a provenance issue. The judge-label construction is not established as
  independent human span annotation.
- [Skill-normalisation-ESCO-graded](https://huggingface.co/datasets/TechWolf/Skill-normalisation-ESCO-graded/tree/d1de594eb6b6259651b83b5f927b35383018ac12)
  declares CC BY 4.0 and describes **50 validation / 450 test surface-term
  queries**, the same **13,891-skill** corpus, and **694,550 / 6,250,950** judgments.
  These are card-reported counts of graded retrieval targets, not skill spans.

Both use BEIR-style query/corpus/qrels Parquet files with scores 0–4. A score of 1
explicitly means a plausible domain skill **not mentioned in the sentence**;
scores 2 and 3 can also represent recommended or implied skills. Converting all
positive scores into explicit mention gold would change the task. They are
deferred retrieval experiments, with no qrels payloads or taxonomy exports
imported here. The cards' ESCO grant statements are recorded as publisher claims,
not a substitute for version-specific original-taxonomy license review.

## Lightweight pretrained encoders for a later NER experiment

No model weights were downloaded, executed, or fine-tuned during this review.
Parameter counts come from HF metadata and are not runtime memory measurements.
These are base encoders requiring a new token-classification head and licensed
span training data; none is an already validated skill extractor.

| Candidate and revision | License evidence actually found | Appropriate experiment and remaining limits |
|---|---|---|
| [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be), **66,985,530 parameters** | **Actual Apache-2.0 `LICENSE`** plus Apache model-card declaration. No separate NOTICE listed. License SHA-256: `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`. | Preferred small **English** base candidate for a controlled token-classifier experiment. Retain Apache terms, attribution and modification notices in any derivative distribution. Card describes BookCorpus/Wikipedia pretraining and inherited biases; underlying corpus rights and redistribution policy still need review. Preserve case-sensitive technical examples in evaluation because the tokenizer is uncased. |
| [distilbert/distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased/tree/6ea81172465e8b0ad3fddeed32b986cdcdcffcf0), **65,812,036 parameters** | Card declares Apache-2.0; **no standalone LICENSE/NOTICE** in the inspected model tree. | Alternative English base if preserving case materially helps validation. Do not assume the neighboring uncased asset's standalone license file is a cased-asset notice. Resolve asset-specific packaging before redistribution. |
| [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2/tree/e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae), **29,376,502 floating-point parameters** | Card declares MIT; **no standalone LICENSE/NOTICE** in the inspected tree. Card says the encoder focuses only on Russian and may be fine-tuned for downstream tasks. | Future small **Russian** candidate after collecting independent Russian human span gold and clarifying retained notices/pretraining provenance. Sentence-similarity initialization is not Russian skill extraction evidence. Do not train BIO from title tags or generated skill lists. |

The existing [extractor research](../skill_extractor_research.md) already records
the missing model-specific grants for exported JobBERT skill/knowledge models.
Pretrained skill extractors may also have been fitted on SkillSpan, which would
make contamination difficult to exclude. A licensed generic base encoder with a
new head fitted only on audited training data gives a clearer comparison.

## Search scope and remaining uncertainty

The search covered public Hugging Face dataset listings for skill extraction,
SkillSpan, TechWolf, Skill-XL, Synthetic ESCO, Russian skills, vacancies,
headhunter, Russian job retrieval, soft skills and JobBot; GitHub repository
search and the linked academic papers/cards were also inspected. SkillNER's
EMSI/Lightcast dictionary and Nesta's library/model were already researched in
the repository; their code/model licenses do not automatically authorize every
underlying dictionary, job text or annotation asset. No new independently
licensed Russian human skill-span corpus was established in this search.

The evidence snapshot is [additional_sources_evidence.json](additional_sources_evidence.json).
Hugging Face and GitHub metadata were readable through approved public network
access. ACL/arXiv PDF requests and Kaggle license endpoints returned proxy HTTP
403; dataset-viewer requests for the two explicitly noted candidates failed.
Those failures limit verification and do not establish that a license or dataset
does not exist. Publisher clarification or a separate manual legal review is
required for missing/ambiguous grants, original job-platform rights, and release
of derivatives. No author was contacted and no authorization was inferred from
public availability alone.
