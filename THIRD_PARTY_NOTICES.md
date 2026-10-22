# Third-party notices

ResumeAnalysis source is MIT, copyright 2026 ResumeAnalysis contributors.
Reference implementations were inspected but their source code was not copied or vendored:

* ResuMatch — JananiV07/resumatch, MIT, copyright (c) 2026 vdenterprises.
* Nesta OJD DAPS Skills — nestauk/ojd_daps_skills, MIT, copyright (c) 2024 Nesta.
* SkillAlign — Y4SSERk/SkillAlign, MIT, copyright (c) 2025 YASSER.

Exact commits and files consulted are recorded in [docs/research.md](docs/research.md).
Runtime dependencies retain their own licenses; this project's MIT license does not
relicense models, datasets, or taxonomy data.

## ESCO identifiers

This service uses the ESCO classification of the European Commission.
ESCO content: European Union, Creative Commons Attribution 4.0 International
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), as documented in
[SkillAlign NOTICE](https://github.com/Y4SSERk/SkillAlign/blob/cf0681df995838b14823825f0a44ec18c3903108/NOTICE).
Official source: <https://esco.ec.europa.eu/en/use-esco/download>.

Changes: four identifiers from the ESCO v1.1.1 formatted table distributed by Nesta
were associated with the manually curated Python, SQL, PostgreSQL and MySQL names.
Local aliases and weighted skill relationships are independently authored, not official
ESCO statements. No endorsement by the European Union or European Commission is implied.

## Downloaded models and datasets (not included in Git)

* `intfloat/multilingual-e5-small`: MIT according to the pinned Hugging Face model card.
  The download retains the card and verifies the safetensors SHA256 against upstream metadata.
* `michaelozon/candidate-matching-synthetic`: MIT according to the pinned dataset card.
  The download retains the card and hashes all files. Profiles/jobs are Qwen-generated,
  positives heuristic; our negative examples are explicitly weak synthetic labels.
* SkillSpan: MIT, copyright (c) 2023 Mike Zhang. Data remains in the external checkout,
  no dataset records are redistributed here. Required citation:

  Zhang, Mike; Jensen, Kristian Nørgaard; Sonniks, Sif; Plank, Barbara (2022).
  **SkillSpan: Hard and Soft Skill Extraction from English Job Postings.**
  NAACL-HLT, pp. 4962–4984. <https://aclanthology.org/2022.naacl-main.366>.

`med2425/resume-job-fit-merged-v1` is not included, downloaded or trained on:
no license was declared in its inspected card. Its adapter requires separate reviewed
license evidence for the merged source datasets.
