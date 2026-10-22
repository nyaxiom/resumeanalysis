"""Adapters for deliberately different schemas; labels retain their source provenance."""

import json
import random
from pathlib import Path

from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.normalization.skill_normalizer import SkillNormalizer
from resumeanalysis.schemas import Profile, Vacancy

from .dataset import SPLITS, PairRecord, fingerprint, grouped_split, save_splits
from .download import sha256_file


def string_list(value) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        raise ValueError("Expected a list of strings, or a JSON-encoded list")
    return value


def partition(text: str, seed: int) -> str:
    x = int(fingerprint(f"{seed}:{text}")[:12], 16) / 16**12
    return "train" if x < 0.7 else "validation" if x < 0.85 else "test"


def prepare_candidates(
    raw: str | Path,
    destination: str | Path,
    max_jobs: int = 250,
    pairs_per_job: int = 2,
    seed: int = 42,
) -> dict:
    """Partition entities BEFORE pairing, discard cross-partition positive edges.

    Published relevant IDs are positives. Unlisted resumes are NOT known negatives.
    Diagnostic negatives additionally require <60% required-skill coverage, explicitly
    using the source's synthetic rule. These are weak labels, never hiring decisions.
    """
    import pyarrow.parquet as pq

    if max_jobs < 6 or pairs_per_job < 1:
        raise ValueError("Use at least six jobs and one pair per job")
    raw = Path(raw)
    manifest = json.loads((raw / "manifest.json").read_text())
    if manifest["license"] != "mit":
        raise ValueError("Candidate dataset license not reviewed")
    for relative, digest in manifest["sha256"].items():
        if sha256_file(raw / relative) != digest:
            raise ValueError(f"Raw source changed: {relative}")
    tables = {
        name: pq.read_table(raw / name / "train-00000-of-00001.parquet").to_pylist()
        for name in ["resumes", "jobs", "matches"]
    }
    expected = {
        "resumes": {"resume_id", "summary", "skills", "years_experience", "experience_bullets"},
        "jobs": {"job_id", "description", "must_have_skills", "nice_to_have_skills", "job_title"},
        "matches": {"job_id", "relevant_resume_ids"},
    }
    for name, rows in tables.items():
        if not rows or not expected[name].issubset(rows[0]):
            raise ValueError(f"Unexpected {name} schema; inspect the downloaded parquet")
        id_column = "resume_id" if name == "resumes" else "job_id"
        if len({r[id_column] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {name} IDs")
    normalizer = SkillNormalizer()
    extractor = SkillExtractor(normalizer)
    rng = random.Random(seed)
    profiles, candidates_by_split, skills = {}, {s: [] for s in SPLITS}, {}
    for row in tables["resumes"]:
        rid = row["resume_id"]
        if rid in profiles:
            raise ValueError("Duplicate candidate ID")
        profiles[rid] = Profile(
            resume_text=row["summary"],
            skills=string_list(row["skills"]),
            projects=string_list(row["experience_bullets"]),
            education=[row["education"]] if row.get("education") else [],
            experience_years=row["years_experience"],
            seniority={"Junior": "junior", "Mid": "middle", "Senior": "senior"}.get(
                row.get("seniority")
            ),
        )
        # Group identical summaries even when IDs or supplemental fields differ.
        split = partition(row["summary"], seed)
        candidates_by_split[split].append(rid)
        evidence = extractor.extract(
            "\n".join(
                [profiles[rid].resume_text, *profiles[rid].projects, *profiles[rid].education]
            )
        )
        skills[rid] = (
            normalizer.normalize_many(profiles[rid].skills) | evidence.confirmed
        ) - evidence.denied
    matches = {r["job_id"]: set(string_list(r["relevant_resume_ids"])) for r in tables["matches"]}
    splits = {s: [] for s in SPLITS}
    jobs = list(tables["jobs"])
    rng.shuffle(jobs)
    quotas = dict(
        zip(SPLITS, [round(max_jobs * 0.7), round(max_jobs * 0.15), round(max_jobs * 0.15)])
    )
    quotas["train"] += max_jobs - sum(quotas.values())
    used = {s: 0 for s in SPLITS}
    dropped_edges = 0
    for row in jobs:
        jid = row["job_id"]
        split = partition(row["description"], seed)
        if used[split] >= quotas[split]:
            continue
        if jid not in matches or not matches[jid].issubset(profiles):
            raise ValueError("Match table has missing foreign keys")
        positive_pool = sorted(matches[jid] & set(candidates_by_split[split]))
        dropped_edges += len(matches[jid]) - len(positive_pool)
        if not positive_pool:
            continue
        vacancy = Vacancy(
            title=row["job_title"],
            description=row["description"],
            required_skills=string_list(row["must_have_skills"]),
            optional_skills=string_list(row["nice_to_have_skills"]),
            seniority={"Junior": "junior", "Mid": "middle", "Senior": "senior"}.get(
                row.get("seniority")
            ),
        )
        required = normalizer.normalize_many(vacancy.required_skills)
        if not required:
            continue
        negatives = []
        for rid in candidates_by_split[split]:
            coverage = len(required & skills[rid]) / len(required)
            if rid not in matches[jid] and coverage < 0.6:
                negatives.append((coverage, rng.random(), rid))
        # Most overlapping sub-threshold profiles are hard negatives.
        negatives.sort(reverse=True)
        count = min(pairs_per_job, len(positive_pool), len(negatives))
        if not count:
            continue
        rng.shuffle(positive_pool)
        for rid in positive_pool[:count]:
            splits[split].append(
                PairRecord(
                    candidate_id=rid,
                    vacancy_id=jid,
                    profile=profiles[rid],
                    vacancy=vacancy,
                    label=1,
                    provenance="candidate-matching-synthetic:published-positive-list",
                )
            )
        for coverage, _, rid in negatives[:count]:
            splits[split].append(
                PairRecord(
                    candidate_id=rid,
                    vacancy_id=jid,
                    profile=profiles[rid],
                    vacancy=vacancy,
                    label=0,
                    hard_negative=coverage > 0,
                    provenance="candidate-matching-synthetic:weak-negative-below-60pct-with-text-evidence",
                )
            )
        used[split] += 1
    for records in splits.values():
        rng.shuffle(records)
    return save_splits(
        splits,
        destination,
        {
            "source": manifest,
            "seed": seed,
            "max_jobs": max_jobs,
            "pairs_per_job": pairs_per_job,
            "split_strategy": "independent normalized entity-text hash partitions before pair sampling",
            "discarded_cross_partition_positive_edges_for_selected_jobs": dropped_edges,
            "label_warning": "Synthetic positives and rule-derived negatives, not employer decisions.",
            "negative_policy": "Unlisted and below 60% required coverage after structured AND text evidence; highest overlap first",
            "preparation_version": 2,
        },
    )


def prepare_med2425(
    path: str | Path,
    destination: str | Path,
    *,
    license_evidence: str,
    potential_fit: str = "exclude",
    seed: int = 42,
) -> dict:
    """Disabled until the user obtains a suitable license for ALL merged sources.

    The upstream card currently declares none. Evidence is a local reviewed manifest,
    not a license override inferred from availability. Ambiguous Potential Fit is explicit.
    """
    import pyarrow.parquet as pq

    evidence = json.loads(Path(license_evidence).read_text())
    if not evidence.get("license") or not evidence.get("source_license_urls"):
        raise ValueError("Missing reviewed license and upstream source evidence")
    if potential_fit not in {"exclude", "positive", "negative"}:
        raise ValueError("Choose an explicit Potential Fit policy")
    mapping = {"Good Fit": 1, "No Fit": 0}
    if potential_fit != "exclude":
        mapping["Potential Fit"] = int(potential_fit == "positive")
    records, excluded = [], 0
    for row in pq.read_table(path).to_pylist():
        if not {"resume", "jd", "label", "source"}.issubset(row):
            raise ValueError("Unexpected med2425 schema")
        if row["label"] == "Potential Fit" and potential_fit == "exclude":
            excluded += 1
            continue
        if row["label"] not in mapping:
            raise ValueError("Unrecognized source label")
        records.append(
            PairRecord(
                candidate_id=fingerprint(row["resume"]),
                vacancy_id=fingerprint(row["jd"]),
                profile=Profile(resume_text=row["resume"]),
                vacancy=Vacancy(description=row["jd"]),
                label=mapping[row["label"]],
                provenance=f"med2425:{row['source']}:Qwen-label",
            )
        )
    return save_splits(
        grouped_split(records, seed),
        destination,
        {
            "source": "med2425/resume-job-fit-merged-v1",
            "raw_sha256": sha256_file(path),
            "license_evidence": evidence,
            "potential_fit_policy": potential_fit,
            "excluded": excluded,
            "label_warning": "Qwen-derived labels; not independent employer ground truth",
            "seed": seed,
        },
    )
