from dataclasses import dataclass

import numpy as np

from resumeanalysis.embeddings.encoder import Encoder
from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.schemas import Profile, Vacancy

FEATURE_NAMES = [
    "required_match",
    "required_unknown",
    "required_denied",
    "optional_match",
    "optional_unknown",
    "optional_denied",
    "semantic_similarity",
    "related_support",
    "experience_match",
    "experience_known",
    "experience_gap_scaled",
    "seniority_match",
    "seniority_known",
    "required_count_scaled",
    "optional_count_scaled",
    "mentioned_match",
    "requirements_known",
]
LEVELS = {k: n for n, k in enumerate(["intern", "junior", "middle", "senior", "lead"])}
PIPELINE_VERSION = 1


@dataclass
class FeatureDetail:
    vector: np.ndarray
    resume_embedding: np.ndarray
    vacancy_embedding: np.ndarray
    required: set[str]
    optional: set[str]
    matched: set[str]
    unknown: set[str]
    contradicted: set[str]
    experience_gap: float | None


def profile_text(profile: Profile) -> str:
    return "\n".join(
        [
            profile.resume_text,
            "Skills: " + ", ".join(profile.skills),
            *profile.projects,
            *profile.education,
            "Explicitly not known: " + ", ".join(profile.denied_skills),
        ]
    )


def vacancy_text(vacancy: Vacancy) -> str:
    # Job titles are intentionally excluded: compare evidence of competencies.
    return "\n".join(
        [
            vacancy.description,
            "Required: " + ", ".join(vacancy.required_skills),
            "Optional: " + ", ".join(vacancy.optional_skills),
        ]
    )


class FeatureEngineering:
    def __init__(self, encoder: Encoder, extractor: SkillExtractor | None = None):
        self.encoder, self.extractor = encoder, extractor or SkillExtractor()

    def build(self, profile: Profile, vacancy: Vacancy) -> FeatureDetail:
        norm = self.extractor.normalizer
        p = self.extractor.extract(
            "\n".join([profile.resume_text, *profile.projects, *profile.education])
        )
        v = self.extractor.extract(vacancy.description, vacancy=True)
        confirmed = p.confirmed | norm.normalize_many(profile.skills, profile.resume_text)
        denied = p.denied | norm.normalize_many(profile.denied_skills, profile.resume_text)
        conflict = confirmed & denied
        confirmed -= conflict
        denied -= conflict
        required = v.required | norm.normalize_many(vacancy.required_skills, vacancy.description)
        optional = (
            v.optional | norm.normalize_many(vacancy.optional_skills, vacancy.description)
        ) - required
        wanted = required | optional
        matched, contradicted = wanted & confirmed, wanted & denied
        unknown = wanted - matched - contradicted

        def ratio(subset, pool):
            return len(subset & pool) / len(pool) if pool else 0.0

        years = (
            profile.experience_years if profile.experience_years is not None else p.experience_years
        )
        minimum = (
            vacancy.min_experience_years
            if vacancy.min_experience_years is not None
            else v.experience_years
        )
        gap = max(0.0, minimum - years) if years is not None and minimum is not None else None
        pl, vl = profile.seniority or p.seniority, vacancy.seniority or v.seniority
        r = self.encoder.encode(profile_text(profile), "query")
        j = self.encoder.encode(vacancy_text(vacancy), "passage")
        semantic = float(np.dot(r, j))
        related = sum(norm.taxonomy.related_support(confirmed, s) for s in required - confirmed)
        values = [
            ratio(matched, required),
            ratio(unknown, required),
            ratio(contradicted, required),
            ratio(matched, optional),
            ratio(unknown, optional),
            ratio(contradicted, optional),
            semantic,
            related / max(1, len(required)),
            float(gap == 0) if gap is not None else 0,
            float(gap is not None),
            min(gap or 0, 20) / 20,
            float(LEVELS[pl] >= LEVELS[vl]) if pl and vl else 0,
            float(bool(pl and vl)),
            min(len(required), 20) / 20,
            min(len(optional), 20) / 20,
            ratio(confirmed, v.confirmed - wanted),
            float(bool(required)),
        ]
        vector = np.asarray(values, dtype=np.float32)
        if (
            not np.isfinite(vector).all()
            or r.shape != (self.encoder.dimension,)
            or j.shape != r.shape
        ):
            raise ValueError("Invalid feature/embedding output")
        return FeatureDetail(vector, r, j, required, optional, matched, unknown, contradicted, gap)
