"""Evidence-constrained skill reporting with a prompted semantic relevance decision."""

import numpy as np
from pydantic import ValidationError

from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.features.feature_builder import FeatureDetail
from resumeanalysis.inference.backends import MatchingProfile
from resumeanalysis.jsonio import loads
from resumeanalysis.postprocessing.rules import Postprocessor
from resumeanalysis.qwen.config import QwenConfig
from resumeanalysis.qwen.prompts import Decision, build_messages
from resumeanalysis.qwen.runtime import Generator, TransformersGenerator
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


class InvalidModelOutput(ValueError):
    """No validated decision was produced; raw personal input is never in this error."""


def skill_evidence(profile: Profile, vacancy: Vacancy) -> FeatureDetail:
    extractor = SkillExtractor()
    norm = extractor.normalizer
    p = extractor.extract("\n".join([profile.resume_text, *profile.projects, *profile.education]))
    v = extractor.extract(vacancy.description, vacancy=True)
    confirmed = p.confirmed | norm.normalize_many(profile.skills)
    denied = p.denied | norm.normalize_many(profile.denied_skills)
    conflict = confirmed & denied
    confirmed -= conflict
    denied -= conflict
    required = v.required | norm.normalize_many(vacancy.required_skills)
    optional = (v.optional | norm.normalize_many(vacancy.optional_skills)) - required
    wanted = required | optional
    matched, contradicted = wanted & confirmed, wanted & denied
    # Duration is reported only from explicit structured fields, not career inference.
    years, minimum = profile.experience_years, vacancy.min_experience_years
    gap = max(0.0, minimum - years) if years is not None and minimum is not None else None
    empty = np.empty(0, dtype=np.float32)
    return FeatureDetail(
        empty,
        empty,
        empty,
        required,
        optional,
        matched,
        wanted - matched - contradicted,
        contradicted,
        gap,
    )


def resolve_pointer(document, pointer: str):
    if not pointer.startswith(("/profile/", "/vacancy/")):
        raise ValueError("Evidence must reference an input field")
    value = document
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class QwenPredictor:
    def __init__(self, config: QwenConfig | None = None, generator: Generator | None = None):
        self.config = config or QwenConfig()
        self.generator = generator if generator is not None else TransformersGenerator(self.config)

    def predict(self, profile: Profile | dict, vacancy: Vacancy | dict) -> VacancyAnalysis:
        p = MatchingProfile.model_validate(
            profile.model_dump() if isinstance(profile, Profile) else profile, strict=True
        )
        v = (
            vacancy
            if isinstance(vacancy, Vacancy)
            else Vacancy.model_validate(vacancy, strict=True)
        )
        detail = skill_evidence(p, v)
        catalog = {
            "matched_skills": sorted(detail.matched),
            "unknown_skills": sorted(detail.unknown),
            "contradicted_skills": sorted(detail.contradicted),
        }
        messages = build_messages(p, v, catalog, self.config.threshold)
        document = {"profile": p.model_dump(), "vacancy": v.model_dump()}
        decision = None
        for attempt in range(self.config.retries + 1):
            raw = self.generator.generate(messages, Decision.model_json_schema())
            try:
                candidate = Decision.model_validate(loads(raw), strict=True)
                for key, values in catalog.items():
                    if getattr(candidate, key) != values:
                        raise ValueError("Unsupported skill claims")
                if candidate.relevant != (candidate.relevance >= self.config.threshold):
                    raise ValueError("Inconsistent decision and score")
                has_evidence = bool(
                    p.resume_text.strip()
                    or p.skills
                    or p.denied_skills
                    or p.projects
                    or p.education
                    or p.experience_years is not None
                    or p.seniority
                )
                if not has_evidence and (candidate.relevant or candidate.relevance != 0):
                    raise ValueError("Candidate evidence is empty")
                for citation in candidate.evidence:
                    source = resolve_pointer(document, citation.path)
                    if not isinstance(source, str) or citation.quote not in source:
                        raise ValueError("Unsupported evidence quote")
                decision = candidate
                break
            except (ValueError, ValidationError, KeyError, IndexError, TypeError):
                if attempt == self.config.retries:
                    raise InvalidModelOutput(
                        "Qwen output failed JSON, schema or evidence validation"
                    ) from None
                # Never echo invalid output into another prompt or error message.
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": "The previous response failed validation. Return the required JSON schema, "
                        "copying the catalog exactly, with literal evidence and consistent score.",
                    },
                ]
        assert decision is not None
        result = Postprocessor(self.config.threshold).process(decision.relevance, detail, p, v)
        result.explanation = (
            f"Qwen relevance assessment: {decision.relevance:g} (uncalibrated score). "
            + result.explanation
        )
        if decision.evidence:
            result.explanation += (
                " Supplied evidence (quoted input, not verified claims): "
                + "; ".join(f"{c.path}: {c.quote!r}" for c in decision.evidence)
            )
        return result
