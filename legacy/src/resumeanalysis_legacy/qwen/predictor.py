"""Evidence-constrained skill reporting with a prompted semantic relevance decision."""

from pydantic import ValidationError

from resumeanalysis.cover_letters.qwen_runtime import Generator, TransformersGenerator
from resumeanalysis.extraction.registry import Extractor
from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.jsonio import loads
from resumeanalysis.matching.evidence import skill_evidence
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis
from resumeanalysis_legacy.inference.backends import MatchingProfile
from resumeanalysis_legacy.postprocessing.rules import Postprocessor
from resumeanalysis_legacy.qwen.config import QwenConfig
from resumeanalysis_legacy.qwen.prompts import Decision, build_messages


class InvalidModelOutput(ValueError):
    """No validated decision was produced; raw personal input is never in this error."""


def resolve_pointer(document, pointer: str):
    if not pointer.startswith(("/profile/", "/vacancy/")):
        raise ValueError("Evidence must reference an input field")
    value = document
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class QwenPredictor:
    def __init__(
        self,
        config: QwenConfig | None = None,
        generator: Generator | None = None,
        extractor: Extractor | None = None,
    ):
        self.config = config or QwenConfig()
        self.generator = generator if generator is not None else TransformersGenerator(self.config)
        self.extractor = extractor or SkillExtractor()

    def predict(self, profile: Profile | dict, vacancy: Vacancy | dict) -> VacancyAnalysis:
        p = MatchingProfile.model_validate(
            profile.model_dump() if isinstance(profile, Profile) else profile, strict=True
        )
        v = (
            vacancy
            if isinstance(vacancy, Vacancy)
            else Vacancy.model_validate(vacancy, strict=True)
        )
        detail = skill_evidence(p, v, self.extractor)
        catalog = {
            "matched_skills": sorted(detail.matched),
            "unknown_skills": sorted(detail.unknown),
            "contradicted_skills": sorted(detail.contradicted),
        }
        messages = build_messages(p, v, catalog, self.config.threshold)
        if detail.alternative_groups:
            import json

            messages[1]["content"] += (
                "\nRequirement metadata (do not flatten alternatives): "
                + json.dumps(detail.alternative_groups, ensure_ascii=False)
            )
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
