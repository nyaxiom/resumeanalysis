"""Conservative requirement policy layered over the existing extraction interface.

Only evidence in the supplied resume and explicit fields can confirm a skill.
An unmentioned requirement needs review; it is never a claim of incapacity.
"""

import re
from dataclasses import dataclass
from typing import Literal

from resumeanalysis.extraction.registry import Extractor
from resumeanalysis.features.feature_builder import (
    LEVELS,
    FeatureDetail,
    profile_text,
    vacancy_text,
)
from resumeanalysis.qwen.predictor import skill_evidence
from resumeanalysis.schemas import Profile, Vacancy

SERIALIZATION_VERSION = "resume-vacancy-evidence-v1"

# These narrow text cues request verification; they never establish a hard
# minimum or overwrite a structured field. Bare dates, years and colleague role
# mentions are intentionally excluded.
_REQUIREMENT = re.compile(
    r"\brequired\b|\brequirements?\b|\bmust(?:\s+have)?\b|\bmandatory\b|"
    r"\bat\s+least\b|\bminimum\b|требован\w*|требуется|необходим\w*|"
    r"обязател\w*|не\s+менее",
    re.I,
)
_OPTIONAL = re.compile(
    r"\boptional\b|\bpreferred\b|\bnice\s+to\s+have\b|\ba\s+plus\b|"
    r"\bnot\s+(?:required|necessary)\b|желатель\w*|будет\s+плюсом|"
    r"не\s+(?:требуется|обязател\w*)",
    re.I,
)
_EXPERIENCE = re.compile(
    r"(?P<en_before>\d+(?:[.,]\d+)?)\s*\+?\s*years?\s*(?:of\s+)?"
    r"(?:(?:professional|relevant|commercial|work)\s+)?experience\b|"
    r"\bexperience\s*(?:of\s+|:)?\s*(?:at\s+least\s+|minimum\s+|from\s+)?"
    r"(?P<en_after>\d+(?:[.,]\d+)?)\s*\+?\s*years?\b|"
    r"опыт(?:а)?\s*(?:работы\s*)?(?:от\s+|не\s+менее\s+|минимум\s+|:\s*)?"
    r"(?P<ru_after>\d+(?:[.,]\d+)?)\s*\+?\s*(?:лет|года?)\b|"
    r"(?P<ru_before>\d+(?:[.,]\d+)?)\s*\+?\s*(?:лет|года?)\s+опыта\b",
    re.I,
)
_LEVEL = re.compile(
    r"\bintern\b|\bjunior\b|\bmid(?:dle)?\b|\bsenior\b|\blead\b|"
    r"стаж[её]р|младш\w*|старш\w*|ведущ\w*",
    re.I,
)
_LEVEL_CONTEXT = re.compile(
    r"\bseniority\b|\blevel\b|уровень|квалификаци\w*|"
    r"^\s*(?:(?:required|mandatory|requirements?|требуется|необходим)\s*[:\-]?\s*)?"
    r"(?:intern|junior|mid(?:dle)?|senior|lead|стаж[её]р|младш\w*|старш\w*|ведущ\w*)\b",
    re.I,
)
_OFFICE_ONLY = re.compile(
    r"\b(?:on[- ]?site|in[- ]?office|office)(?:\s+(?:work|role|position))?\s+only\b|"
    r"\bno\s+remote(?:\s+work)?\b|"
    r"\bremote\s+(?:work\s+)?(?:is\s+)?(?:not\s+(?:available|allowed|offered|possible)|unavailable)\b|"
    r"(?:только|исключительно)\s+(?:работа\s+)?в\s+офисе|"
    r"удал[её]нная\s+работа\s+(?:не\s+предусмотрена|невозможна|не\s+допускается)",
    re.I,
)
_REMOTE_OFFERED = re.compile(
    r"\b(?:fully|100%)\s+remote\b|\bremote[- ]only\b|"
    r"\bremote\s+work\s+is\s+available\b|полностью\s+удал[её]нн\w*|"
    r"удал[её]нная\s+работа\s+(?:доступна|предусмотрена)",
    re.I,
)


def _text_requirement_cues(text):
    years, levels = set(), set()
    required_section = False
    for clause in re.split(r"(?<=[.!?;])\s+|\n+", text):
        # Explicit section headings persist across bullet lines, but optional,
        # responsibilities and company-description headings end that context.
        if clause.strip().endswith(":"):
            required_section = bool(_REQUIREMENT.search(clause)) and not _OPTIONAL.search(clause)
        if _OPTIONAL.search(clause):
            continue
        required = required_section or bool(_REQUIREMENT.search(clause))
        if required:
            for match in _EXPERIENCE.finditer(clause):
                value = next(value for value in match.groupdict().values() if value is not None)
                years.add(float(value.replace(",", ".")))
            if _LEVEL_CONTEXT.search(clause):
                for match in _LEVEL.finditer(clause):
                    word = match.group().casefold()
                    level = next(
                        (
                            canonical
                            for prefix, canonical in (
                                ("стаж", "intern"),
                                ("младш", "junior"),
                                ("старш", "senior"),
                                ("ведущ", "lead"),
                                ("mid", "middle"),
                            )
                            if word.startswith(prefix)
                        ),
                        word,
                    )
                    levels.add(level)
    return {"experience_years": sorted(years), "seniority": sorted(levels)}


def _text_review(vacancy, cues):
    reasons = []
    years, levels = cues["experience_years"], cues["seniority"]
    if years and (
        vacancy.min_experience_years is None
        or any(value != vacancy.min_experience_years for value in years)
    ):
        reasons.append(
            "Explicit experience requirements in the vacancy text need verification against "
            "the structured minimum; prose values do not establish a verified hard minimum."
        )
    if levels and (
        vacancy.seniority is None or any(level != vacancy.seniority for level in levels)
    ):
        reasons.append(
            "Explicit seniority requirements in the vacancy text need verification against "
            "the structured level."
        )

    def asserted(pattern):
        # Do not reinterpret "not office only" or "not fully remote" as
        # affirmative arrangements. The negative inside "no remote work" is
        # part of the office-only cue itself and is still recognized.
        return any(
            not re.search(
                r"(?:\bnot|\bno|\bwithout|\bне)\s+(?:an?\s+)?$",
                vacancy.description[max(0, match.start() - 24) : match.start()],
                re.I,
            )
            for match in pattern.finditer(vacancy.description)
        )

    office_only = asserted(_OFFICE_ONLY)
    remote_offered = asserted(_REMOTE_OFFERED)
    if (
        (office_only and vacancy.remote is True)
        or (remote_offered and vacancy.remote is False)
        or (office_only and remote_offered)
    ):
        reasons.append(
            "Remote-work statements conflict with the structured availability or with each other; "
            "verify the vacancy's work arrangement."
        )
    cues["office_only"] = office_only
    cues["remote_offered"] = remote_offered
    return reasons


def cross_encoder_texts(profile: Profile, vacancy: Vacancy) -> tuple[str, str]:
    """Serialize both sides consistently for standalone and pipeline evaluation.

    Job titles and preferred roles do not establish competencies. Explicit
    numeric/level fields supplement the existing content serializers. Preferences
    are enforced independently instead of being left to semantic similarity.
    """
    resume, job = [profile_text(profile)], [vacancy_text(vacancy)]
    if profile.experience_years is not None:
        resume.append(f"Stated experience: {profile.experience_years:g} years")
    if profile.seniority is not None:
        resume.append(f"Stated seniority: {profile.seniority}")
    if vacancy.min_experience_years is not None:
        job.append(f"Minimum experience: {vacancy.min_experience_years:g} years")
    if vacancy.seniority is not None:
        job.append(f"Required seniority: {vacancy.seniority}")
    return "\n".join(resume), "\n".join(job)


def preference_conflicts(profile: Profile, vacancy: Vacancy) -> list[str]:
    """Cheap known hard-preference checks, safe before retrieval preselection."""
    result = []
    if profile.remote_only and vacancy.remote is False:
        result.append("Remote work is required, but the vacancy does not offer it.")
    if profile.allowed_locations and vacancy.location is not None:
        if vacancy.location.casefold() not in {
            location.casefold() for location in profile.allowed_locations
        }:
            result.append("The location does not match the allowed locations.")
    return result


@dataclass
class EvidenceSummary:
    detail: FeatureDetail
    blocked_reasons: list[str]
    review_reasons: list[str]
    metadata: dict

    @property
    def status(self) -> Literal["eligible", "needs_review", "blocked"]:
        if self.blocked_reasons:
            return "blocked"
        return "needs_review" if self.review_reasons else "eligible"


class _RecordingExtractor:
    """Capture the two existing helper calls without running extraction twice."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.normalizer = delegate.normalizer
        self.outputs = []

    def extract(self, text, *, vacancy=False):
        result = self.delegate.extract(text, vacancy=vacancy)
        self.outputs.append(result)
        return result


def _nonempty(values):
    return any(value.strip() for value in values)


def has_candidate_evidence(profile: Profile) -> bool:
    """Cheap positive-content check shared by pair and batch inference."""
    return bool(
        profile.resume_text.strip()
        or _nonempty(profile.skills)
        or _nonempty(profile.projects)
        or _nonempty(profile.education)
        or profile.experience_years is not None
        or profile.seniority is not None
    )


def build_evidence(profile: Profile, vacancy: Vacancy, extractor: Extractor) -> EvidenceSummary:
    """Reuse extraction/normalization and add conservative automatic-action gates.

    Conflict recovery uses retained mentions: legacy and V2 deliberately remove
    contradictory skills from their aggregate confirmed/denied sets. Explicit
    fields must not silently override those contradictory text mentions.
    """
    recorder = _RecordingExtractor(extractor)
    detail = skill_evidence(profile, vacancy, recorder)
    resume, job = recorder.outputs
    norm = extractor.normalizer
    positive = resume.confirmed | norm.normalize_many(profile.skills)
    negative = resume.denied | norm.normalize_many(profile.denied_skills)
    positive |= {m.skill for m in resume.mentions if m.status == "confirmed"}
    negative |= {m.skill for m in resume.mentions if m.status == "denied"}
    conflicted = positive & negative
    confirmed, denied = positive - conflicted, negative - conflicted
    wanted = detail.required | detail.optional
    detail.matched = wanted & confirmed
    detail.contradicted = wanted & denied
    detail.unknown = wanted - detail.matched - detail.contradicted

    text_cues = _text_requirement_cues(vacancy.description)
    blocked, review = preference_conflicts(profile, vacancy), _text_review(vacancy, text_cues)
    if profile.remote_only and vacancy.remote is None:
        review.append("Remote work has not been confirmed.")
    if profile.allowed_locations and vacancy.location is None:
        review.append("The vacancy location is not specified.")
    required_denied = detail.required & denied
    required_conflict = detail.required & conflicted
    required_unknown = detail.required - confirmed - denied - conflicted
    if required_denied:
        blocked.append(
            "Mandatory skills are explicitly denied: " + ", ".join(sorted(required_denied)) + "."
        )
    if required_conflict:
        blocked.append(
            "Conflicting positive and negative evidence requires resolution for mandatory skills: "
            + ", ".join(sorted(required_conflict))
            + "."
        )
    if required_unknown:
        review.append(
            "Mandatory skill evidence is unknown, not proof of absence: "
            + ", ".join(sorted(required_unknown))
            + "."
        )
    for group in detail.alternative_groups:
        options = set(group["skills"])
        matched = options & confirmed
        group["matched_skills"] = sorted(matched)
        if matched:
            group["status"] = "matched"
        elif options <= denied:
            group["status"] = "contradicted"
        elif options <= denied | conflicted:
            group["status"] = "conflicted"
        else:
            group["status"] = "unknown"
        if group["requirement"] in {"required", "optional"}:
            detail.matched |= matched
        if group["requirement"] != "required" or matched:
            continue
        description = " or ".join(group["skills"])
        if options <= denied | conflicted:
            blocked.append(
                "Every option of a mandatory alternative is denied or has conflicting evidence: "
                + description
                + "."
            )
        else:
            review.append(
                "At least one mandatory alternative needs confirmation: " + description + "."
            )

    years, minimum = profile.experience_years, vacancy.min_experience_years
    if minimum is not None:
        if years is None:
            review.append("Candidate experience duration is unknown for the stated minimum.")
        elif years < minimum:
            blocked.append("Stated experience duration is below the mandatory minimum.")
    level, required_level = profile.seniority, vacancy.seniority
    if required_level is not None:
        if level is None:
            review.append("Candidate seniority is unknown for the stated requirement.")
        elif LEVELS[level] < LEVELS[required_level]:
            blocked.append("Stated seniority is below the mandatory level.")

    has_resume = has_candidate_evidence(profile)
    has_vacancy = bool(
        vacancy.description.strip()
        or _nonempty(vacancy.required_skills)
        or _nonempty(vacancy.optional_skills)
        or minimum is not None
        or required_level is not None
    )
    if not has_resume:
        review.append("Positive candidate evidence is empty; a semantic decision is unavailable.")
    if not has_vacancy:
        review.append("Vacancy competency content is empty; a job title alone is insufficient.")

    metadata = {
        "usable_input": has_resume and has_vacancy,
        "required_skills": sorted(detail.required),
        "optional_skills": sorted(detail.optional),
        "alternative_groups": detail.alternative_groups,
        "confirmed_skills": sorted(confirmed),
        "denied_skills": sorted(denied),
        "conflicted_skills": sorted(conflicted),
        "experience": {
            "candidate_years": years,
            "minimum_years": minimum,
            "source": "explicit_structured_fields_only",
            "unverified_text_candidate_years": resume.experience_years,
            "unverified_text_vacancy_years": job.experience_years,
        },
        "seniority": {
            "candidate": level,
            "required": required_level,
            "source": "explicit_structured_fields_only",
            "unverified_text_candidate": resume.seniority,
            "unverified_text_vacancy": job.seniority,
        },
        "preference_source": "explicit_structured_fields_only",
        "unverified_text_requirements": text_cues,
    }
    return EvidenceSummary(detail, blocked, review, metadata)
