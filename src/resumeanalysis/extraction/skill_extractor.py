"""Deterministic RU/EN fallback with spans, section context and explicit negation.

No title-to-competence inference and no model downloads during extraction.
"""

import re
from dataclasses import dataclass, field

from resumeanalysis.normalization.skill_normalizer import SkillNormalizer

OPTIONAL = re.compile(
    r"nice.to.have|optional|preferred|a plus|желательн|будет плюсом|приветствуется", re.I
)
REQUIRED = re.compile(
    r"required|requirements|must.have|mandatory|обязател|требования|необходим", re.I
)
NEGATION = re.compile(
    r"(?:\bno (?:experience|knowledge|skills?)(?:\s+(?:with|in|of))?|"
    r"\b(?:do not|don't|cannot|can't) (?:know|use)|\bwithout (?:experience|knowledge)(?:\s+(?:in|of|with))?|"
    r"не (?:знаю|владею|умею)|нет (?:опыта|знаний)(?:\s+(?:с|в))?|без опыта(?:\s+с)?)\s*[:—-]?\s*",
    re.I,
)
FUTURE = re.compile(
    r"want to learn|plan to learn|learning goals?|хочу изучить|планирую изучить", re.I
)
NOT_REQUIRED = re.compile(r"not required|not necessary|не требуется|не обязател", re.I)
SENIORITY = {
    "intern": r"\bintern\b|стаж[её]р",
    "junior": r"\bjunior\b|младший",
    "middle": r"\bmid(?:dle)?\b",
    "senior": r"\bsenior\b|старший",
    "lead": r"\blead\b|ведущий",
}


@dataclass
class SkillMention:
    skill: str
    text: str
    start: int
    end: int
    status: str
    requirement: str


@dataclass
class Extraction:
    confirmed: set[str] = field(default_factory=set)
    denied: set[str] = field(default_factory=set)
    uncertain: set[str] = field(default_factory=set)
    required: set[str] = field(default_factory=set)
    optional: set[str] = field(default_factory=set)
    mentions: list[SkillMention] = field(default_factory=list)
    experience_years: float | None = None
    seniority: str | None = None


class SkillExtractor:
    def __init__(self, normalizer: SkillNormalizer | None = None):
        self.normalizer = normalizer or SkillNormalizer()
        aliases = sorted(self.normalizer.lookup, key=len, reverse=True)
        self.pattern = re.compile(
            r"(?<![\w+#])(?:"
            + "|".join(re.escape(s) for s in aliases + ["torch"])
            + r")(?![\w+#])",
            re.I,
        )

    def extract(self, text: str, *, vacancy: bool = False) -> Extraction:
        result = Extraction()
        section = "mentioned"
        # Sentence boundaries avoid splitting node.js and scikit-learn.
        for segment in re.finditer(r"[^\n;.!?]+(?:\.(?!\s|$)[^\n;.!?]+)*", text):
            clause = segment.group()
            local = (
                "optional"
                if OPTIONAL.search(clause)
                else ("required" if REQUIRED.search(clause) else section)
            )
            if clause.strip().endswith(":") or not self.pattern.search(clause):
                if OPTIONAL.search(clause):
                    section = "optional"
                elif REQUIRED.search(clause):
                    section = "required"
                elif re.search(r"responsibilities|benefits|обязанности|условия", clause, re.I):
                    section = "mentioned"
            for match in self.pattern.finditer(clause):
                skill = self.normalizer.normalize(match.group(), clause)
                prefix = clause[: match.start()]
                # A contrast ends a negative list; commas and 'and/or' preserve it.
                prefix = re.split(r"\bbut\b|\bhowever\b|\bно\b|\bзато\b", prefix, flags=re.I)[-1]
                status = (
                    "denied"
                    if NEGATION.search(prefix)
                    else ("unknown" if FUTURE.search(prefix) else "confirmed")
                )
                if vacancy and NOT_REQUIRED.search(clause):
                    requirement = "optional"
                else:
                    requirement = local
                start = segment.start() + match.start()
                result.mentions.append(
                    SkillMention(
                        skill,
                        match.group(),
                        start,
                        segment.start() + match.end(),
                        status,
                        requirement,
                    )
                )
                getattr(
                    result,
                    {"confirmed": "confirmed", "denied": "denied", "unknown": "uncertain"}[status],
                ).add(skill)
                if vacancy and requirement in {"required", "optional"}:
                    getattr(result, requirement).add(skill)
        conflicting = result.confirmed & result.denied
        result.uncertain |= conflicting
        result.confirmed -= conflicting
        result.denied -= conflicting
        # 'not required' and optional sections never turn into mandatory requirements.
        result.optional -= result.required
        years = re.findall(r"(\d+(?:[.,]\d+)?)\s*\+?\s*(?:years?\b|лет\b|года?\b)", text, re.I)
        if years:
            values = [float(s.replace(",", ".")) for s in years]
            result.experience_years = min(values) if vacancy else max(values)
        for level, pattern in SENIORITY.items():
            if re.search(pattern, text, re.I):
                result.seniority = level
        return result
