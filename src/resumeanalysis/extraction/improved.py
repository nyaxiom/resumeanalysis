"""Original local phrase matcher plus conservative EN/RU requirement classification.

No external vocabulary/model is loaded implicitly. Alternative requirements stay
grouped and never enter the independent mandatory-skills set.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from resumeanalysis.extraction.compatibility import identity_source_hashes
from resumeanalysis.extraction.skill_extractor import Extraction, SkillExtractor, SkillMention
from resumeanalysis.extraction.vocabulary import make_normalizer, phrase_aliases
from resumeanalysis.normalization.skill_normalizer import SkillNormalizer, clean

REQUIRED = re.compile(
    r"\b(?:requir(?:ed|es?|ements?)|essential(?:\s+skills)?|must[ -]have|mandatory|qualifications?)\b|обязател\w*|требовани\w*|необходим\w*|нужн\w*",
    re.I,
)
OPTIONAL = re.compile(
    r"\b(?:nice[ -]to[ -]have|optional|preferred(?:\s+qualifications)?|desirable|a plus|an advantage|bonus)\b|желател\w*|будет\s+(?:плюсом|преимуществом)|приветствуется|предпочтитель\w*",
    re.I,
)
RESET = re.compile(
    r"^(?:responsibilities|duties|benefits|about(?:\s+us)?|what we offer|company|обязанности|условия|о компании|мы предлагаем)\b",
    re.I,
)
NOT_REQUIRED = re.compile(
    r"\b(?:not\s+(?:required|necessary|mandatory|essential)|no\b.{0,100}?\b(?:experience|knowledge).{0,40}?required|without\s+(?:prior\s+)?(?:experience|knowledge).{0,60}?welcome)\b|не\s+(?:требуется|обязател\w*|нужн\w*)|без\s+опыта.{0,60}?(?:можно|допускается)",
    re.I,
)
NEGATIVE = re.compile(
    r"\b(?:no\s+(?:experience|knowledge|skills?)(?:\s+(?:with|in|of))?|without\s+(?:experience|knowledge)(?:\s+(?:in|with|of))?|(?:do not|don't|cannot|can't)\s+(?:know|use))\b|(?:не\s+(?:знаю|владею|умею)|нет\s+(?:опыта|знаний)(?:\s+(?:с|в))?|без\s+опыта(?:\s+с)?)",
    re.I,
)
UNCERTAIN = re.compile(
    r"want to learn|plan to learn|learning goals?|хочу изучить|планирую изучить", re.I
)
CONTRAST = re.compile(r"\b(?:but|however|whereas|но|зато|однако)\b", re.I)
ALTERNATIVE = re.compile(r"^\s*,?\s*(?:or|and/or|или|либо)\s*$", re.I)
COMMA = re.compile(r"^\s*,\s*$")
NODE_CONTEXT = re.compile(
    r"javascript|typescript|back[ -]?end|npm|express|node\.?js|сервер|бэкенд", re.I
)
LANGUAGE_CONTEXT = re.compile(
    r"programming|languages?|development|golang|statistics|data|c\+\+|язык|программ|разработ|статист|данн",
    re.I,
)


@dataclass
class EvidenceMention(SkillMention):
    provenance: str = "curated-v2"


@dataclass
class AlternativeGroup:
    skills: tuple[str, ...]
    requirement: str
    start: int
    end: int
    text: str


@dataclass
class EnhancedExtraction(Extraction):
    alternative_groups: list[AlternativeGroup] = field(default_factory=list)
    negated_requirements: set[str] = field(default_factory=set)


def alias_pattern(alias):
    # Permit tokenized punctuation while retaining exact character offsets.
    parts = re.split(r"([.\-])", alias)
    result = "".join(
        r"\s*" + re.escape(p) + r"\s*" if p in {".", "-"} else re.escape(p).replace(r"\ ", r"\s+")
        for p in parts
    )
    if result.startswith(r"\s*"):
        result = result[3:]
    if result.endswith(r"\s*"):
        result = result[:-3]
    return result


def scoped_boundaries(content):
    points = {0, len(content), *[m.end() for m in CONTRAST.finditer(content)]}
    # Split explicitly labelled mixed clauses while preserving unlabelled skill lists.
    for match in re.finditer(r",|;|\b(?:and|и)\b", content, re.I):
        before, after = content[: match.start()], content[match.end() :]
        if (REQUIRED.search(before) or OPTIONAL.search(before)) and (
            REQUIRED.search(after) or OPTIONAL.search(after)
        ):
            points.add(match.end())
    return sorted(points)


def sentence_clauses(raw, matches):
    points = {0, len(raw)}
    for punctuation in re.finditer(r";|[.!?](?=\s)", raw):
        if not any(m.start() <= punctuation.start() < m.end() for m in matches):
            points.add(punctuation.end())
    ordered = sorted(points)
    return [(start, raw[start:end]) for start, end in zip(ordered, ordered[1:])]


class ImprovedRuleBasedExtractor:
    pipeline_version = 2

    def __init__(self, normalizer: SkillNormalizer | None = None, lexicon_path=None):
        self.lexicon = None
        extra = {}
        if lexicon_path:
            from resumeanalysis.extraction.phrase_lexicon import load_lexicon

            self.lexicon = load_lexicon(lexicon_path)
            extra = phrase_aliases(self.lexicon["phrases"])
        self.normalizer = normalizer or make_normalizer(extra)
        entries = sorted(self.normalizer.lookup.items(), key=lambda pair: (-len(pair[0]), pair[0]))
        self.entries = dict(entries)
        aliases = [a for a, _ in entries] + ["torch", "node", "R", "C", "Go"]
        self.pattern = re.compile(
            r"(?<![\w+#])(?:" + "|".join(alias_pattern(a) for a in aliases) + r")(?![\w+#])", re.I
        )
        # A small index handles whitespace around punctuation without a second normalizer.
        self.surface = {re.sub(r"\s*([.\-])\s*", r"\1", a): name for a, name in entries}
        self._metadata = SkillExtractor()

    @property
    def identity(self):
        return (
            "rules-v2:"
            + hashlib.sha256(
                json.dumps(
                    [
                        identity_source_hashes(Path(__file__).parent),
                        hashlib.sha256(
                            Path(__file__)
                            .parent.parent.joinpath("normalization/skill_normalizer.py")
                            .read_bytes()
                        ).hexdigest(),
                        self.normalizer.taxonomy.identity,
                        self.normalizer.lookup,
                        self.surface,
                        self.pattern.pattern,
                        self.pattern.flags,
                        self.lexicon["identity"] if self.lexicon else None,
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )

    def _skill(self, match, context):
        text = match.group()
        key = re.sub(r"\s*([.\-])\s*", r"\1", clean(text))
        if key == "node":
            return "node.js" if NODE_CONTEXT.search(context) else None
        if key in {"r", "c", "go"}:
            if not LANGUAGE_CONTEXT.search(context) or (key in {"r", "c"} and text != text.upper()):
                return None
            if key == "go" and re.search(r"\bgo\s+(?:to|on|for|ahead|through)\b", context, re.I):
                return None
            if key == "c" and re.search(
                r"\b(?:grade|rating|level)\s+C\b|\bC\s+(?:grade|rating|level)\b", context, re.I
            ):
                return None
            return {"go": "golang"}.get(key, key)
        canonical = self.surface.get(key, key)
        return self.normalizer.normalize(canonical, context)

    def extract(self, text: str, *, vacancy: bool = False) -> EnhancedExtraction:
        if not isinstance(text, str):
            raise TypeError("Skill extraction requires text")
        result = EnhancedExtraction()
        section = "mentioned"
        for line in re.finditer(r"[^\n]+", text):
            raw = line.group()
            stripped = re.sub(r"^[\s#*•\-]+", "", raw).strip()
            matches = list(self.pattern.finditer(raw))
            if RESET.search(stripped):
                section = "mentioned"
            # A header may carry skills after its colon, but prose is local scope.
            heading = stripped.split(":", 1)[0]
            if vacancy and (not matches or ":" in stripped) and len(heading.split()) <= 6:
                if OPTIONAL.search(heading):
                    section = "optional"
                elif REQUIRED.search(heading):
                    section = "required"
            for clause_start, content in sentence_clauses(raw, matches):
                # Contrast separates negated/optional scope inside the same sentence.
                boundaries = scoped_boundaries(content)
                for left, right in zip(boundaries, boundaries[1:]):
                    segment = content[left:right]
                    base = line.start() + clause_start + left
                    local = (
                        "optional"
                        if OPTIONAL.search(segment)
                        else ("required" if REQUIRED.search(segment) else section)
                    )
                    negated = vacancy and bool(NOT_REQUIRED.search(segment))
                    if negated:
                        local = "negated"
                    mentions = []
                    for match in self.pattern.finditer(segment):
                        skill = self._skill(match, raw)
                        if skill is None:
                            continue
                        prefix = segment[: match.start()]
                        status = (
                            "confirmed"
                            if vacancy
                            else (
                                "denied"
                                if NEGATIVE.search(prefix)
                                else ("unknown" if UNCERTAIN.search(prefix) else "confirmed")
                            )
                        )
                        provenance = (
                            "skillspan-train-phrase"
                            if self.lexicon and clean(match.group()) in self.lexicon["phrases"]
                            else "curated-v2"
                        )
                        mentions.append(
                            EvidenceMention(
                                skill,
                                match.group(),
                                base + match.start(),
                                base + match.end(),
                                status,
                                local,
                                provenance,
                            )
                        )
                    groups = []
                    if vacancy:
                        for mention in mentions:
                            connector = text[groups[-1][-1].end : mention.start] if groups else ""
                            previous_has_or = bool(groups) and any(
                                ALTERNATIVE.fullmatch(text[a.end : b.start])
                                for a, b in zip(groups[-1], groups[-1][1:])
                            )
                            if groups and (
                                ALTERNATIVE.fullmatch(connector)
                                or (COMMA.fullmatch(connector) and not previous_has_or)
                            ):
                                groups[-1].append(mention)
                            else:
                                groups.append([mention])
                        for group in groups:
                            has_or = any(
                                ALTERNATIVE.fullmatch(text[a.end : b.start])
                                for a, b in zip(group, group[1:])
                            )
                            if len(group) > 1 and has_or:
                                result.alternative_groups.append(
                                    AlternativeGroup(
                                        tuple(dict.fromkeys(m.skill for m in group)),
                                        local,
                                        group[0].start,
                                        group[-1].end,
                                        text[group[0].start : group[-1].end],
                                    )
                                )
                                for m in group:
                                    m.requirement = "alternative"
                    for mention in mentions:
                        result.mentions.append(mention)
                        if vacancy and local == "negated":
                            result.negated_requirements.add(mention.skill)
                        elif not vacancy or mention.requirement != "negated":
                            getattr(
                                result,
                                {
                                    "confirmed": "confirmed",
                                    "denied": "denied",
                                    "unknown": "uncertain",
                                }[mention.status],
                            ).add(mention.skill)
                        if vacancy and mention.requirement in {"required", "optional"}:
                            getattr(result, mention.requirement).add(mention.skill)
        conflicting = result.confirmed & result.denied
        result.uncertain |= conflicting
        result.confirmed -= conflicting
        result.denied -= conflicting
        result.optional -= result.required
        # Preserve baseline duration/seniority behavior without sharing its skill semantics.
        duration = self._metadata.extract(text, vacancy=vacancy)
        result.experience_years, result.seniority = duration.experience_years, duration.seniority
        return result
