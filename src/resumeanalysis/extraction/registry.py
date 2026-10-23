"""Extractor interface, explicit selection and checkpoint identity boundaries."""

import hashlib
from pathlib import Path
from typing import Protocol

from resumeanalysis.extraction.skill_extractor import Extraction, SkillExtractor
from resumeanalysis.normalization.skill_normalizer import SkillNormalizer
from resumeanalysis.normalization.taxonomy import Taxonomy

LegacyRuleBasedExtractor = SkillExtractor


class Extractor(Protocol):
    normalizer: SkillNormalizer

    def extract(self, text: str, *, vacancy: bool = False) -> Extraction: ...


def is_legacy(extractor) -> bool:
    standard = (
        type(extractor) is SkillExtractor
        and type(extractor.normalizer) is SkillNormalizer
        and type(extractor.normalizer.taxonomy) is Taxonomy
        and "extract" not in extractor.__dict__
        and "normalize" not in extractor.normalizer.__dict__
    )
    if not standard:
        return False
    path = Path(__file__).parent
    if (
        hashlib.sha256((path / "skill_extractor.py").read_bytes()).hexdigest()
        != "ee89e4804cc7871da02afe422b21b0a25e71a161726b948cc96bfeb08217000d"
        or hashlib.sha256(
            (path.parent / "normalization/skill_normalizer.py").read_bytes()
        ).hexdigest()
        != "a9ed664bf332b42d84c7be5f8da0b91c3564d9c6a89062de51d236074042f9c8"
    ):
        return False
    reference = SkillExtractor(SkillNormalizer(extractor.normalizer.taxonomy))
    return extractor.normalizer.lookup == reference.normalizer.lookup and (
        extractor.pattern.pattern,
        extractor.pattern.flags,
    ) == (reference.pattern.pattern, reference.pattern.flags)


def extractor_identity(extractor) -> str:
    if is_legacy(extractor):
        return "legacy-rules-v1:ee89e4804cc7871da02afe422b21b0a25e71a161726b948cc96bfeb08217000d"
    from resumeanalysis.extraction.improved import ImprovedRuleBasedExtractor

    if (
        type(extractor) is not ImprovedRuleBasedExtractor
        or type(extractor.normalizer) is not SkillNormalizer
    ):
        raise ValueError(
            "Unknown extractor/normalizer implementation; explicit registered identity required"
        )
    return extractor.identity


def create_extractor(name="legacy", *, lexicon_path=None) -> Extractor:
    if name == "legacy":
        if lexicon_path:
            raise ValueError("Legacy extraction cannot load a v2 phrase lexicon")
        return SkillExtractor()
    if name == "v2":
        from resumeanalysis.extraction.improved import ImprovedRuleBasedExtractor

        return ImprovedRuleBasedExtractor(lexicon_path=lexicon_path)
    raise ValueError("Extractor must be legacy or v2")
