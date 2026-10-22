import re
import unicodedata

from .taxonomy import Taxonomy


def clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class SkillNormalizer:
    def __init__(self, taxonomy: Taxonomy | None = None):
        self.taxonomy = taxonomy or Taxonomy()
        self.lookup = {clean(a): k for k, aliases in self.taxonomy.aliases.items() for a in aliases}

    def normalize(self, skill: str, context: str = "") -> str:
        name = clean(skill)
        # Torch can also mean the Lua framework or an ordinary torch.
        if name == "torch" and re.search(r"\b(pytorch|python|tensor|tensors)\b", context, re.I):
            return "pytorch"
        return self.lookup.get(name, name)

    def normalize_many(self, skills: list[str], context: str = "") -> set[str]:
        return {self.normalize(s, context) for s in skills if s.strip()}
