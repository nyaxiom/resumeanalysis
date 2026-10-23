"""Optional train-only phrase lexicon from the pinned MIT SkillSpan training split.

This artifact is an exact phrase matcher, not a pretrained neural NER model.
Training/test text and generated lexicons remain outside version control.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

from resumeanalysis.data.skillspan import bio_spans
from resumeanalysis.jsonio import loads, write_json

TRAIN_SHA256 = "5e884c3426087d14fc437d328eb72132082edbfe1e776292ae0e6e9b36c1e1f6"
REVISION = "2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b"
AMBIGUOUS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "c",
    "for",
    "go",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "r",
    "the",
    "to",
    "us",
}


def _identity(document):
    import json

    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in document.items() if k != "identity"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def build_lexicon(source, output, *, min_frequency=2, max_tokens=12) -> dict:
    if min_frequency < 1 or max_tokens < 1:
        raise ValueError("Phrase frequency and token limits must be positive")
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError(
            "Expected the exact pinned SkillSpan TRAIN split; test/dev/modified data rejected"
        )
    if output.exists():
        raise FileExistsError("Use a new lexicon artifact path")
    counts, spans, rows = Counter(), 0, 0
    for line in raw.decode().splitlines():
        row = loads(line)
        tokens = row["tokens"]
        for start, end in bio_spans(tokens, row["tags_skill"]) | bio_spans(
            tokens, row["tags_knowledge"]
        ):
            spans += 1
            phrase = " ".join(tokens[start:end]).casefold().strip()
            if (
                end - start <= max_tokens
                and len(phrase) >= 3
                and phrase not in AMBIGUOUS
                and not re.search(r"\b(?:or|или|либо)\b", phrase)
                and any(c.isalpha() for c in phrase)
            ):
                counts[phrase] += 1
        rows += 1
    phrases = sorted(p for p, n in counts.items() if n >= min_frequency)
    if not phrases:
        raise ValueError("No phrases survived the declared training filters")
    document = {
        "format_version": 1,
        "source": "kris927b/SkillSpan",
        "revision": REVISION,
        "source_split": "train",
        "source_sha256": TRAIN_SHA256,
        "license": "MIT",
        "copyright": "Copyright (c) 2023 Mike Zhang",
        "parameters": {
            "min_frequency": min_frequency,
            "max_tokens": max_tokens,
            "ambiguous_single_terms_excluded": sorted(AMBIGUOUS),
            "disjunction_phrases_excluded": True,
        },
        "training_rows": rows,
        "annotated_spans": spans,
        "phrases": phrases,
        "limitations": "Train-only lexical coverage; no semantic NER, no Russian benchmark, no validation/test phrase harvesting",
    }
    # Retain the actual upstream notice inside the derivative artifact as well.
    from importlib.metadata import distribution

    notice = Path(__file__).resolve().parents[3] / "LICENSES/references/skillspan-LICENSE.txt"
    if not notice.exists():
        dist = distribution("resumeanalysis")
        notice = next(
            (
                dist.locate_file(p)
                for p in dist.files or []
                if str(p).endswith("LICENSES/references/skillspan-LICENSE.txt")
            ),
            None,
        )
    if not notice or not Path(notice).is_file():
        raise FileNotFoundError("Retained original SkillSpan MIT notice unavailable")
    document["license_notice"] = Path(notice).read_text()
    document["identity"] = _identity(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, document)
    return {k: v for k, v in document.items() if k not in {"phrases", "license_notice"}} | {
        "phrase_count": len(phrases)
    }


def load_lexicon(path) -> dict:
    document = loads(Path(path).read_text())
    if (
        document.get("format_version") != 1
        or document.get("source_sha256") != TRAIN_SHA256
        or document.get("source_split") != "train"
        or document.get("license") != "MIT"
        or document.get("revision") != REVISION
        or document.get("identity") != _identity(document)
    ):
        raise ValueError("Phrase lexicon integrity/source provenance mismatch")
    phrases = document.get("phrases")
    if (
        not isinstance(phrases, list)
        or not phrases
        or any(not isinstance(p, str) or not p.strip() for p in phrases)
    ):
        raise ValueError("Invalid phrase vocabulary")
    return document
