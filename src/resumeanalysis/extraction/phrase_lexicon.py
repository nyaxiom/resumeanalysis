"""Load optional phrase artifacts fitted from the pinned MIT SkillSpan training split.

This artifact is an exact phrase matcher, not a pretrained neural NER model.
Training/test text and generated lexicons remain outside version control.
"""

import hashlib
from pathlib import Path

from resumeanalysis.jsonio import loads

TRAIN_SHA256 = "5e884c3426087d14fc437d328eb72132082edbfe1e776292ae0e6e9b36c1e1f6"
REVISION = "2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b"


def _identity(document):
    import json

    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in document.items() if k != "identity"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


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
