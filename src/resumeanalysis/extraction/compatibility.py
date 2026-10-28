"""Keep the reviewed import-only refactor compatible with existing extractor state.

Only the complete, exact recorded source set may use the previous identity.
Any subsequent implementation change receives its actual new source fingerprint.
"""

import hashlib
import json
from pathlib import Path

IDENTITY_FILES = ("improved.py", "vocabulary.py", "phrase_lexicon.py", "skill_extractor.py")


def identity_source_hashes(directory: Path) -> dict[str, str]:
    current = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (*IDENTITY_FILES, "compatibility.py")
    }
    manifest = json.loads((directory / "identity_compatibility.json").read_text())
    if current == manifest["refactored_sources"]:
        return manifest["previous_sources"]
    return current
