"""Text-only contamination fingerprints: no held-out annotation is decoded or exported."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis_legacy.skill_data.sources import TEST_SHA, file_sha

FINGERPRINT_COUNT = 2660
FINGERPRINT_SHA256 = "8d15c8d470cc8aa7cc35bab9ccb6e1e5fe6e3c6e9d5faf41e51a3263b7a1ff68"


def load_guard(path):
    result = loads(Path(path).read_text())
    fingerprints = result.get("fingerprints", {})
    checksum = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        result.get("source_sha256") != TEST_SHA
        or result.get("source_rows") != 3569
        or len(fingerprints) != FINGERPRINT_COUNT
        or checksum != FINGERPRINT_SHA256
    ):
        raise ValueError("Pinned SkillSpan test text guard integrity mismatch")
    return result


def normalized(text):
    return " ".join(text.casefold().split())


def text_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def shingles(text):
    words = re.findall(r"\w+|[^\w\s]", normalized(text))
    width = min(3, len(words))
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + width]).encode(), digest_size=8).digest()
        )
        for i in range(len(words) - width + 1)
    }


SALTS = [int.from_bytes(hashlib.sha256(f"skill-data:{i}".encode()).digest()[:8]) for i in range(32)]


def signature(values):
    return tuple(min(v ^ salt for v in values) for salt in SALTS)


def bands(sig):
    return [(i, sig[i * 4 : i * 4 + 4]) for i in range(8)]


class NearIndex:
    """Deterministic 32-hash, 8-band candidate search, exact shingle Jaccard verification.

    This is approximate candidate retrieval, not proof of absence of near-duplicates.
    Large buckets fail closed rather than quietly skipping candidates.
    """

    def __init__(self, threshold=0.85, bucket_limit=20000):
        if not 0 < threshold <= 1:
            raise ValueError("Jaccard threshold must be in (0, 1]")
        self.threshold, self.bucket_limit = threshold, bucket_limit
        self.buckets = defaultdict(list)
        self.values = {}

    def query(self, values):
        candidates = set()
        for band in bands(signature(values)):
            bucket = self.buckets[band]
            if len(bucket) > self.bucket_limit:
                raise ValueError("Near-duplicate bucket limit exceeded; audit incomplete")
            candidates.update(bucket)
        found = []
        for key in sorted(candidates):
            other = self.values[key]
            intersection = len(values & other)
            score = intersection / len(values | other)
            if score >= self.threshold:
                found.append((key, score))
        return found

    def add(self, key, values):
        self.values[key] = values
        for band in bands(signature(values)):
            self.buckets[band].append(key)


def token_text_only(line):
    # Decode only the array following the tokens key. The remainder, including
    # tags_skill/tags_knowledge, is never passed to a JSON decoder.
    match = re.search(r'(?:^|,)\s*"tokens"\s*:', line.lstrip().removeprefix("{"))
    if not match:
        raise ValueError("Missing text tokens in sealed source")
    tokens, _ = json.JSONDecoder().raw_decode(
        line.lstrip().removeprefix("{")[match.end() :].lstrip()
    )
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(t, str) or not t for t in tokens)
    ):
        raise ValueError("Invalid sealed text tokens")
    return " ".join(tokens)


def build_guard(path, output):
    if file_sha(path) != TEST_SHA:
        raise ValueError("Guard requires the exact pinned SkillSpan test bytes")
    rows, unique = 0, {}
    for line in Path(path).read_text().splitlines():
        text = token_text_only(line)
        unique[text_hash(text)] = sorted(shingles(text))
        rows += 1
    if file_sha(path) != TEST_SHA:
        raise ValueError("Sealed source changed during text fingerprinting")
    result = {
        "schema_version": 1,
        "source_sha256": TEST_SHA,
        "source_rows": rows,
        "operation": "text tokens only; no held-out annotation decoded, exported or scored",
        "normalization": "casefold and collapse whitespace; 3-token punctuation-aware shingles",
        "fingerprints": unique,
    }
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, result)
    load_guard(output)
    return {"source_rows": rows, "unique_texts": len(unique), "guard_sha256": file_sha(output)}
