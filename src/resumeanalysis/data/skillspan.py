"""Evaluate the dictionary fallback against ALL skill/knowledge spans, not a filtered subset."""

import json
from collections import Counter
from pathlib import Path

from resumeanalysis.extraction.skill_extractor import SkillExtractor


def bio_spans(tokens: list[str], tags: list[str]) -> set[tuple[int, int]]:
    if len(tokens) != len(tags):
        raise ValueError("SkillSpan tokens and tags differ in length")
    result, start = set(), None
    for i, tag in enumerate([*tags, "O"]):
        if tag.startswith("B") or tag == "O":
            if start is not None:
                result.add((start, i))
                start = None
        if tag.startswith("B") or (tag.startswith("I") and start is None):
            start = i
        if tag != "O" and not tag.startswith(("B", "I")):
            raise ValueError("Unexpected BIO tag")
    return result


def evaluate_skillspan(path: str | Path) -> dict:
    extractor = SkillExtractor()
    span_tp = span_fp = span_fn = token_tp = token_fp = token_fn = rows = 0
    missed = Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row["tokens"]
        gold = bio_spans(tokens, row["tags_skill"]) | bio_spans(tokens, row["tags_knowledge"])
        text = " ".join(tokens)
        starts, ends, cursor = [], [], 0
        for token in tokens:
            starts.append(cursor)
            ends.append(cursor + len(token))
            cursor += len(token) + 1
        predicted = set()
        for mention in extractor.extract(text).mentions:
            indices = [
                i
                for i, (a, b) in enumerate(zip(starts, ends))
                if a < mention.end and b > mention.start
            ]
            if indices:
                predicted.add((min(indices), max(indices) + 1))
        span_tp += len(gold & predicted)
        span_fp += len(predicted - gold)
        span_fn += len(gold - predicted)
        gt = {i for a, b in gold for i in range(a, b)}
        pt = {i for a, b in predicted for i in range(a, b)}
        token_tp += len(gt & pt)
        token_fp += len(pt - gt)
        token_fn += len(gt - pt)
        missed.update(" ".join(tokens[a:b]).casefold() for a, b in gold - predicted)
        rows += 1

    def report(tp, fp, fn):
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }

    if not rows:
        raise ValueError("No SkillSpan examples evaluated")
    return {
        "rows": rows,
        "exact_span": report(span_tp, span_fp, span_fn),
        "token": report(token_tp, token_fp, token_fn),
        "common_missed_spans": missed.most_common(15),
        "scope": "All SKILL and KNOWLEDGE annotations; finite RU/EN dictionary fallback, not trained NER",
    }
