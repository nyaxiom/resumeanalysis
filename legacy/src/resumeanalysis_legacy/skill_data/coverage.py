"""Train-only vocabulary intervention and validation-only error decomposition."""

import re
from collections import Counter, defaultdict
from pathlib import Path

from resumeanalysis.extraction.improved import ImprovedRuleBasedExtractor
from resumeanalysis.extraction.skill_extractor import SkillExtractor
from resumeanalysis.extraction.vocabulary import make_normalizer, phrase_aliases
from resumeanalysis.jsonio import write_json
from resumeanalysis.normalization.skill_normalizer import clean
from resumeanalysis_legacy.extraction.phrase_lexicon import AMBIGUOUS
from resumeanalysis_legacy.skill_data.audit import partition_records
from resumeanalysis_legacy.skill_data.sources import digest, file_sha

CONTEXT_CUE = re.compile(
    r"\b(?:ability|able|develop\w*|manag\w*|analy[sz]\w*|collaborat\w*|maintain\w*|implement\w*)\b"
    r"|(?:умени|способност|разработ|управлен|анализ|взаимодейств)",
    re.I,
)


def fit_phrases(audit_root, datasets, min_frequency=2, max_tokens=12):
    """Fit only span gold from eligible TRAIN, with sentence-level frequency counting."""
    if min_frequency < 1 or max_tokens < 1 or not datasets:
        raise ValueError("Declare positive frequency/token bounds and training datasets")
    counts, seen, sources = Counter(), set(), {}
    rows = mentions = 0
    for record in partition_records(audit_root, roles={"train"}, datasets=set(datasets)):
        if record.annotation_type != "span":
            raise ValueError("Vocabulary fitting requires original span annotations")
        rows += 1
        sources[record.provenance.dataset] = record.provenance.source_sha256
        for annotation in record.annotations:
            if annotation.category not in {"SKILL", "KNOWLEDGE"}:
                continue
            mentions += 1
            phrase = clean(annotation.surface)
            if (
                3 <= len(phrase)
                and len(phrase.split()) <= max_tokens
                and phrase not in AMBIGUOUS
                and any(c.isalpha() for c in phrase)
                and not re.search(r"\b(?:or|или|либо)\b", phrase)
            ):
                key = (clean(record.text), phrase)
                if key not in seen:
                    counts[phrase] += 1
                    seen.add(key)
    if set(datasets) != set(sources):
        raise ValueError("Requested training dataset absent or has no eligible records")
    return {
        "schema_version": 1,
        "role": "train_only_experimental_vocabulary",
        "audit_sha256": file_sha(Path(audit_root) / "audit.json"),
        "training_sources": sources,
        "training_rows": rows,
        "training_mentions": mentions,
        "min_frequency": min_frequency,
        "max_tokens": max_tokens,
        "frequency_unit": "unique normalized sentence/phrase pair, across selected training sources",
        "phrases": sorted(p for p, n in counts.items() if n >= min_frequency),
        "limitations": "Literal phrase intervention, not semantic NER or production vocabulary. No test/validation gold harvesting.",
    }


def metrics(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def token_bounds(record):
    cursor, positions = 0, []
    for token in record.tokens:
        positions.append((cursor, cursor + len(token)))
        cursor += len(token) + 1
    return positions


def interval(start, end, positions):
    indices = [i for i, (a, b) in enumerate(positions) if a < end and b > start]
    if not indices:
        raise ValueError("Predicted/gold mention overlaps no token")
    return min(indices), max(indices) + 1


def evaluate_validation(records, extractor):
    totals, groups, errors = Counter(), defaultdict(Counter), Counter()
    vocabulary = extractor.normalizer.lookup
    for record in records:
        if record.provenance.role != "validation" or record.provenance.dataset != "skillspan":
            raise ValueError("This diagnostic accepts only audited SkillSpan validation")
        positions = token_bounds(record)
        gold = defaultdict(list)
        for a in record.annotations:
            if a.category in {"SKILL", "KNOWLEDGE"}:
                gold[interval(a.start, a.end, positions)].append(a)
        result = extractor.extract(record.text)
        predicted, chars = set(), set()
        for mention in result.mentions:
            if record.text[mention.start : mention.end] != mention.text:
                raise ValueError("Extractor returned nonliteral evidence")
            predicted.add(interval(mention.start, mention.end, positions))
            chars.add((mention.start, mention.end))
        totals.update(
            rows=1,
            gold=len(gold),
            predictions=len(predicted),
            tp=len(set(gold) & predicted),
            fp=len(predicted - set(gold)),
            fn=len(set(gold) - predicted),
        )
        for (begin, end), annotations in gold.items():
            a = annotations[0]
            surface = a.surface
            alias_known = clean(surface) in vocabulary
            pattern_known = extractor.pattern.fullmatch(surface) is not None
            correct = (begin, end) in predicted
            overlap = any(p < end and q > begin for p, q in predicted)
            totals["alias_covered"] += alias_known
            totals["pattern_covered"] += pattern_known
            totals["char_tp"] += (a.start, a.end) in chars
            bins = [
                "alias_known" if alias_known else "alias_unknown",
                "pattern_known" if pattern_known else "pattern_unknown",
                "single_word" if end - begin == 1 else "multi_word",
                "length_1"
                if end - begin == 1
                else "length_2_3"
                if end - begin <= 3
                else "length_4_plus",
                "contextual_surface_proxy"
                if CONTEXT_CUE.search(surface)
                else "no_contextual_surface_cue",
                "language:" + record.language,
                *["category:" + name for name in sorted({v.category for v in annotations})],
            ]
            for group in bins:
                groups[group].update(gold=1, tp=int(correct), alias_covered=int(alias_known))
            if not correct:
                if overlap:
                    errors["overlapping_prediction_wrong_boundary"] += 1
                else:
                    errors["no_overlapping_prediction"] += 1
                if not alias_known and extractor.pattern.search(surface):
                    errors["unknown_full_surface_contains_known_substring"] += 1
                if pattern_known:
                    errors["full_pattern_known_but_missed_in_context"] += 1
    if not totals["rows"]:
        raise ValueError("No eligible SkillSpan validation rows")
    return {
        "rows": totals["rows"],
        "gold_mentions": totals["gold"],
        "token_exact": metrics(totals["tp"], totals["fp"], totals["fn"]),
        "char_exact_recall": totals["char_tp"] / totals["gold"] if totals["gold"] else None,
        "vocabulary": {
            "alias_entries": len(vocabulary),
            "gold_alias_coverage": totals["alias_covered"] / totals["gold"]
            if totals["gold"]
            else None,
            "gold_full_pattern_coverage": totals["pattern_covered"] / totals["gold"]
            if totals["gold"]
            else None,
        },
        "groups": {
            name: dict(count) | {"recall": count["tp"] / count["gold"]}
            for name, count in sorted(groups.items())
        },
        "miss_diagnostics": dict(errors),
    }


def run_coverage(audit_root, output, *, min_frequency=2):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new experiment directory")
    output.mkdir(parents=True)
    validation = list(partition_records(audit_root, roles={"validation"}, datasets={"skillspan"}))
    if not validation:
        raise ValueError("No audited validation partition")
    experiments = {"legacy": SkillExtractor(), "v2": ImprovedRuleBasedExtractor()}
    fitted = {}
    for name, datasets in [
        ("expanded_skillspan", ["skillspan"]),
        ("expanded_skillspan_green", ["skillspan", "green"]),
    ]:
        artifact = fit_phrases(audit_root, datasets, min_frequency=min_frequency)
        notices = []
        for record in partition_records(audit_root, roles={"train"}, datasets=set(datasets)):
            notice = (
                record.provenance.dataset,
                record.provenance.attribution,
                record.provenance.license,
            )
            if notice not in notices:
                notices.append(notice)
        artifact["attributions"] = sorted(notices)
        # Include original full license text in each derivative artifact.
        root = Path(__file__).resolve().parents[3]
        artifact["license_notices"] = {
            "MIT SkillSpan": (root / "LICENSES/references/skillspan-LICENSE.txt").read_text(),
        }
        if "green" in datasets:
            artifact["license_notices"]["CC-BY-4.0 Green et al."] = (
                root / "LICENSES/datasets/green-LICENSE.txt"
            ).read_text()
        artifact["identity"] = digest(str(artifact).encode())
        write_json(output / (name + ".json"), artifact)
        fitted[name] = {
            k: v for k, v in artifact.items() if k not in {"phrases", "license_notices"}
        }
        fitted[name]["phrase_count"] = len(artifact["phrases"])
        experiments[name] = ImprovedRuleBasedExtractor(
            normalizer=make_normalizer(phrase_aliases(artifact["phrases"]))
        )
    results = {
        name: evaluate_validation(validation, extractor) for name, extractor in experiments.items()
    }
    report = {
        "schema_version": 1,
        "scope": "One preregistered min-frequency intervention; audited SkillSpan validation only; no held-out test scoring, tuning or model selection",
        "audit_sha256": file_sha(Path(audit_root) / "audit.json"),
        "validation_partition": "skillspan/validation; exact/near heldout and training overlaps quarantined; within-partition exact/conflicting duplicates excluded",
        "definitions": {
            "alias_coverage": "Full gold surface after NFKC/casefold/whitespace cleaning equals an existing alias. No semantic synonym assumption.",
            "pattern_coverage": "Full gold surface matches the extractor's lexical regex, before sentence/context guards; distinct from alias lookup.",
            "known_unknown": "Defined independently of predictions from fixed vocabulary/regex; recall counts exact token boundaries over all gold SKILL/KNOWLEDGE mentions.",
            "contextual_proxy": "Gold surface contains a declared action/ability cue; this is a diagnostic proxy, not a human contextual-skill or requirement label.",
            "metrics": "Token-boundary exact P/R/F1, same union of SKILL and KNOWLEDGE as historical SkillSpan scoring; character exact recall also reported. All mention statuses count for extraction, not competence assertion.",
        },
        "fitted_vocabularies": fitted,
        "results": results,
        "russian": {
            "span_recall": None,
            "reason": "No verified compatible independent Russian human span gold was found. Russian title skill lists are classification labels, not source mentions; do not compare their label coverage with English span recall.",
        },
        "limitations": [
            "Validation diagnosis is not an independent final-test estimate; original test was historically exposed before this task and is now frozen.",
            "More literal vocabulary is an intervention on matching coverage, not evidence that naive dataset concatenation or synthetic scale improves semantic NER.",
            "No downstream resume-vacancy model was trained or selected; original extractor files and checkpoints are untouched.",
        ],
    }
    write_json(output / "coverage.json", report)
    return report
