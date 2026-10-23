"""CPU skill-extractor evaluation without changing the historical SkillSpan baseline.

SkillSpan supplies span labels, not requirement categories or recruitment labels.
Role Radar reports are descriptive unless independent annotations are supplied.
"""

import dataclasses
import hashlib
import html
import math
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from resumeanalysis.data.adapters import fields
from resumeanalysis.data.download import sha256_file
from resumeanalysis.data.skillspan import bio_spans
from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.schemas import Profile, Vacancy

SKILLSPAN_REVISION = "2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b"
SKILLSPAN_TEST_SHA256 = "8438eda34939a5877b984ea5daf9e6ff39b2d5c7e3d73cca0499d3de3784d24a"
CATEGORIES = ("required", "optional", "mentioned", "negated_requirements", "alternative_groups")
SCORE_FIELDS = {"score", "scores", "composite_score", "role_radar_score", "relevance_score"}


def serializable(value):
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(serializable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Extractor result is not a supported structured value")


def extraction_details(result, text: str) -> dict:
    """Validate offsets before reporting them; never quietly drop malformed spans."""
    for mention in result.mentions:
        if (
            type(mention.start) is not int
            or type(mention.end) is not int
            or not 0 <= mention.start < mention.end <= len(text)
            or text[mention.start : mention.end] != mention.text
        ):
            raise ValueError("Extractor produced an invalid source span")
    return serializable(result)


def _source_id(checksum, index):
    return hashlib.sha256(f"{checksum}:{index}".encode()).hexdigest()


def _metrics(tp, fp, fn):
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _peak_rss():
    try:
        import resource

        amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return amount if sys.platform == "darwin" else amount * 1024
    except ImportError:
        return None


def _performance(wall, cpu, latencies):
    return {
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "average_process_cpu_percent": 100 * cpu / wall if wall else None,
        "process_peak_rss_bytes": _peak_rss(),
        "extraction_wall_seconds": sum(latencies),
        "mean_extraction_seconds": statistics.mean(latencies) if latencies else None,
        "p50_extraction_seconds": _percentile(latencies, 0.5),
        "p95_extraction_seconds": _percentile(latencies, 0.95),
        "measurement_scope": "Wall/CPU start after source bytes are loaded and include JSON parsing and metric aggregation; extraction latency times extract calls only. Peak RSS is process-lifetime high-water including interpreter, imports and loaded model, not standalone extractor memory. CPU percent uses one core as 100%; platforms without resource report RSS=null.",
    }


def _phrases(counter, limit):
    return [
        {"text": phrase, "count": count}
        for phrase, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def evaluate_skillspan(
    path, extractor, *, example_limit=20, include_records=False, dataset_revision=None
) -> dict:
    """Evaluate every line and every SKILL/KNOWLEDGE gold span on the same input."""
    if example_limit < 0:
        raise ValueError("example_limit must be nonnegative")
    path = Path(path)
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    span_tp = span_fp = span_fn = token_tp = token_fp = token_fn = count = 0
    false_positives, false_negatives = Counter(), Counter()
    examples, records, latencies = [], [], []
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for index, line in enumerate(content.decode("utf-8").splitlines(), 1):
        row = loads(line)  # Blank/malformed lines fail; no unseen filtering.
        if not isinstance(row, dict):
            raise ValueError(f"SkillSpan row {index} must be an object")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, str) and t for t in tokens):
            raise ValueError(f"SkillSpan row {index} has invalid tokens")
        for name in ("tags_skill", "tags_knowledge"):
            if not isinstance(row.get(name), list) or not all(
                isinstance(v, str) for v in row[name]
            ):
                raise ValueError(f"SkillSpan row {index} has invalid BIO tags")
        gold = bio_spans(tokens, row["tags_skill"]) | bio_spans(tokens, row["tags_knowledge"])
        text, starts, ends, cursor = " ".join(tokens), [], [], 0
        for token in tokens:
            starts.append(cursor)
            ends.append(cursor + len(token))
            cursor += len(token) + 1
        start = time.perf_counter()
        result = extractor.extract(text)
        latencies.append(time.perf_counter() - start)
        details = extraction_details(result, text)
        predicted = set()
        for mention in result.mentions:
            overlapping = [
                i
                for i, (a, b) in enumerate(zip(starts, ends))
                if a < mention.end and b > mention.start
            ]
            if overlapping:
                predicted.add((min(overlapping), max(overlapping) + 1))
            else:
                raise ValueError("Extractor span overlaps no source token")
        span_tp += len(gold & predicted)
        span_fp += len(predicted - gold)
        span_fn += len(gold - predicted)
        gt = {i for a, b in gold for i in range(a, b)}
        pt = {i for a, b in predicted for i in range(a, b)}
        token_tp += len(gt & pt)
        token_fp += len(pt - gt)
        token_fn += len(gt - pt)
        false_positives.update(" ".join(tokens[a:b]) for a, b in predicted - gold)
        false_negatives.update(" ".join(tokens[a:b]) for a, b in gold - predicted)
        item = {
            "source_id": _source_id(checksum, index),
            "source_line": index,
            "false_positives": [
                {"token_span": [a, b], "text": " ".join(tokens[a:b])}
                for a, b in sorted(predicted - gold)
            ],
            "false_negatives": [
                {"token_span": [a, b], "text": " ".join(tokens[a:b])}
                for a, b in sorted(gold - predicted)
            ],
        }
        if len(examples) < example_limit and (item["false_positives"] or item["false_negatives"]):
            examples.append(item)
        if include_records:
            records.append(
                {
                    **item,
                    "gold_token_spans": sorted(gold),
                    "predicted_token_spans": sorted(predicted),
                    "extraction": details,
                }
            )
        count += 1
    if not count:
        raise ValueError("No SkillSpan records evaluated")
    if sha256_file(path) != checksum:
        raise ValueError("SkillSpan source changed during evaluation")
    verified = checksum == SKILLSPAN_TEST_SHA256
    report = {
        "rows": count,
        "source": {
            "sha256": checksum,
            "dataset": "kris927b/SkillSpan"
            if verified
            else "Caller-supplied SkillSpan-compatible JSONL",
            "revision": SKILLSPAN_REVISION if verified else dataset_revision,
            "revision_status": "Pinned test hash verified"
            if verified
            else "Unverified/caller declared",
            "language": "en" if verified else "Not inferred from BIO format",
        },
        "exact_span": _metrics(span_tp, span_fp, span_fn),
        "token": _metrics(token_tp, token_fp, token_fn),
        "common_false_positive_spans": _phrases(false_positives, example_limit),
        "common_missed_spans": _phrases(false_negatives, example_limit),
        "error_examples": examples,
        "performance": _performance(
            time.perf_counter() - wall_start, time.process_time() - cpu_start, latencies
        ),
        "scope": "All SKILL and KNOWLEDGE BIO spans; exact token-boundary match after character-to-token overlap mapping, identical to legacy evaluation. Token metrics compare unions of covered token indices. No requirement-category, Russian or recruitment-accuracy claim follows from English SkillSpan.",
    }
    if include_records:
        report["record_diagnostics"] = records
    return report


def requirement_sets(result):
    groups = getattr(result, "alternative_groups", [])
    return {
        "required": set(result.required),
        "optional": set(result.optional),
        "mentioned": {m.skill for m in result.mentions if m.requirement == "mentioned"},
        "negated_requirements": set(getattr(result, "negated_requirements", set())),
        "alternative_groups": {tuple(sorted(g.skills)) for g in groups},
    }


def evaluate_fixtures(path, extractor) -> dict:
    path = Path(path)
    content = path.read_bytes()
    rows = loads(content.decode())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Controlled fixtures must be a nonempty JSON array")
    languages, details, latencies = {}, [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("language") not in {"en", "ru"}:
            raise ValueError("Each fixture needs an explicit English or Russian language")
        text, expected = row.get("text"), row.get("expected")
        if (
            not isinstance(text, str)
            or not isinstance(expected, dict)
            or not expected
            or not set(expected) <= set(CATEGORIES)
        ):
            raise ValueError("Unsupported controlled fixture schema")
        start = time.perf_counter()
        result = extractor.extract(text, vacancy=row.get("vacancy", True))
        latencies.append(time.perf_counter() - start)
        actual = requirement_sets(result)
        language = languages.setdefault(
            row["language"],
            {
                "rows": 0,
                "all_graded_categories_correct": 0,
                "category_counts": {c: [0, 0, 0] for c in CATEGORIES},
                "category_graded_rows": dict.fromkeys(CATEGORIES, 0),
            },
        )
        errors = {}
        for category, values in expected.items():
            if not isinstance(values, list):
                raise ValueError("Expected categories must be lists")
            gold = (
                {tuple(sorted(v)) for v in values}
                if category == "alternative_groups"
                else set(values)
            )
            observed = actual[category]
            counts = language["category_counts"][category]
            language["category_graded_rows"][category] += 1
            counts[0] += len(gold & observed)
            counts[1] += len(observed - gold)
            counts[2] += len(gold - observed)
            if gold != observed:
                errors[category] = {
                    "false_positives": serializable(observed - gold),
                    "false_negatives": serializable(gold - observed),
                }
        language["rows"] += 1
        language["all_graded_categories_correct"] += not errors
        details.append(
            {
                "id": row.get("id", index),
                "language": row["language"],
                "extraction": extraction_details(result, text),
                "errors": errors,
            }
        )
    for language in languages.values():
        language["categories"] = {
            c: _metrics(*counts) if language["category_graded_rows"][c] else None
            for c, counts in language.pop("category_counts").items()
        }
    return {
        "rows": len(rows),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "languages": languages,
        "records": details,
        "mean_extraction_seconds": statistics.mean(latencies),
        "scope": "Manually authored regression fixtures, reported independently by language; not an independent real-world language benchmark. Only explicitly supplied expected categories are graded.",
    }


def _score_metadata(value):
    if value is None or type(value) is int:
        return value
    if isinstance(value, dict):
        return {str(k): _score_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_score_metadata(v) for v in value]
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Source score metadata must contain only finite numbers")


def evaluate_role_radar(path, extractor, *, backend=None) -> dict:
    """Offline per-pair extraction diagnostics; composite scores are never labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Role Radar input is unavailable; supply the local inference_pairs.jsonl path"
        )
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    outcomes, latencies, field_counts = [], [], Counter()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    profile_fields = set(Profile.model_fields) | {"preferred_roles"}
    vacancy_fields = set(Vacancy.model_fields)
    for index, raw in enumerate(content.splitlines(keepends=True), 1):
        item = {
            "source_id": _source_id(checksum, index),
            "source_line": index,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "status": "rejected",
            "diagnostics": [],
            "unmapped_fields": [],
            "source_scores_metadata": {},
        }
        try:
            row = loads(raw.decode())
            if not isinstance(row, dict):
                raise ValueError("Expected pair object")
            for field in fields(row):
                parts = field.split("/")
                known = len(parts) == 3 and (
                    (parts[1] == "profile" and parts[2] in profile_fields)
                    or (parts[1] == "vacancy" and parts[2] in vacancy_fields)
                )
                known = known or (len(parts) == 2 and parts[1] in {"profile", "vacancy"})
                known = known or (len(parts) >= 2 and parts[1] in SCORE_FIELDS)
                if not known:
                    item["unmapped_fields"].append(field)
                    field_counts[field] += 1
            profile, vacancy = row.get("profile"), row.get("vacancy")
            if not isinstance(profile, dict) or not isinstance(vacancy, dict):
                raise ValueError("Expected profile/vacancy objects")
            # Validate known fields individually via the canonical schemas, keeping
            # every unknown field as a reported omission rather than inserting it.
            p_known = {k: v for k, v in profile.items() if k in Profile.model_fields}
            v_known = {k: v for k, v in vacancy.items() if k in Vacancy.model_fields}
            p = Profile.model_validate(p_known, strict=True)
            v = Vacancy.model_validate(v_known, strict=True)
            if "preferred_roles" in profile and (
                not isinstance(profile["preferred_roles"], list)
                or not all(isinstance(role, str) for role in profile["preferred_roles"])
            ):
                raise ValueError("Preferred roles must be explicit strings")
            if "description" not in vacancy or not isinstance(vacancy["description"], str):
                raise ValueError("Vacancy description must be explicitly supplied")
            for key in SCORE_FIELDS & set(row):
                item["source_scores_metadata"][key] = _score_metadata(row[key])
            start = time.perf_counter()
            result = extractor.extract(v.description, vacancy=True)
            latencies.append(time.perf_counter() - start)
            effective_required = result.required | extractor.normalizer.normalize_many(
                v.required_skills, v.description
            )
            item.update(
                {
                    "status": "diagnosed_with_unmapped_fields"
                    if item["unmapped_fields"]
                    else "diagnosed",
                    "extraction": extraction_details(result, v.description),
                    "explicit_profile_skills": p.skills,
                    "explicit_vacancy_required_skills": v.required_skills,
                    "explicit_vacancy_optional_skills": v.optional_skills,
                    "normalized_explicit_required_skills": sorted(
                        extractor.normalizer.normalize_many(v.required_skills, v.description)
                    ),
                    "effective_required_skills": sorted(effective_required),
                    "required_not_in_explicit_profile": sorted(
                        effective_required
                        - extractor.normalizer.normalize_many(p.skills, p.resume_text)
                    ),
                    "unconfirmed_semantics": "Not in supplied profile skills means unconfirmed, not definitely absent",
                    "model_accuracy_metrics": None,
                }
            )
            if "preferred_roles" in profile:
                item["diagnostics"].append(
                    {
                        "code": "preferred_roles_preserved_in_source_only",
                        "field": "/profile/preferred_roles",
                    }
                )
            if backend is not None:
                model_start = time.perf_counter()
                try:
                    item["analysis"] = backend.predict(p, v).model_dump()
                    item["model_status"] = "succeeded"
                    item["score_interpretation"] = (
                        "Uncalibrated matching score; not a Role Radar gold label"
                    )
                except Exception as exc:
                    # Preserve extraction diagnostics and process later source rows;
                    # model failures must not echo resume text from exception messages.
                    item["model_status"] = "failed"
                    item["status"] = "diagnosed_with_model_error"
                    item["diagnostics"].append(
                        {"code": "model_inference_failed", "error_type": type(exc).__name__}
                    )
                finally:
                    item["model_inference_seconds"] = time.perf_counter() - model_start
        except (ValueError, TypeError, KeyError, UnicodeError):
            item["status"] = "rejected"
            item["diagnostics"].append({"code": "invalid_pair_or_extractor_result"})
        outcomes.append(item)
    if not outcomes:
        raise ValueError("Role Radar source is empty")
    if sha256_file(path) != checksum:
        raise ValueError("Role Radar source changed during evaluation")
    counts = Counter(o["status"] for o in outcomes)
    return {
        "source_sha256": checksum,
        "input_records": len(outcomes),
        "outcome_counts": dict(counts),
        "count_reconciliation": sum(counts.values()) == len(outcomes),
        "unmapped_field_counts": dict(sorted(field_counts.items())),
        "record_outcomes": outcomes,
        "performance": _performance(
            time.perf_counter() - wall_start, time.process_time() - cpu_start, latencies
        ),
        "accuracy_metrics": None,
        "scope": "Descriptive offline extraction diagnostics. No independently annotated skill spans or requirements are assumed. Role Radar composite scores are source metadata, never calibrated probabilities, binary ground truth or independent human judgments. Unknown fields are reported and not model features. Raw records remain with the source custodian; hashes are not full reversibility.",
    }


def benchmark_skillspan(
    path,
    names=("legacy", "v2"),
    *,
    lexicon_path=None,
    threads=2,
    include_records=False,
    isolated=True,
) -> dict:
    if not names or threads < 1 or len(set(names)) != len(names):
        raise ValueError("Unique extractor names and positive threads are required")
    checksum = sha256_file(path)
    reports = {}
    for name in names:
        if isolated:
            with tempfile.TemporaryDirectory(prefix="resumeanalysis-extractor-") as temporary:
                output = Path(temporary) / "report.json"
                command = [
                    sys.executable,
                    "-m",
                    "resumeanalysis.extraction.cli",
                    "_skillspan-worker",
                    "--input",
                    str(Path(path).resolve()),
                    "--extractor",
                    name,
                    "--output",
                    str(output),
                    "--threads",
                    str(threads),
                ]
                if lexicon_path and name == "v2":
                    command += ["--lexicon-path", str(Path(lexicon_path).resolve())]
                if include_records:
                    command.append("--include-records")
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                report = loads(output.read_text())
        else:
            from resumeanalysis.extraction.registry import create_extractor

            start = time.perf_counter()
            extractor = create_extractor(name, lexicon_path=lexicon_path if name == "v2" else None)
            initialization = time.perf_counter() - start
            report = evaluate_skillspan(path, extractor, include_records=include_records)
            report["initialization_seconds"] = initialization
        if report["source"]["sha256"] != checksum:
            raise ValueError("Extractor evaluations used different source data")
        report["extractor_name"] = name
        report["lexicon_sha256"] = (
            sha256_file(lexicon_path) if lexicon_path and name == "v2" else None
        )
        reports[name] = report
    return {
        "source_sha256": checksum,
        "backends": reports,
        "execution": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "isolated_processes": isolated,
            "configured_torch_threads": threads,
        },
        "memory_comparison_scope": "Fresh independent subprocess per extractor"
        if isolated
        else "Shared process; RSS peaks cannot be attributed independently to each extractor",
        "claim": "Compare precision, recall, F1 and errors together. English SkillSpan does not establish Russian or recruitment accuracy.",
    }


def comparison_html(report) -> str:
    """Standalone escaped HTML with inline SVG; no plotting/runtime dependencies."""
    backends = report.get("backends", {})
    colors = ("#2563eb", "#16a34a", "#a855f7", "#ea580c")
    chart = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="780" height="320" viewBox="0 0 780 320" role="img" aria-label="Exact-span precision, recall and F1">'
    ]
    chart.append('<rect width="780" height="320" fill="white"/>')
    for number in range(6):
        y = 255 - number * 42
        chart.append(
            f'<line x1="55" y1="{y}" x2="745" y2="{y}" stroke="#ddd"/><text x="12" y="{y + 4}" font-size="12">{number / 5:.1f}</text>'
        )
    bars = max(1, len(backends))
    for group, metric in enumerate(("precision", "recall", "f1")):
        x = 80 + group * 230
        chart.append(f'<text x="{x}" y="280" font-size="15">{metric}</text>')
        for index, (name, result) in enumerate(backends.items()):
            value = result["exact_span"][metric]
            width = min(60, 180 / bars)
            chart.append(
                f'<rect x="{x + index * width}" y="{255 - value * 210}" width="{width - 4}" height="{value * 210}" fill="{colors[index % len(colors)]}"><title>{html.escape(name)} {metric}: {value:.4f}</title></rect>'
            )
    for index, name in enumerate(backends):
        chart.append(
            f'<text x="{60 + index * 180}" y="308" fill="{colors[index % len(colors)]}" font-size="14">{html.escape(name)}</text>'
        )
    chart.append("</svg>")
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Skill extraction comparison</title><body><h1>Skill extraction comparison</h1>',
        "<p>All input spans evaluated. Metrics require gold annotations; no recruitment accuracy or calibrated-probability claim.</p>",
        "".join(chart),
        "<table><thead><tr><th>Extractor</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mean extraction ms</th><th>Process peak RSS MiB</th></tr></thead><tbody>",
    ]
    for name, result in backends.items():
        metrics, performance = result["exact_span"], result["performance"]
        rss = performance["process_peak_rss_bytes"]
        rss_text = f"{rss / 1024**2:.1f}" if rss is not None else "unavailable"
        body.append(
            f"<tr><td>{html.escape(name)}</td><td>{metrics['precision']:.4f}</td><td>{metrics['recall']:.4f}</td><td>{metrics['f1']:.4f}</td><td>{1000 * performance['mean_extraction_seconds']:.3f}</td><td>{rss_text}</td></tr>"
        )
    body.append("</tbody></table>")
    for name, result in backends.items():
        body.append(f"<h2>{html.escape(name)} errors</h2>")
        for title, key in (
            ("False positives", "common_false_positive_spans"),
            ("False negatives", "common_missed_spans"),
        ):
            body.append(f"<h3>{title}</h3><ul>")
            body.extend(
                f"<li>{html.escape(item['text'])}: {item['count']}</li>" for item in result[key]
            )
            body.append("</ul>")
    body.append(
        "<p>RSS includes interpreter/runtime/model and is a process-lifetime high-water. See JSON for timing scope and provenance.</p></body></html>"
    )
    return "\n".join(body)


def save_comparison(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    output.with_suffix(".html").write_text(comparison_html(report))
