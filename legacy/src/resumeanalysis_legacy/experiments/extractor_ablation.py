"""Four controlled downstream extraction conditions on the same original pairs.

This experiment changes observations, not labels. Source fit labels are retained
only as a diagnostic reference: they are not independently validated for the
description-only intervention and do not establish extraction or hiring accuracy.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from resumeanalysis.extraction.registry import create_extractor
from resumeanalysis_legacy.data.adapters import digest
from resumeanalysis_legacy.data.dataset import SPLITS
from resumeanalysis_legacy.data.download import sha256_file
from resumeanalysis_legacy.experiments.comparison import decision_metrics
from resumeanalysis_legacy.experiments.extractor_evaluation import extraction_details
from resumeanalysis_legacy.experiments.partial_profiles import load_splits
from resumeanalysis_legacy.inference.predictor import Predictor

CONDITIONS = (
    ("legacy_description_only", "legacy", False),
    ("v2_description_only", "v2", False),
    ("legacy_explicit_requirements", "legacy", True),
    ("v2_explicit_requirements", "v2", True),
)


def _peak_rss():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


def _checkpoint(path, expected_data_hashes):
    metadata = torch.load(path, map_location="cpu", weights_only=True)
    hashes = metadata.get("data_sha256")
    if not isinstance(hashes, dict) or any(
        hashes.get(split) != expected for split, expected in expected_data_hashes.items()
    ):
        raise ValueError(
            "Ablation checkpoints must use the same original train/validation/test partitions"
        )
    return {
        "sha256": sha256_file(path),
        "format_version": metadata.get("format_version"),
        "pipeline_version": metadata.get("pipeline_version"),
        "extractor_identity": metadata.get("extractor_identity"),
        "training_data_sha256": {split: hashes[split] for split in SPLITS},
        "training_config": metadata.get("training_config"),
        "best_epoch": metadata.get("best_epoch"),
    }


def run_ablation(
    data,
    legacy_checkpoint,
    v2_checkpoint,
    encoder,
    output,
    lexicon_path=None,
    limit=None,
):
    """Verify pipelines, preserve source pairs, evaluate all four explicit conditions.

    Both checkpoints must have been trained on the supplied original partitions.
    No fitting, threshold selection, repartitioning or checkpoint writing occurs.
    Output must be a fresh JSON path outside the source dataset directory.
    """
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh ablation report path")
    if output.is_relative_to(data):
        raise ValueError("Ablation reports must remain outside the original dataset directory")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    splits = load_splits(data)  # Includes canonical validation and all-partition leakage checks.
    source_hashes = {split: sha256_file(data / (split + ".jsonl")) for split in SPLITS}
    records = splits["test"][:limit]
    checkpoints = {
        "legacy": _checkpoint(legacy_checkpoint, source_hashes),
        "v2": _checkpoint(v2_checkpoint, source_hashes),
    }
    models, initialization = {}, {}
    for name, path in (("legacy", legacy_checkpoint), ("v2", v2_checkpoint)):
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        extractor = create_extractor(name, lexicon_path=lexicon_path if name == "v2" else None)
        # There is deliberately no option to force an incompatible old checkpoint.
        models[name] = Predictor(path, encoder, extractor=extractor)
        checkpoints[name]["runtime_extractor_identity"] = models[name].builder.extractor_identity
        initialization[name] = {
            "wall_seconds": time.perf_counter() - started_wall,
            "process_cpu_seconds": time.process_time() - started_cpu,
        }
    before_records = [digest(record.model_dump()) for record in records]
    conditions = {}
    experiment_wall, experiment_cpu = time.perf_counter(), time.process_time()
    for condition, name, explicit in CONDITIONS:
        model = models[name]
        extractor, outcomes, predictions, latencies, cpu_times = (
            model.builder.extractor,
            [],
            [],
            [],
            [],
        )
        for index, record in enumerate(records):
            vacancy = record.vacancy.model_copy(deep=True)
            removed = {}
            if not explicit:
                removed = {
                    "/vacancy/required_skills": list(vacancy.required_skills),
                    "/vacancy/optional_skills": list(vacancy.optional_skills),
                }
                vacancy.required_skills = []
                vacancy.optional_skills = []
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            analysis = model.predict(record.profile, vacancy)
            latency, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
            latencies.append(latency)
            cpu_times.append(cpu)
            predictions.append(analysis)
            # Reporting extraction is separate from timed inference, avoiding a
            # second embedding/model pass and making instrumentation cost visible.
            metadata_start = time.perf_counter()
            extracted = extractor.extract(vacancy.description, vacancy=True)
            details = extraction_details(extracted, vacancy.description)
            required = extractor.normalizer.normalize_many(
                vacancy.required_skills, vacancy.description
            )
            optional = extractor.normalizer.normalize_many(
                vacancy.optional_skills, vacancy.description
            )
            effective_required = extracted.required | required
            effective_optional = (extracted.optional | optional) - effective_required
            outcomes.append(
                {
                    "source_record_sha256": before_records[index],
                    "candidate_id": record.candidate_id,
                    "vacancy_id": record.vacancy_id,
                    "source_split": "test",
                    "source_provenance": record.provenance,
                    "source_label_metadata": record.label,
                    "input_sha256": digest(
                        {"profile": record.profile.model_dump(), "vacancy": vacancy.model_dump()}
                    ),
                    "intervention": {
                        "type": "original_supplied_structured_lists"
                        if explicit
                        else "remove_structured_vacancy_skill_lists",
                        "removed_values": removed,
                        "unchanged": "Candidate profile, vacancy title/description, other vacancy fields, IDs and source partition",
                        "label_status": "Original label is not independently revalidated for this intervention",
                    },
                    "supplied_required_skills": list(vacancy.required_skills),
                    "supplied_optional_skills": list(vacancy.optional_skills),
                    "vacancy_extraction": details,
                    "normalized_structured_required_skills": sorted(required),
                    "normalized_structured_optional_skills": sorted(optional),
                    "effective_independent_required_skills": sorted(effective_required),
                    "effective_independent_optional_skills": sorted(effective_optional),
                    "analysis": analysis.model_dump(),
                    "false_positive_vs_source_label": analysis.relevant and record.label == 0,
                    "false_negative_vs_source_label": not analysis.relevant and record.label == 1,
                    "skill_span_or_requirement_accuracy": None,
                    "inference_wall_seconds": latency,
                    "inference_process_cpu_seconds": cpu,
                    "reporting_extraction_wall_seconds": time.perf_counter() - metadata_start,
                }
            )
        wall, cpu = sum(latencies), sum(cpu_times)
        conditions[condition] = {
            "extractor": name,
            "structured_requirement_condition": "original_supplied"
            if explicit
            else "description_only",
            "checkpoint_sha256": checkpoints[name]["sha256"],
            "extractor_identity": checkpoints[name]["runtime_extractor_identity"],
            "count": len(records),
            "source_record_sha256": before_records,
            "source_label_diagnostic_metrics": decision_metrics(
                [r.label for r in records], predictions
            ),
            "label_warning": "Source fit labels may be synthetic/heuristic and were not revalidated after removing information. These intervention diagnostics do not establish independent matching, extraction or hiring accuracy.",
            "performance": {
                "inference_wall_seconds": wall,
                "inference_process_cpu_seconds": cpu,
                "average_process_cpu_percent": 100 * cpu / wall if wall else None,
                "mean_inference_seconds": statistics.mean(latencies),
                "median_inference_seconds": statistics.median(latencies),
                "process_peak_rss_bytes": _peak_rss(),
                "scope": "Timed predict calls only; excludes separately reported model initialization and reporting-only extraction. No warmup. Shared process with both CPU MLPs and encoder loaded; lifetime RSS high-water is not attributable per condition. E5 caches and fixed condition order can influence latency; this is not an isolated speed ranking.",
            },
            "records": outcomes,
        }
    if [digest(record.model_dump()) for record in records] != before_records:
        raise ValueError("Ablation mutated original source records")
    if any(
        sha256_file(data / (split + ".jsonl")) != expected
        for split, expected in source_hashes.items()
    ):
        raise ValueError("Ablation source partitions changed during evaluation")
    for name, path in (("legacy", legacy_checkpoint), ("v2", v2_checkpoint)):
        if sha256_file(path) != checkpoints[name]["sha256"]:
            raise ValueError("Ablation checkpoint changed during evaluation")
    report = {
        "format_version": 1,
        "source_split": "test",
        "source_data_sha256": source_hashes,
        "source_partition_counts": {split: len(rows) for split, rows in splits.items()},
        "source_record_sha256": before_records,
        "count": len(records),
        "limit": limit,
        "condition_order": [name for name, _, _ in CONDITIONS],
        "lexicon_sha256": sha256_file(lexicon_path) if lexicon_path else None,
        "partition_check": "Passed candidate/vacancy ID and normalized-content checks on all original partitions",
        "checkpoints": checkpoints,
        "model_initialization": initialization,
        "conditions": conditions,
        "experiment_wall_seconds": time.perf_counter() - experiment_wall,
        "experiment_process_cpu_seconds": time.process_time() - experiment_cpu,
        "process_peak_rss_bytes": _peak_rss(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "encoder_identity": encoder.identity,
        },
        "claim": "Identical ordered source pairs across four controlled conditions. No extraction-quality or real-world hiring-accuracy claim; source-label intervention metrics are diagnostic only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier results even if a concurrent run races.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report
