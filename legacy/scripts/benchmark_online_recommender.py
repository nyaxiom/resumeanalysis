"""Measure actual local E5/River CPU work on a disclosed authored preference scenario."""

import argparse
import json
import resource
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from resumeanalysis_legacy.experiments.online_evaluation import FeedbackRound, prequential_evaluate

from resumeanalysis.embeddings.encoder import TextEncoder
from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.recommendation.engine import RecommendationConfig, RecommendationEngine
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis.recommendation.learner import LearnerConfig, PreferenceLearner
from resumeanalysis.recommendation.repository import SQLiteFeedbackRepository
from resumeanalysis.recommendation.schema import StructuredProfile, StructuredVacancy, fingerprint


def summary(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": 1000 * statistics.mean(values),
        "p95_ms": 1000 * ordered[max(0, math_ceil(0.95 * len(ordered)) - 1)],
    }


def math_ceil(value):
    import math

    return math.ceil(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="evaluation/online_recommendations/scenario.json")
    parser.add_argument("--e5-path", default="artifacts/e5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scenario = loads(Path(args.scenario).read_text())
    profiles = [StructuredProfile.model_validate(p, strict=True) for p in scenario["profiles"]]
    vacancies = [StructuredVacancy.model_validate(v, strict=True) for v in scenario["vacancies"]]
    config = LearnerConfig.model_validate(scenario["learner_config"], strict=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    encoder = TextEncoder(args.e5_path, cache_dir=output / "embedding-cache")
    initialization = time.perf_counter() - started
    builders = {
        "river_compact": PreferenceFeatureBuilder(encoder),
        "river_projected": PreferenceFeatureBuilder(encoder, embedding_components=16),
    }
    feature_timings = {}
    for name, builder in builders.items():
        cold, warm = [], []
        for values in [cold, warm]:
            for profile in profiles:
                for vacancy in vacancies:
                    start = time.perf_counter()
                    builder.build(profile, vacancy)
                    values.append(time.perf_counter() - start)
        feature_timings[name] = {
            "first_pass": summary(cold),
            "cached_pass": summary(warm),
            "feature_count": len(builder.build(profiles[0], vacancies[0]).snapshot.values),
            "feature_identity": builder.identity,
        }
    # The enhanced pass reuses the compact pass's E5 cache; report that explicitly.
    rounds = []
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for phase in scenario["phases"]:
        for iteration in range(scenario["rounds_per_phase"]):
            for profile in profiles:
                interested = [v.vacancy_id in phase["likes"][profile.user_id] for v in vacancies]
                # Each round represents fresh postings from six authored templates.
                # Repeating IDs would violate the engine's feedback idempotency policy.
                fresh_vacancies = [
                    v.model_copy(update={"vacancy_id": f"{v.vacancy_id}:round-{len(rounds)}"})
                    for v in vacancies
                ]
                rounds.append(
                    FeedbackRound(
                        timestamp + timedelta(seconds=len(rounds)),
                        profile.user_id,
                        profile,
                        fresh_vacancies,
                        interested,
                        iteration % len(vacancies),
                        phase["name"],
                    )
                )
    quality = prequential_evaluate(rounds, builders, config=config)
    write_json(output / "prequential.json", quality)
    performance = {}
    for name, builder in builders.items():
        database = output / f"{name}.sqlite"
        repository = SQLiteFeedbackRepository(database)
        engine = RecommendationEngine(
            builder,
            repository,
            learner_config=config,
            config=RecommendationConfig(exploration_fraction=0.0),
        )
        recommend_times, update_times = [], []
        for i in range(20):
            vacancy = vacancies[i % len(vacancies)].model_copy(update={"vacancy_id": f"perf-{i}"})
            start = time.perf_counter()
            engine.recommend(profiles[0].user_id, profiles[0], [vacancy])
            recommend_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            engine.record_feedback(
                profiles[0].user_id, vacancy.vacancy_id, "like" if i % 2 else "dislike"
            )
            update_times.append(time.perf_counter() - start)
        repository.close()
        started = time.perf_counter()
        restored = SQLiteFeedbackRepository(database)
        again = RecommendationEngine(
            builder,
            restored,
            learner_config=config,
            config=RecommendationConfig(exploration_fraction=0.0),
        )
        again.recommend(profiles[0].user_id, profiles[0], [vacancies[0]])
        restore_time = time.perf_counter() - started
        restored.close()
        snapshot = builder.build(profiles[0], vacancies[0]).snapshot
        learner = PreferenceLearner(config)
        predict_times, learn_times = [], []
        for i in range(200):
            started = time.perf_counter()
            learner.predict_proba_one(snapshot.values)
            predict_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            learner.learn_one(snapshot.values, bool(i % 2))
            learn_times.append(time.perf_counter() - started)
        performance[name] = {
            "recommend_one_cached": summary(recommend_times),
            "record_feedback_transaction": summary(update_times),
            "predict_one": summary(predict_times),
            "learn_one_with_prequential_metrics": summary(learn_times),
            "restore_replay_and_recommend_ms": restore_time * 1000,
            "sqlite_bytes_after_20_events": database.stat().st_size,
            "projection_matrix_bytes": builder.projector.components_.nbytes
            if builder.projector
            else 0,
        }
    report = {
        "scenario": args.scenario,
        "scenario_sha256": fingerprint(scenario),
        "provenance": "Authored fictional profiles, jobs and changing interest labels; no historical users or real quality evidence",
        "encoder_identity": encoder.identity,
        "learner_config": config.model_dump(),
        "model_version": config.model_version,
        "model_initialization_seconds": initialization,
        "feature_timing": feature_timings,
        "performance": performance,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "vram_bytes": 0,
        "device": "cpu",
        "threads": 2,
        "torch_version": torch.__version__,
        "prequential": quality["methods"],
        "timing_note": "Projected first pass reuses E5 cache from compact first pass. Single execution; process RSS includes E5/Python and both learners, not isolated per-user model memory.",
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
