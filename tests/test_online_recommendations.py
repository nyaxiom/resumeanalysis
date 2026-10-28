"""Authored synthetic behavior tests; no claims about real user recommendation quality."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.recommendation.engine import RecommendationConfig, RecommendationEngine
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis.recommendation.learner import LearnerConfig, PreferenceLearner
from resumeanalysis.recommendation.repository import SQLiteFeedbackRepository
from resumeanalysis.recommendation.schema import (
    HardConstraints,
    Preferences,
    Salary,
    StructuredProfile,
    StructuredVacancy,
)
from resumeanalysis.schemas import Profile, Vacancy


def profile(user="alice", **kwargs):
    return StructuredProfile(user_id=user, facts=Profile(skills=["Python", "SQL"]), **kwargs)


def job(identity="python", **kwargs):
    return StructuredVacancy(
        vacancy_id=identity,
        facts=Vacancy(
            title="Python developer",
            description="Required: Python and SQL.",
            required_skills=["Python", "SQL"],
            **kwargs,
        ),
    )


def engine(path=":memory:", *, projection=0, repository=None, window=512):
    return RecommendationEngine(
        PreferenceFeatureBuilder(LexicalSmokeEncoder(), embedding_components=projection),
        repository or SQLiteFeedbackRepository(path),
        learner_config=LearnerConfig(history_window=window),
        config=RecommendationConfig(exploration_fraction=0.0),
    )


def test_cold_start_needs_no_personalized_weights_and_does_not_fit_on_recommendation():
    recommender = engine()
    jobs = [job(), StructuredVacancy(vacancy_id="unknown", facts=Vacancy(title="Other"))]
    first = recommender.recommend("alice", profile(), jobs)
    second = recommender.recommend("alice", profile(), jobs)
    assert first[0].vacancy_id == "python"
    assert all(row.preference_score == 0.5 for row in first + second)
    assert first[0].qualification.matched_skills == ["python", "sql"]
    assert recommender.online_metrics("alice")["count"] == 0


def test_explicit_feedback_changes_interest_without_changing_qualification():
    recommender = engine()
    before = recommender.recommend("alice", profile(), [job()])[0]
    feedback = recommender.record_feedback("alice", "python", "like", event_id="event-1")
    after = recommender.recommend("alice", profile(), [job()])[0]
    assert feedback.prediction_before_update == before.preference_score == 0.5
    assert after.preference_score > before.preference_score
    assert after.qualification == before.qualification
    metrics = recommender.online_metrics("alice")
    assert metrics["count"] == 1 and metrics["brier"] == 0.25


def test_users_are_isolated_even_with_same_vacancy_and_event_identifiers():
    recommender = engine()
    for user in ["alice", "bob"]:
        recommender.recommend(user, profile(user), [job()])
    recommender.record_feedback("alice", "python", "like", event_id="same-id")
    assert recommender.recommend("bob", profile("bob"), [job()])[0].preference_score == 0.5
    recommender.record_feedback("bob", "python", "dislike", event_id="same-id")
    assert recommender.recommend("bob", profile("bob"), [job()]) == []
    assert recommender.recommend("alice", profile(), [job()])
    with pytest.raises(ValueError, match="different user"):
        recommender.recommend("bob", profile(), [job()])


def test_restart_restores_scores_and_recorded_prequential_metrics(tmp_path):
    path = tmp_path / "feedback.sqlite"
    first = engine(path)
    first.recommend("alice", profile(), [job()])
    first.record_feedback("alice", "python", "like")
    score = first.recommend("alice", profile(), [job()])[0].preference_score
    metrics = first.online_metrics("alice")
    first.repository.close()
    second = engine(path)
    assert second.recommend("alice", profile(), [job()])[0].preference_score == score
    assert second.online_metrics("alice") == metrics
    second.replay("alice")
    assert second.recommend("alice", profile(), [job()])[0].preference_score == score


def test_duplicate_retry_is_idempotent_and_conflicting_label_requires_correction():
    recommender = engine()
    recommender.recommend("alice", profile(), [job()])
    original = recommender.record_feedback("alice", "python", "dislike", event_id="one")
    assert recommender.record_feedback("alice", "python", "dislike", event_id="one") == original
    assert recommender.record_feedback("alice", "python", "dislike") == original
    with pytest.raises(ValueError, match="different payload"):
        recommender.record_feedback("alice", "python", "like", event_id="one")
    with pytest.raises(ValueError, match="corrects_event_id"):
        recommender.record_feedback("alice", "python", "like")
    correction = recommender.record_feedback(
        "alice", "python", "like", event_id="two", corrects_event_id="one"
    )
    assert correction.feature_snapshot == original.feature_snapshot
    assert recommender.online_metrics("alice")["count"] == 1
    assert recommender.recommend("alice", profile(), [job()])[0].preference_score > 0.5
    with recommender.repository.transaction("alice") as session:
        assert len(session.events()) == 2
    with pytest.raises(ValueError, match="active feedback"):
        recommender.record_feedback("alice", "python", "dislike", corrects_event_id="one")


def test_feedback_cannot_use_another_users_snapshot_or_an_unseen_vacancy():
    recommender = engine()
    rec = recommender.recommend("alice", profile(), [job()])[0]
    with pytest.raises(ValueError, match="previously shown"):
        recommender.record_feedback(
            "bob", "python", "like", recommendation_id=rec.recommendation_id
        )
    with pytest.raises(ValueError, match="previously shown"):
        recommender.record_feedback("alice", "unseen", "like")
    for action in ["view", "ignore", "apply", "interview"]:
        with pytest.raises(ValueError, match="explicit"):
            recommender.record_feedback("alice", "python", action)


def test_reset_is_explicit_and_replay_can_restore_earlier_history():
    recommender = engine()
    recommender.recommend("alice", profile(), [job("one"), job("two")])
    recommender.record_feedback("alice", "one", "like")
    recommender.record_feedback("alice", "two", "dislike")
    score = recommender.recommend("alice", profile(), [job("one")])[0].preference_score
    recommender.reset("alice")
    assert recommender.recommend("alice", profile(), [job("one")])[0].preference_score == 0.5
    assert recommender.recommend("alice", profile(), [job("two")]) == []
    recommender.replay("alice", include_before_reset=True)
    assert recommender.recommend("alice", profile(), [job("one")])[0].preference_score == score


def test_recent_window_can_adapt_to_a_change_in_preferences(tmp_path):
    recommender = engine(tmp_path / "drift.sqlite", window=8)
    for index in range(24):
        identity = f"job-{index}"
        recommender.recommend("alice", profile(), [job(identity)])
        recommender.record_feedback("alice", identity, "like" if index < 12 else "dislike")
        if index == 11:
            old_score = recommender.recommend("alice", profile(), [job("probe")])[
                0
            ].preference_score
    new_score = recommender.recommend("alice", profile(), [job("probe")])[0].preference_score
    assert new_score < 0.5 < old_score
    restored = engine(tmp_path / "drift.sqlite", window=8)
    assert restored.recommend("alice", profile(), [job("probe")])[0].preference_score == new_score


def test_missing_attributes_are_unknown_not_hard_rejections():
    recommender = engine()
    empty = StructuredProfile(user_id="alice")
    built = recommender.builder.build(empty, job(min_experience_years=2))
    assert built.qualification.score is None
    assert built.qualification.experience_gap_years is None
    assert built.snapshot.values["experience_match_known"] == 0
    assert built.snapshot.values["salary_match_known"] == 0
    assert built.qualification.unknown_skills == ["python", "sql"]
    assert built.constraints.status == "eligible"


def test_hard_constraints_cannot_be_overridden_by_interest_or_exploration():
    recommender = engine()
    recommender.config = RecommendationConfig(exploration_fraction=0.5)
    p = profile(
        preferences=Preferences(hard=HardConstraints(remote_only=True, locations=["London"]))
    )
    jobs = [
        job("blocked", remote=False, location="London"),
        job("unknown", remote=None),
        job("allowed", remote=True, location="London"),
    ]
    result = recommender.recommend("alice", p, jobs)
    assert {r.vacancy_id for r in result} == {"unknown", "allowed"}
    assert next(r for r in result if r.vacancy_id == "unknown").hard_constraints.status == "unknown"


@pytest.mark.parametrize(
    "offered,status",
    [
        (None, "unknown"),
        (Salary(minimum=10, maximum=20, currency="EUR", period="year", gross=True), "unknown"),
        (Salary(minimum=10, maximum=20, currency="USD", period="month", gross=True), "unknown"),
        (Salary(minimum=10, maximum=20, currency="USD", period="year", gross=True), "blocked"),
        (Salary(minimum=100, maximum=200, currency="USD", period="year", gross=True), "eligible"),
    ],
)
def test_salary_units_are_never_guessed(offered, status):
    recommender = engine()
    p = profile(
        preferences=Preferences(
            hard=HardConstraints(
                salary=Salary(minimum=100, currency="USD", period="year", gross=True)
            )
        )
    )
    vacancy = job().model_copy(update={"salary": offered})
    assert recommender.builder.build(p, vacancy).constraints.status == status


def test_projection_is_small_deterministic_and_distinctly_versioned():
    small, projected, another = engine(), engine(projection=16), engine(projection=16)
    plain = small.builder.build(profile(), job()).snapshot
    enhanced = projected.builder.build(profile(), job()).snapshot
    assert len(enhanced.values) == len(plain.values) + 17
    assert len(enhanced.values) < 80
    assert enhanced.identity != plain.identity
    assert enhanced == another.builder.build(profile(), job()).snapshot


def test_config_changes_and_unversioned_profile_edits_fail_explicitly(tmp_path):
    path = tmp_path / "features.sqlite"
    recommender = engine(path)
    recommender.recommend("alice", profile(), [job()])
    with pytest.raises(ValueError, match="configuration changed"):
        engine(path, projection=16).recommend("alice", profile(), [job()])
    p = profile()
    p.facts.skills.append("Docker")
    with pytest.raises(ValueError, match="without increasing"):
        recommender.recommend("alice", p, [job()])
    p.version += 1
    assert recommender.recommend("alice", p, [job()])


def test_failure_before_commit_rolls_back_event_and_does_not_publish_learned_cache():
    backing = SQLiteFeedbackRepository(":memory:")

    class FailingRepository:
        fail = False

        @contextmanager
        def transaction(self, user_id):
            with backing.transaction(user_id) as session:
                yield session
                if self.fail:
                    raise RuntimeError("Injected commit failure")

    repository = FailingRepository()
    recommender = engine(repository=repository)
    recommender.recommend("alice", profile(), [job()])
    repository.fail = True
    with pytest.raises(RuntimeError, match="commit failure"):
        recommender.record_feedback("alice", "python", "like", event_id="retry")
    repository.fail = False
    assert recommender.online_metrics("alice")["count"] == 0
    assert recommender.recommend("alice", profile(), [job()])[0].preference_score == 0.5
    recommender.record_feedback("alice", "python", "like", event_id="retry")
    assert recommender.online_metrics("alice")["count"] == 1


def test_independent_app_instances_do_not_lose_concurrent_feedback(tmp_path):
    path = tmp_path / "shared.sqlite"
    first, second = engine(path), engine(path)
    first.recommend("alice", profile(), [job("one"), job("two")])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(client.record_feedback, "alice", identity, "like")
            for client, identity in [(first, "one"), (second, "two")]
        ]
        for future in futures:
            future.result(timeout=10)
    assert first.online_metrics("alice")["count"] == second.online_metrics("alice")["count"] == 2
    left = first.recommend("alice", profile(), [job("probe")])[0]
    right = second.recommend("alice", profile(), [job("probe")])[0]
    assert left.preference_score == right.preference_score


def test_river_learns_before_next_prediction_and_predictions_do_not_fit_scaler():
    learner = PreferenceLearner()
    x = {"remote_match": 1.0, "salary_match": 0.0}
    for _ in range(5):
        assert learner.predict_proba_one(x) == 0.5
    assert not learner.model["StandardScaler"].counts
    assert learner.learn_one(x, True) == 0.5
    assert learner.predict_proba_one(x) > 0.5


def test_feedback_can_discover_remote_preference_without_an_explicit_initial_preference():
    recommender = engine()
    for i in range(24):
        vacancy = job(f"learn-{i}", remote=bool(i % 2))
        recommender.recommend("alice", profile(), [vacancy])
        recommender.record_feedback("alice", vacancy.vacancy_id, "like" if i % 2 else "dislike")
    result = recommender.recommend(
        "alice", profile(), [job("probe-remote", remote=True), job("probe-office", remote=False)]
    )
    assert result[0].vacancy_id == "probe-remote"
    assert result[0].preference_score > result[1].preference_score
