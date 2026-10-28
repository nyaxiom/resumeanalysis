from datetime import datetime, timedelta, timezone

import pytest
from resumeanalysis_legacy.experiments.online_evaluation import FeedbackRound, prequential_evaluate

from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis.recommendation.schema import StructuredProfile, StructuredVacancy
from resumeanalysis.schemas import Profile, Vacancy


def inputs():
    p = StructuredProfile(user_id="alice", facts=Profile(skills=["Python"]))
    jobs = [
        StructuredVacancy(
            vacancy_id=f"job-{i}",
            facts=Vacancy(
                title="Developer",
                description="Python work",
                required_skills=["Python"],
                remote=bool(i % 2),
            ),
        )
        for i in range(4)
    ]
    return p, jobs


def test_prequential_predictions_precede_learning_and_do_not_use_unobserved_pool_labels():
    p, jobs = inputs()
    builder = PreferenceFeatureBuilder(LexicalSmokeEncoder())
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def rounds(unobserved):
        return [
            FeedbackRound(
                start, "alice", p, jobs, [True, unobserved, unobserved, unobserved], 0, "initial"
            ),
            FeedbackRound(
                start + timedelta(seconds=1), "alice", p, jobs, [False] * 4, 1, "initial"
            ),
        ]

    left = prequential_evaluate(rounds(False), {"river": builder})
    right = prequential_evaluate(rounds(True), {"river": builder})
    assert left["trace"][0]["raw_preference_before_update"]["river"] == 0.5
    assert (
        left["trace"][1]["raw_preference_before_update"]
        == right["trace"][1]["raw_preference_before_update"]
    )
    assert left["trace"][1]["raw_preference_before_update"]["river"] > 0.5
    with pytest.raises(ValueError, match="strictly increasing"):
        prequential_evaluate(list(reversed(rounds(False))), {"river": builder})


def test_prequential_evaluation_uses_independent_user_models():
    p, jobs = inputs()
    bob = p.model_copy(update={"user_id": "bob"})
    now = datetime.now(timezone.utc)
    rounds = [
        FeedbackRound(now, "alice", p, jobs, [True] * 4, 0, "initial"),
        FeedbackRound(now + timedelta(seconds=1), "bob", bob, jobs, [False] * 4, 0, "initial"),
    ]
    result = prequential_evaluate(
        rounds, {"river": PreferenceFeatureBuilder(LexicalSmokeEncoder())}
    )
    assert result["trace"][1]["raw_preference_before_update"]["river"] == 0.5


def test_evaluation_does_not_silently_train_twice_on_duplicate_feedback():
    p, jobs = inputs()
    now = datetime.now(timezone.utc)
    rounds = [
        FeedbackRound(now, "alice", p, jobs, [True] * 4, 0, "initial"),
        FeedbackRound(now + timedelta(seconds=1), "alice", p, jobs, [False] * 4, 0, "initial"),
    ]
    with pytest.raises(ValueError, match="Duplicate user/vacancy"):
        prequential_evaluate(rounds, {"river": PreferenceFeatureBuilder(LexicalSmokeEncoder())})
