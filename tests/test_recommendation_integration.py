"""Local integration, CLI persistence and chronological evaluation contracts."""

import json
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from resumeanalysis import cli
from resumeanalysis.embeddings.encoder import LexicalSmokeEncoder
from resumeanalysis.integration.recommendations import (
    LocalRecommendationClient,
    ResolvedVacancy,
    adapt_applypilot_description,
    adapt_applypilot_link,
)
from resumeanalysis.recommendation.adapters import from_legacy_profile
from resumeanalysis.recommendation.engine import RecommendationConfig, RecommendationEngine
from resumeanalysis.recommendation.evaluation import FeedbackRound, prequential_evaluate
from resumeanalysis.recommendation.features import PreferenceFeatureBuilder
from resumeanalysis.recommendation.repository import SQLiteFeedbackRepository
from resumeanalysis.recommendation.schema import (
    Preferences,
    SkillExperience,
    StructuredProfile,
    StructuredVacancy,
)
from resumeanalysis.schemas import Profile, Vacancy


def engine():
    return RecommendationEngine(
        PreferenceFeatureBuilder(LexicalSmokeEncoder()),
        SQLiteFeedbackRepository(":memory:"),
        config=RecommendationConfig(exploration_fraction=0),
    )


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


def test_existing_applypilot_link_shapes_are_adapted_without_assuming_vacancy_content():
    @dataclass
    class ParserLink:
        url: str
        source_domain: str
        vacancy_id: str

    class Resolver:
        def resolve(self, link):
            return ResolvedVacancy(facts=Vacancy(description="Fetched by the host application"))

    parser_link = ParserLink("https://hh.ru/vacancy/123", "hh.ru", "123")
    row = SimpleNamespace(
        url="https://example.test/vacancy/123", domain="example.test", vacancy_id="123"
    )
    first, second = [adapt_applypilot_link(link, Resolver()) for link in (parser_link, row)]
    assert first.vacancy_id == "hh.ru:123" and second.vacancy_id == "example.test:123"
    assert first.salary is None and first.facts.remote is None
    with pytest.raises(ValueError, match="disagree"):
        adapt_applypilot_link(ParserLink("https://other.test/123", "hh.ru", "123"), Resolver())


def test_unmerged_applypilot_description_preserves_unknown_salary_and_untrusted_enrichment():
    link = SimpleNamespace(url="https://hh.ru/vacancy/123", domain="hh.ru", vacancy_id="123")
    description = SimpleNamespace(
        title="Developer",
        description_plain="Python development",
        description_html="<p>Python development</p>",
        location="London",
        salary="up to 100k",
        enrichment={"remote": True, "required_skills": ["Invented skill"]},
    )
    result = adapt_applypilot_description(link, description)
    assert result.facts.description == "Python development"
    assert result.salary is None and result.facts.remote is None
    assert result.facts.required_skills == []
    description.description_plain = None
    with pytest.raises(ValueError, match="plain text"):
        adapt_applypilot_description(link, description)


def test_local_provider_adapter_and_cover_letter_bridge_preserve_task_boundaries():
    recommender = engine()
    p, jobs = inputs()
    p.skill_experience = [SkillExperience(skill="SQL", years=2)]
    p.preferences = Preferences(preferred_roles=["Team lead"])
    client = LocalRecommendationClient(
        recommender,
        SimpleNamespace(get_profile=lambda user: p),
        SimpleNamespace(list_vacancies=lambda user: jobs),
    )
    rec = client.recommend("alice", 2)[0]
    client.record_feedback("alice", rec.vacancy_id, "like", event_id="host-event")
    letter = client.cover_letter_input("alice", rec.vacancy_id, resume_id=3)
    assert letter.resumes[0].id == 3
    assert "SQL" in letter.resumes[0].profile.skills
    assert letter.resumes[0].profile.experience_years is None
    assert "Team lead" not in letter.resumes[0].profile.skills
    assert letter.candidate_context == ["Explicitly stated experience with SQL: 2 years."]
    with pytest.raises(ValueError, match="another user"):
        client.cover_letter_input("bob", rec.vacancy_id, resume_id=3)


def test_legacy_remote_constraints_are_migrated_explicitly_and_mlp_is_not_loaded():
    legacy = Profile(skills=["Python"], remote_only=True, allowed_locations=["London"])
    p = from_legacy_profile("alice", legacy)
    assert not p.facts.remote_only and p.preferences.hard.remote_only
    assert p.preferences.hard.locations == ["London"]
    assert legacy.remote_only  # Original object is preserved.


def test_local_ranking_and_feedback_do_not_open_network_connections(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Recommendation/feedback must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    recommender = engine()
    p, jobs = inputs()
    result = recommender.recommend("alice", p, jobs)
    recommender.record_feedback("alice", result[0].vacancy_id, "like")
    recommender.replay("alice")


def test_review_required_profiles_are_not_ranked_and_exploration_has_an_explicit_slot():
    recommender = engine()
    p, jobs = inputs()
    p.review_status = "needs_review"
    with pytest.raises(ValueError, match="Confirm"):
        recommender.recommend("alice", p, jobs)
    p.review_status = "confirmed"
    recommender.config = RecommendationConfig(exploration_fraction=0.5)
    result = recommender.recommend("alice", p, jobs, limit=2)
    assert len(result) == 2 and sum(r.exploration for r in result) == 1
    assert len({r.vacancy_id for r in result}) == 2


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


def invoke(monkeypatch, capsys, args):
    monkeypatch.setattr(sys, "argv", ["resumeanalysis", *args])
    assert cli.main() is None
    return json.loads(capsys.readouterr().out)


def test_cli_feedback_and_restart_use_saved_features_without_initializing_e5(
    monkeypatch, tmp_path, capsys
):
    from resumeanalysis.recommendation import cli as recommender_cli

    p, jobs = inputs()
    profile_file, vacancies_file, database = (
        tmp_path / "p.json",
        tmp_path / "v.json",
        tmp_path / "state.sqlite",
    )
    profile_file.write_text(p.model_dump_json())
    vacancies_file.write_text(json.dumps([v.model_dump() for v in jobs]))
    command = [
        "recommend",
        "--smoke-lexical",
        "--profile",
        str(profile_file),
        "--vacancies",
        str(vacancies_file),
        "--database",
        str(database),
        "--exploration-fraction",
        "0",
    ]
    before = invoke(monkeypatch, capsys, command)

    def forbidden(*args, **kwargs):
        raise AssertionError("Feedback must not need an encoder or model checkpoint")

    monkeypatch.setattr(recommender_cli, "TextEncoder", forbidden)
    feedback = invoke(
        monkeypatch,
        capsys,
        [
            "feedback",
            "--database",
            str(database),
            "--user-id",
            "alice",
            "--vacancy-id",
            before[0]["vacancy_id"],
            "--action",
            "dislike",
            "--event-id",
            "cli-event",
        ],
    )
    assert feedback["prediction_before_update"] == 0.5
    after = invoke(monkeypatch, capsys, command)
    assert before[0]["vacancy_id"] not in {r["vacancy_id"] for r in after}
    metrics = invoke(
        monkeypatch,
        capsys,
        ["preferences", "metrics", "--user-id", "alice", "--database", str(database)],
    )
    assert metrics["count"] == 1


def test_cli_invalid_profile_is_rejected_without_private_error_payload(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "bad.json"
    path.write_text('{"user_id":"alice","facts":{"experience_years":"private data"}}')
    monkeypatch.setattr(
        sys,
        "argv",
        ["resumeanalysis", "recommend", "--profile", str(path), "--vacancies", "unused"],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and "private data" not in capsys.readouterr().err
