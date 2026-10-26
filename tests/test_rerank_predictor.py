"""Evidence policy and public integration, using authored deterministic scores."""

import asyncio
import os

import numpy as np
import pytest

from resumeanalysis.inference.backends import MatchingProfile, create_predictor
from resumeanalysis.integration.applypilot import LocalInferenceClient
from resumeanalysis.rerank.predictor import RerankPredictor
from resumeanalysis.schemas import Vacancy, VacancyAnalysis


class Encoder:
    dimension = 2
    identity = "authored-two-dimensional-encoder-v1"

    def __init__(self):
        self.calls = []

    def encode(self, text, role="query"):
        self.calls.append((text, role))
        return np.array([1.0, 0.0], dtype=np.float32)


class Scorer:
    identity = "authored-constant-logit-v1"

    def __init__(self, score=10.0):
        self.score = score
        self.calls = []

    def score_pairs(self, pairs):
        self.calls.append(pairs)
        return [self.score for _ in pairs]


def predictor(score=10.0, **kwargs):
    return RerankPredictor(encoder=Encoder(), reranker=Scorer(score), **kwargs)


def test_factory_and_applypilot_preserve_public_contract():
    backend = create_predictor("rerank", encoder=Encoder(), reranker=Scorer())

    class Profiles:
        def get_profile(self):
            return MatchingProfile(skills=["Python"], preferred_roles=["Developer"])

    async def process_with_deadline():
        return await asyncio.wait_for(
            LocalInferenceClient(backend, Profiles()).process(Vacancy(required_skills=["Python"])),
            timeout=5,
        )

    result = asyncio.run(process_with_deadline())
    assert isinstance(result, VacancyAnalysis)
    assert result.relevant
    assert result.matched_skills == ["python"]
    assert "uncalibrated" in result.explanation


@pytest.mark.parametrize("profile", [{}, {"denied_skills": ["Python"]}])
def test_empty_positive_evidence_cannot_receive_a_positive_model_score(profile):
    backend = predictor()
    result = backend.analyze_pair(profile, {"required_skills": ["Python"]})
    assert result.rerank_score is None
    assert not result.analysis.relevant
    assert result.analysis.relevance == 0
    assert not backend.reranker.calls


def test_title_never_supplies_skill_or_requirement_evidence():
    backend = predictor()
    result = backend.analyze_pair({"skills": ["Python"]}, {"title": "Senior Python developer"})
    assert result.evidence_status == "needs_review"
    assert result.rerank_score is None
    assert result.analysis.matched_skills == []


@pytest.mark.parametrize(
    "profile,vacancy",
    [
        ({"skills": ["Java"], "denied_skills": ["Python"]}, {"required_skills": ["Python"]}),
        ({"skills": ["Python"], "denied_skills": ["Python"]}, {"required_skills": ["Python"]}),
        ({"skills": ["Python"], "experience_years": 1.0}, {"min_experience_years": 3.0}),
        ({"skills": ["Python"], "seniority": "junior"}, {"seniority": "senior"}),
        ({"skills": ["Python"], "remote_only": True}, {"description": "Python", "remote": False}),
        (
            {"skills": ["Python"], "allowed_locations": ["London"]},
            {"description": "Python", "location": "Paris"},
        ),
    ],
)
def test_semantic_score_never_overrides_explicit_hard_conflicts(profile, vacancy):
    backend = predictor(score=1000)
    result = backend.analyze_pair(profile, vacancy)
    assert result.evidence_status == "blocked"
    assert not result.analysis.relevant
    assert backend.rank(profile, [vacancy]) == []


@pytest.mark.parametrize(
    "profile,vacancy",
    [
        ({"skills": ["Python"]}, {"required_skills": ["Java"]}),
        ({"skills": ["Python"]}, {"min_experience_years": 3.0}),
        ({"skills": ["Python"]}, {"seniority": "senior"}),
        ({"skills": ["Python"], "remote_only": True}, {"description": "Python"}),
    ],
)
def test_unknown_requirements_need_review_without_claiming_absence(profile, vacancy):
    result = predictor().analyze_pair(profile, vacancy)
    assert result.evidence_status == "needs_review"
    assert not result.analysis.relevant
    assert not result.analysis.contradicted_skills


def test_batched_ranking_stable_ties_and_hard_prefilter_before_top_k():
    backend = predictor()
    profile = {"skills": ["Python"], "remote_only": True}
    vacancies = [
        {"title": "Wrong preference", "required_skills": ["Python"], "remote": False},
        {"title": "First allowed", "required_skills": ["Python"], "remote": True},
        {"title": "Second allowed", "required_skills": ["Python"], "remote": True},
    ]
    assert [r.index for r in backend.rank(profile, vacancies, preselect_k=1)] == [1]
    results = backend.rank(profile, vacancies)
    assert [r.index for r in results] == [1, 2]
    assert len(backend.reranker.calls[-1]) == 2
    assert backend.last_rank_diagnostics["preference_filtered_count"] == 1
    assert backend.last_rank_diagnostics["returned_count"] == 2


def test_confirmed_recommendations_precede_high_scoring_unknowns():
    backend = predictor()
    result = backend.rank(
        {"skills": ["Python"]},
        [{"required_skills": ["Java"]}, {"required_skills": ["Python"]}],
    )
    assert [r.index for r in result] == [1, 0]
    assert [r.analysis.relevant for r in result] == [True, False]


def test_empty_pool_and_top_zero_do_not_infer():
    backend = predictor()
    assert backend.rank({}, []) == []
    assert backend.rank({}, [{"description": "Python"}], top_k=0) == []
    assert backend.reranker.calls == []
    assert backend.retriever.encoder.calls == []


def test_empty_profile_batch_skips_both_models():
    backend = predictor()
    assert backend.rank({}, [{"description": "Python"}]) == []
    assert backend.last_rank_diagnostics["reason"] == "Positive candidate evidence is empty"
    assert not backend.retriever.encoder.calls and not backend.reranker.calls


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_preselection_is_rejected(value):
    with pytest.raises(ValueError, match="preselect_k"):
        predictor().rank({}, [], preselect_k=value)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_model_scores_are_rejected(score):
    with pytest.raises(ValueError, match="non-finite"):
        predictor(score).predict({"skills": ["Python"]}, {"required_skills": ["Python"]})


def test_raw_threshold_does_not_use_saturated_sigmoid_for_decisions():
    result = predictor(score=999, decision_logit_threshold=1000).predict(
        {"skills": ["Python"]}, {"required_skills": ["Python"]}
    )
    assert result.relevance == 1
    assert not result.relevant


def test_language_does_not_change_structured_confirmed_evidence():
    for text in (
        "I use Python and SQL.",
        "Я использую Python и SQL.",
        "Backend: Python, опыт SQL.",
    ):
        result = predictor().predict({"resume_text": text}, {"required_skills": ["Python", "SQL"]})
        assert result.relevant
        assert result.matched_skills == ["python", "sql"]


def test_unknown_factory_backend_is_explicit():
    with pytest.raises(ValueError, match="mlp, qwen or rerank"):
        create_predictor("unknown")


@pytest.mark.parametrize("change", ["scorer_identity", "scorer", "extractor", "threshold"])
def test_pipeline_identity_changes_require_a_new_predictor(change):
    backend = predictor()
    if change == "scorer_identity":
        backend.reranker.identity = "different-revision"
    elif change == "scorer":
        backend.reranker = Scorer()
    elif change == "extractor":
        from resumeanalysis.extraction.registry import create_extractor

        backend.extractor = create_extractor("v2")
    else:
        backend.decision_logit_threshold = 0.1
    with pytest.raises(ValueError, match="identity changed"):
        backend.predict({"skills": ["Python"]}, {"required_skills": ["Python"]})


def test_mutation_inside_scorer_callback_is_rejected():
    class MutatingScorer(Scorer):
        def score_pairs(self, pairs):
            self.identity = "mutated-during-inference"
            return super().score_pairs(pairs)

    backend = RerankPredictor(encoder=Encoder(), reranker=MutatingScorer())
    with pytest.raises(ValueError, match="identity changed"):
        backend.predict({"skills": ["Python"]}, {"required_skills": ["Python"]})


def test_result_metadata_cannot_change_backend_or_sibling_diagnostics():
    backend = predictor()
    results = backend.rank({"skills": ["Python"]}, [{"required_skills": ["Python"]}] * 2)
    original = backend._components.copy()
    results[0].diagnostics["identities"]["reranker"] = "changed-by-caller"
    results[0].diagnostics["evidence"]["required_skills"].clear()
    assert backend._components == original
    assert results[1].diagnostics["identities"] == original
    assert results[1].diagnostics["evidence"]["required_skills"] == ["python"]


@pytest.mark.skipif(
    os.environ.get("RESUMEANALYSIS_RUN_RERANK_TESTS") != "1"
    or os.environ.get("RESUMEANALYSIS_RUN_E5_TESTS") != "1",
    reason="Requires explicitly downloaded E5 and reranker assets",
)
def test_actual_offline_backend_pair_and_batch(monkeypatch):
    import socket

    from resumeanalysis.embeddings.encoder import TextEncoder
    from resumeanalysis.rerank.config import RerankConfig

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Offline inference attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    backend = RerankPredictor(
        encoder=TextEncoder("artifacts/e5"),
        config=RerankConfig(model_path="artifacts/reranker"),
    )
    profile = {
        "resume_text": "I build Python services. Работаю с Python.",
        "skills": ["Python"],
        "remote_only": True,
    }
    vacancy = {"description": "Required: Python.", "remote": True}
    pair = backend.analyze_pair(profile, vacancy)
    assert pair.rerank_score is not None
    assert pair.evidence_status == "eligible"
    assert pair.analysis.matched_skills == ["python"]
    results = backend.rank(profile, [vacancy, {**vacancy, "remote": False}], preselect_k=2)
    assert len(results) == 1 and results[0].index == 0
    assert results[0].rerank_score == pytest.approx(pair.rerank_score, abs=1e-5)
