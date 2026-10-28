"""Authored contract and failure tests, not evidence of model writing quality."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from resumeanalysis_legacy.cover_letters.service import AnalysisService
from resumeanalysis_legacy.integration.cover_letters import (
    DirectInferenceClient,
    KafkaInferenceClient,
    KafkaRequestReplyTransport,
    KafkaWorkerAdapter,
    ModelWorker,
)

from resumeanalysis.cover_letters.generator import (
    CoverLetterError,
)
from resumeanalysis.cover_letters.schema import AnalysisRequest, Resume
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


def resume(identity=1, **profile):
    return Resume(id=identity, profile=Profile(**profile))


def run(awaitable):
    return asyncio.run(awaitable)


def analysis(score, relevant=True):
    return VacancyAnalysis(
        relevant=relevant,
        relevance=score,
        matched_skills=[],
        missing_skills=[],
        unknown_skills=[],
        contradicted_skills=[],
        experience_gap_years=None,
        explanation="Existing relevance decision",
    )


class Relevance:
    def __init__(self, results):
        self.results, self.calls = iter(results), []

    def predict(self, profile, vacancy):
        self.calls.append((profile, vacancy))
        return next(self.results)


class Letter:
    def __init__(self, value="Authored test draft"):
        self.calls, self.value = [], value

    async def generate(self, vacancy, resume, **kwargs):
        self.calls.append((vacancy, resume, kwargs))
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def request():
    return AnalysisRequest(
        vacancy=Vacancy(description="Job"),
        resumes=[resume(10, skills=["Python"]), resume(20, skills=["SQL"])],
    )


def test_selects_best_relevant_resume_and_preserves_score():
    relevance, letters = Relevance([analysis(0.7), analysis(0.9)]), Letter()
    result = run(AnalysisService(relevance, letters).analyze(request()))
    assert result.resume_id == 20 and result.relevance == 0.9 and result.relevant
    assert result.cover_letter == "Authored test draft"
    assert len(relevance.calls) == 2 and len(letters.calls) == 1


def test_high_score_rejected_by_policy_does_not_override_positive_decision():
    result = run(
        AnalysisService(Relevance([analysis(0.99, False), analysis(0.7)]), Letter()).analyze(
            request()
        )
    )
    assert result.resume_id == 20 and result.relevance == 0.7


def test_ties_preserve_original_resume_order():
    result = run(
        AnalysisService(Relevance([analysis(0.9), analysis(0.9)]), Letter()).analyze(request())
    )
    assert result.resume_id == 10


def test_no_positive_decision_means_no_letter_call():
    letters = Letter()
    result = run(
        AnalysisService(Relevance([analysis(0.9, False), analysis(0.7, False)]), letters).analyze(
            request()
        )
    )
    assert not result.relevant and result.cover_letter is None and not letters.calls


def test_generation_failure_preserves_relevance_and_schema():
    result = run(
        AnalysisService(
            Relevance([analysis(0.9), analysis(0.8)]), Letter(CoverLetterError("failure"))
        ).analyze(request())
    )
    assert result.relevant and result.relevance == 0.9 and result.cover_letter is None
    assert VacancyAnalysis.model_validate(result.model_dump(), strict=True).resume_id == 10


def test_empty_resume_collection_never_calls_models():
    relevance, letter = Relevance([]), Letter()
    result = run(AnalysisService(relevance, letter).analyze(AnalysisRequest(vacancy=Vacancy())))
    assert not result.relevant and result.resume_id is None
    assert not relevance.calls and not letter.calls


def test_duplicate_resume_ids_are_rejected():
    with pytest.raises(ValidationError):
        AnalysisRequest(vacancy=Vacancy(), resumes=[resume(), resume()])


class Provider:
    def get_resumes(self):
        return [resume(skills=["Python"])]


def test_direct_and_kafka_adapters_return_the_same_contract():
    async def exercise():
        direct = DirectInferenceClient(
            AnalysisService(Relevance([analysis(0.9)]), Letter()), Provider()
        )
        worker = ModelWorker(AnalysisService(Relevance([analysis(0.9)]), Letter()))

        class Transport:
            async def request(self, topic, payload, **kwargs):
                return await worker.handle(payload)

        kafka = KafkaInferenceClient(Transport(), Provider())
        vacancy = Vacancy(description="Python")
        return await direct.process(vacancy), await kafka.process(vacancy)

    direct, kafka = run(exercise())
    assert direct.model_dump() == kafka.model_dump()


def test_worker_validation_error_never_echoes_private_input():
    worker = ModelWorker(AnalysisService(Relevance([]), Letter()))
    result = run(
        worker.handle({"request_id": "fixture", "payload": {"private": "candidate secret"}})
    )
    assert result == {"request_id": "fixture", "error": "analysis_failed"}


def test_kafka_acknowledges_only_after_published_response():
    events = []

    class Consumer:
        def __aiter__(self):
            async def rows():
                yield SimpleNamespace(
                    value=json.dumps(
                        {"request_id": "fixture", "payload": request().model_dump()}
                    ).encode()
                )

            return rows()

        async def commit(self):
            events.append("commit")

    class Producer:
        async def send_and_wait(self, topic, **kwargs):
            assert json.loads(kwargs["value"])["request_id"] == "fixture"
            events.append("publish")

    adapter = KafkaWorkerAdapter(
        ModelWorker(AnalysisService(Relevance([analysis(0.9), analysis(0.8)]), Letter())),
        Consumer(),
        Producer(),
        response_topic="fixed-replies",
    )
    run(adapter.run())
    assert events == ["publish", "commit"]


def test_kafka_request_reply_transport_correlates_responses():
    events = []

    class Producer:
        async def send_and_wait(self, topic, **kwargs):
            events.append(json.loads(kwargs["value"]))

    class Consumer:
        def __init__(self):
            self.values = iter([{"request_id": "old"}, {"request_id": "current", "analysis": {}}])

        async def getone(self):
            return SimpleNamespace(value=json.dumps(next(self.values)).encode())

    result = run(
        KafkaRequestReplyTransport(Producer(), Consumer()).request(
            "input-topic", {"request_id": "current", "payload": {}}, timeout=5
        )
    )
    assert result["request_id"] == "current"
    assert events == [{"request_id": "current", "payload": {}}]
