"""Direct and Kafka-facing adapters share the same model-independent request schema."""

import asyncio
import json
from typing import Protocol
from uuid import uuid4

from resumeanalysis.cover_letters.schema import AnalysisRequest, Resume
from resumeanalysis.cover_letters.service import AnalysisService
from resumeanalysis.jsonio import loads
from resumeanalysis.schemas import Vacancy, VacancyAnalysis


class ResumeProvider(Protocol):
    def get_resumes(self) -> list[Resume]: ...


class DirectInferenceClient:
    def __init__(
        self, service: AnalysisService, resumes: ResumeProvider, *, language="en", context=None
    ):
        self.service, self.resumes, self.language = service, resumes, language
        self.context = list(context or [])

    async def process(self, vacancy: Vacancy) -> VacancyAnalysis:
        return await self.service.analyze(
            AnalysisRequest(
                vacancy=vacancy,
                resumes=self.resumes.get_resumes(),
                language=self.language,
                candidate_context=self.context,
            )
        )


class RequestReplyTransport(Protocol):
    async def request(self, topic: str, payload: dict, *, timeout: float) -> dict: ...


class KafkaRequestReplyTransport:
    """Adapter for already-started async Kafka clients, with one request in flight.

    The caller configures a dedicated reply consumer/group and broker credentials.
    No Kafka package or remote service is required for Direct inference.
    """

    def __init__(self, producer, reply_consumer):
        self.producer, self.reply_consumer = producer, reply_consumer
        self._lock = asyncio.Lock()

    async def request(self, topic: str, payload: dict, *, timeout: float) -> dict:
        async with self._lock, asyncio.timeout(timeout):
            await self.producer.send_and_wait(
                topic,
                value=json.dumps(payload, ensure_ascii=False).encode(),
                key=payload["request_id"].encode(),
            )
            while True:
                message = await self.reply_consumer.getone()
                result = loads(message.value.decode("utf-8"))
                if isinstance(result, dict) and result.get("request_id") == payload["request_id"]:
                    return result


class KafkaInferenceClient:
    """Uses ApplyPilot's request/reply transport; no model objects enter transport code."""

    def __init__(
        self,
        transport: RequestReplyTransport,
        resumes: ResumeProvider,
        *,
        topic="vacancy-analysis",
        language="en",
        timeout=900.0,
        context=None,
    ):
        self.transport, self.resumes = transport, resumes
        self.topic, self.language, self.timeout = topic, language, timeout
        self.context = list(context or [])

    async def process(self, vacancy: Vacancy) -> VacancyAnalysis:
        request = AnalysisRequest(
            vacancy=vacancy,
            resumes=self.resumes.get_resumes(),
            language=self.language,
            candidate_context=self.context,
        )
        identifier = str(uuid4())
        response = await self.transport.request(
            self.topic,
            {"request_id": identifier, "payload": request.model_dump()},
            timeout=self.timeout,
        )
        if not isinstance(response, dict) or response.get("request_id") != identifier:
            raise RuntimeError("Model worker response correlation mismatch")
        if response.get("error"):
            raise RuntimeError("Model worker could not analyze the request")
        return VacancyAnalysis.model_validate(response.get("analysis"), strict=True)


class ModelWorker:
    def __init__(self, service: AnalysisService):
        self.service = service

    async def handle(self, envelope: dict) -> dict:
        identifier = envelope.get("request_id") if isinstance(envelope, dict) else None
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            raise ValueError("A bounded request_id is required")
        try:
            if set(envelope) != {"request_id", "payload"}:
                raise ValueError("Unexpected envelope fields")
            request = AnalysisRequest.model_validate(envelope["payload"], strict=True)
            analysis = await self.service.analyze(request)
            return {"request_id": identifier, "analysis": analysis.model_dump()}
        except Exception:
            # Do not put resume text, raw model output or exception details on the wire.
            return {"request_id": identifier, "error": "analysis_failed"}


class KafkaWorkerAdapter:
    """Run with an injected async Kafka consumer and producer (e.g. aiokafka).

    Use an exclusive consumer and disable auto-commit. Replies go to the configured topic, never an
    input-supplied destination. Acknowledge only after the reply is published.
    Retry/deduplication and broker authentication belong to ApplyPilot deployment.
    """

    def __init__(self, worker: ModelWorker, consumer, producer, *, response_topic: str):
        self.worker, self.consumer, self.producer = worker, consumer, producer
        self.response_topic = response_topic

    async def run(self):
        async for message in self.consumer:
            envelope = loads(message.value.decode("utf-8"))
            response = await self.worker.handle(envelope)
            await self.producer.send_and_wait(
                self.response_topic,
                value=json.dumps(response, ensure_ascii=False).encode(),
                key=response["request_id"].encode(),
            )
            await self.consumer.commit()
