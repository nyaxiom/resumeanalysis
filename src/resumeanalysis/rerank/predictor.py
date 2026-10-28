"""Opt-in retrieval, joint document scoring and independently enforced evidence."""

import hashlib
import json
import math
import threading
from copy import deepcopy
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from resumeanalysis.embeddings.encoder import TextEncoder
from resumeanalysis.extraction.registry import create_extractor, extractor_identity
from resumeanalysis.inference.backends import MatchingProfile
from resumeanalysis.postprocessing.rules import Postprocessor
from resumeanalysis.rerank.config import RerankConfig
from resumeanalysis.rerank.evidence import (
    SERIALIZATION_VERSION,
    build_evidence,
    cross_encoder_texts,
    has_candidate_evidence,
    preference_conflicts,
)
from resumeanalysis.rerank.model import CrossEncoder
from resumeanalysis.rerank.retrieval import ExactRetriever
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


def sigmoid(value: float) -> float:
    """Bound the compatibility field without claiming probability calibration."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class RankedVacancy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    index: int = Field(ge=0)
    vacancy: Vacancy
    analysis: VacancyAnalysis
    retrieval_score: float | None = Field(default=None, ge=-1, le=1)
    rerank_score: float | None = None
    evidence_status: Literal["eligible", "needs_review", "blocked"]
    diagnostics: dict = Field(default_factory=dict)
    backend_identity: str


class RerankPredictor:
    """Serializes each request; model weights are CPU-only and never trained here.

    Injected scorers expose ``identity`` and ``score_pairs(pairs)`` returning raw
    finite logits. Extractors retain the existing extract/normalizer interface;
    custom extractors additionally provide a stable, versioned identity.
    """

    def __init__(
        self,
        encoder=None,
        reranker=None,
        config: RerankConfig | None = None,
        extractor=None,
        decision_logit_threshold: float = 0.0,
        cache_dir=None,
    ):
        if (
            isinstance(decision_logit_threshold, bool)
            or not isinstance(decision_logit_threshold, (int, float))
            or not math.isfinite(decision_logit_threshold)
        ):
            raise ValueError("Decision logit threshold must be a finite number")
        if config is not None and not isinstance(config, RerankConfig):
            raise TypeError("config must be a validated RerankConfig")
        self.config = config or getattr(reranker, "config", None) or RerankConfig()
        if (
            reranker is not None
            and config is not None
            and (getattr(reranker, "config", config) != config)
        ):
            raise ValueError("Injected reranker configuration differs from requested configuration")
        self.reranker = reranker if reranker is not None else CrossEncoder(self.config)
        if not isinstance(self.reranker.identity, str) or not self.reranker.identity:
            raise ValueError("Reranker requires a nonempty model identity")
        self.extractor = extractor if extractor is not None else create_extractor("v2")
        extraction_identity = self._extractor_identity()
        self.retriever = ExactRetriever(
            encoder if encoder is not None else TextEncoder("artifacts/e5"),
            cache_dir=cache_dir,
        )
        self.decision_logit_threshold = float(decision_logit_threshold)
        self._components = {
            "retrieval": self.retriever.identity,
            "reranker": self.reranker.identity,
            "extractor": extraction_identity,
            "policy": "required-evidence-v1",
            "serialization": SERIALIZATION_VERSION,
            "decision_logit_threshold": self.decision_logit_threshold,
        }
        self.identity = (
            "retrieve-rerank-v1:"
            + hashlib.sha256(json.dumps(self._components, sort_keys=True).encode()).hexdigest()
        )
        self.last_rank_diagnostics = {}
        self._lock = threading.RLock()
        self._providers = (self.reranker, self.extractor, self.retriever)

    def _extractor_identity(self):
        try:
            return extractor_identity(self.extractor)
        except ValueError:
            identity = getattr(self.extractor, "identity", None)
            if not isinstance(identity, str) or not identity:
                raise ValueError("Custom extractor requires an explicit stable identity") from None
            return identity

    def _assert_identity(self):
        if (
            any(
                current is not original
                for current, original in zip(
                    (self.reranker, self.extractor, self.retriever), self._providers, strict=True
                )
            )
            or self.reranker.identity != self._components["reranker"]
            or self.retriever.identity != self._components["retrieval"]
            or self._extractor_identity() != self._components["extractor"]
            or self.decision_logit_threshold != self._components["decision_logit_threshold"]
        ):
            raise ValueError(
                "Backend component/decision identity changed; construct a new predictor"
            )

    def _evidence(self, profile, vacancy):
        self._assert_identity()
        evidence = build_evidence(profile, vacancy, self.extractor)
        self._assert_identity()
        return evidence

    @staticmethod
    def _profile(profile):
        return MatchingProfile.model_validate(
            profile.model_dump() if isinstance(profile, Profile) else profile, strict=True
        )

    @staticmethod
    def _vacancy(vacancy):
        return Vacancy.model_validate(
            vacancy.model_dump() if isinstance(vacancy, Vacancy) else vacancy, strict=True
        )

    def _score(self, profile, vacancies):
        self._assert_identity()
        scores = self.reranker.score_pairs([cross_encoder_texts(profile, v) for v in vacancies])
        self._assert_identity()
        if len(scores) != len(vacancies):
            raise ValueError("Cross-encoder returned an inconsistent number of scores")
        scores = [float(score) for score in scores]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Cross-encoder returned a non-finite score")
        diagnostics = getattr(self.reranker, "last_diagnostics", [])
        if len(diagnostics) != len(scores):
            diagnostics = [{} for _ in scores]
        return scores, deepcopy(diagnostics)

    def _result(self, index, profile, vacancy, evidence, raw, cosine, model_diagnostics):
        score = sigmoid(raw) if raw is not None else 0.0
        analysis = Postprocessor(sigmoid(self.decision_logit_threshold)).process(
            score, evidence.detail, profile, vacancy
        )
        # Compare raw scores: sigmoid can saturate for extreme finite logits.
        analysis.relevant = bool(
            raw is not None
            and raw >= self.decision_logit_threshold
            and evidence.status == "eligible"
            and not analysis.preference_conflicts
            and not analysis.preference_unknowns
        )
        analysis.explanation = (
            "Cross-encoder document relevance is an uncalibrated score, not a job-fit probability. "
            + analysis.explanation
        )
        if evidence.blocked_reasons:
            analysis.explanation += " Automatic recommendation blocked: " + " ".join(
                evidence.blocked_reasons
            )
        if evidence.review_reasons:
            analysis.explanation += " Requires verification: " + " ".join(evidence.review_reasons)
        return RankedVacancy(
            index=index,
            vacancy=vacancy,
            analysis=analysis,
            retrieval_score=cosine,
            rerank_score=raw,
            evidence_status=evidence.status,
            diagnostics=deepcopy(
                {
                    "evidence": evidence.metadata,
                    "blocked_reasons": evidence.blocked_reasons,
                    "review_reasons": evidence.review_reasons,
                    "model": model_diagnostics,
                    "identities": self._components,
                }
            ),
            backend_identity=self.identity,
        )

    def analyze_pair(self, profile: Profile | dict, vacancy: Vacancy | dict) -> RankedVacancy:
        with self._lock:
            profile, vacancy = self._profile(profile), self._vacancy(vacancy)
            evidence = self._evidence(profile, vacancy)
            if not evidence.metadata["usable_input"]:
                return self._result(0, profile, vacancy, evidence, None, None, {})
            cosine = self.retriever.score_all(profile, [vacancy])[0].cosine_score
            scores, diagnostics = self._score(profile, [vacancy])
            return self._result(0, profile, vacancy, evidence, scores[0], cosine, diagnostics[0])

    def predict(self, profile: Profile | dict, vacancy: Vacancy | dict) -> VacancyAnalysis:
        return self.analyze_pair(profile, vacancy).analysis

    def rank(
        self,
        profile: Profile | dict,
        vacancies: Sequence[Vacancy | dict],
        preselect_k: int = 20,
        top_k: int = 10,
    ) -> list[RankedVacancy]:
        if type(preselect_k) is not int or preselect_k < 1:
            raise ValueError("preselect_k must be a positive integer")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not isinstance(vacancies, (list, tuple)):
            raise TypeError("vacancies must be a list or tuple")
        with self._lock:
            self._assert_identity()
            profile = self._profile(profile)
            vacancies = [self._vacancy(v) for v in vacancies]
            self.last_rank_diagnostics = {
                "input_count": len(vacancies),
                "preference_filtered_count": 0,
                "retrieved_count": 0,
                "reranked_count": 0,
                "blocked_count": 0,
                "returned_count": 0,
                "preselect_k": preselect_k,
                "top_k": top_k,
                "backend_identity": self.identity,
            }
            if not vacancies or top_k == 0:
                return []
            if not has_candidate_evidence(profile):
                self.last_rank_diagnostics["reason"] = "Positive candidate evidence is empty"
                return []
            allowed = [
                (i, v) for i, v in enumerate(vacancies) if not preference_conflicts(profile, v)
            ]
            self.last_rank_diagnostics["preference_filtered_count"] = len(vacancies) - len(allowed)
            if not allowed:
                return []
            retrieved = self.retriever.retrieve(profile, [v for _, v in allowed], preselect_k)
            self.last_rank_diagnostics["retrieved_count"] = len(retrieved)
            evidence = [self._evidence(profile, r.vacancy) for r in retrieved]
            usable = [i for i, e in enumerate(evidence) if e.metadata["usable_input"]]
            scores, diagnostics = (
                self._score(profile, [retrieved[i].vacancy for i in usable]) if usable else ([], [])
            )
            scored = dict(zip(usable, zip(scores, diagnostics, strict=True), strict=True))
            self.last_rank_diagnostics["reranked_count"] = len(usable)
            results = []
            for i, candidate in enumerate(retrieved):
                raw, model_diagnostics = scored.get(i, (None, {}))
                result = self._result(
                    allowed[candidate.index][0],
                    profile,
                    candidate.vacancy,
                    evidence[i],
                    raw,
                    candidate.cosine_score,
                    model_diagnostics,
                )
                if result.evidence_status == "blocked":
                    self.last_rank_diagnostics["blocked_count"] += 1
                else:
                    results.append(result)
            results.sort(
                key=lambda r: (
                    not r.analysis.relevant,
                    -(r.rerank_score if r.rerank_score is not None else -math.inf),
                    r.index,
                )
            )
            results = results[:top_k]
            self.last_rank_diagnostics["returned_count"] = len(results)
            return results
