"""Additive backend selection; the existing public predict remains E5 + MLP."""

from typing import Protocol

from pydantic import Field

from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


class MatchingProfile(Profile):
    """Explicit extension for experimental backends; training PairRecord stays unchanged."""

    preferred_roles: list[str] = Field(default_factory=list)


class MatchingBackend(Protocol):
    def predict(self, profile: Profile | dict, vacancy: Vacancy | dict) -> VacancyAnalysis: ...


def create_predictor(backend: str = "mlp", **kwargs) -> MatchingBackend:
    if backend == "rerank":
        from resumeanalysis.rerank.predictor import RerankPredictor

        return RerankPredictor(**kwargs)
    if backend == "qwen":
        from resumeanalysis.qwen.predictor import QwenPredictor

        return QwenPredictor(**kwargs)
    if backend == "mlp":
        from resumeanalysis.inference.predictor import Predictor

        return Predictor(**kwargs)
    raise ValueError("Backend must be mlp, qwen or rerank")
