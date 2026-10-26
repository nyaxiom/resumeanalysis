import os
from functools import lru_cache
from pathlib import Path

import torch

from resumeanalysis.embeddings.encoder import Encoder, TextEncoder
from resumeanalysis.extraction.registry import Extractor, is_legacy
from resumeanalysis.features.feature_builder import (
    FEATURE_NAMES,
    FeatureEngineering,
)
from resumeanalysis.models.resume_analysis_net import ResumeAnalysisNet
from resumeanalysis.postprocessing.rules import Postprocessor
from resumeanalysis.schemas import Profile, Vacancy, VacancyAnalysis


class Predictor:
    def __init__(
        self,
        checkpoint: str | Path,
        encoder: Encoder,
        threshold: float | None = None,
        extractor: Extractor | None = None,
    ):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            type(data.get("format_version")) is not int
            or data.get("format_version") not in {1, 2}
            or data.get("feature_names") != FEATURE_NAMES
        ):
            raise ValueError("Incompatible checkpoint feature schema")
        if data.get("encoder_identity") != encoder.identity:
            raise ValueError("Checkpoint encoder does not match runtime encoder")
        self.builder = FeatureEngineering(encoder, extractor)
        legacy = is_legacy(self.builder.extractor)
        if data["format_version"] == 1 and not legacy:
            raise ValueError(
                "Legacy checkpoint requires the original SkillExtractor; retrain for changed extraction"
            )
        if data["format_version"] == 2 and (
            legacy or data.get("extractor_identity") != self.builder.extractor_identity
        ):
            raise ValueError(
                "Checkpoint extractor identity mismatch; retrain for changed extraction"
            )
        if (
            type(data.get("pipeline_version")) is not int
            or data.get("pipeline_version") != self.builder.pipeline_version
            or data.get("taxonomy_identity") != self.builder.extractor.normalizer.taxonomy.identity
        ):
            raise ValueError("Checkpoint extraction/taxonomy pipeline mismatch")
        self.model = ResumeAnalysisNet(**data["network_config"])
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()
        self.postprocessor = Postprocessor(data["threshold"] if threshold is None else threshold)

    def predict(self, profile: Profile | dict, vacancy: Vacancy | dict) -> VacancyAnalysis:
        p = profile if isinstance(profile, Profile) else Profile.model_validate(profile)
        v = vacancy if isinstance(vacancy, Vacancy) else Vacancy.model_validate(vacancy)
        detail = self.builder.build(p, v)
        with torch.inference_mode():
            score = (
                self.model(
                    *[
                        torch.from_numpy(a).unsqueeze(0)
                        for a in (detail.resume_embedding, detail.vacancy_embedding, detail.vector)
                    ]
                )
                .sigmoid()
                .item()
            )
        return self.postprocessor.process(score, detail, p, v)


@lru_cache(maxsize=1)
def _default_predictor() -> Predictor:
    checkpoint = os.getenv("RESUMEANALYSIS_CHECKPOINT", "artifacts/run/model.pt")
    model_path = os.getenv("RESUMEANALYSIS_E5_PATH", "artifacts/e5")
    return Predictor(checkpoint, TextEncoder(model_path))


def predict(profile: Profile | dict, vacancy: Vacancy | dict) -> VacancyAnalysis:
    """Local trained-model inference. Missing weights fail explicitly; no heuristic fallback."""
    return _default_predictor().predict(profile, vacancy)
