"""Local job recommendations that learn from explicit user feedback."""

from .recommendation.engine import RecommendationEngine
from .recommendation.repository import SQLiteFeedbackRepository
from .recommendation.schema import StructuredProfile, StructuredVacancy

__all__ = [
    "RecommendationEngine",
    "SQLiteFeedbackRepository",
    "StructuredProfile",
    "StructuredVacancy",
]
