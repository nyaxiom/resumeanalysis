"""Optional historical matching and research tools, excluded from the runtime wheel."""

from .inference.predictor import Predictor, predict

__all__ = ["Predictor", "predict"]
