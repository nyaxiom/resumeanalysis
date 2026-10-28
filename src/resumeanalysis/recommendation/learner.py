"""River's incremental classifier; no custom optimizer or offline model fitting."""

from importlib.metadata import version

from pydantic import Field
from river import linear_model, metrics, optim, preprocessing

from resumeanalysis.recommendation.schema import StrictModel, fingerprint


class LearnerConfig(StrictModel):
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    l2: float = Field(default=0.001, ge=0, le=1)
    history_window: int = Field(default=512, ge=2, le=10000)
    warmup_feedback: int = Field(default=10, ge=1, le=1000)
    preference_weight: float = Field(default=0.7, ge=0, le=1)

    @property
    def model_version(self):
        return (
            "river-logreg-v1:"
            + fingerprint({"config": self.model_dump(), "river_version": version("river")})[:24]
        )


class PreferenceLearner:
    """One user's independent online state. Scores are not calibrated probabilities."""

    def __init__(self, config: LearnerConfig | None = None):
        self.config = config or LearnerConfig()
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(self.config.learning_rate),
            intercept_lr=self.config.learning_rate,
            l2=self.config.l2,
            clip_gradient=10.0,
        )
        self.count = 0
        self.log_loss = metrics.LogLoss()
        self.accuracy = metrics.Accuracy()

    def predict_proba_one(self, features):
        return float(self.model.predict_proba_one(features).get(True, 0.5))

    def learn_one(self, features, liked: bool):
        if type(liked) is not bool:
            raise ValueError("Only explicit like/dislike labels may train preferences")
        prediction = self.predict_proba_one(features)
        self.log_loss.update(liked, {True: prediction, False: 1 - prediction})
        self.accuracy.update(liked, prediction >= 0.5)
        self.model.learn_one(features, liked)
        self.count += 1
        return prediction

    @classmethod
    def replay(cls, events, config=None):
        learner = cls(config)
        for event in events[-learner.config.history_window :]:
            learner.learn_one(event.feature_snapshot.values, event.action == "like")
        return learner


def effective_events(events, *, after_sequence=0):
    """Corrections supersede old labels at correction arrival time, never twice."""
    superseded = {event.corrects_event_id for event in events if event.corrects_event_id}
    return [
        event
        for event in events
        if event.event_id not in superseded and event.sequence > after_sequence
    ]


def recorded_metrics(events):
    """Use recorded pre-update predictions, never predictions from a corrected replay."""
    loss, accuracy = metrics.LogLoss(), metrics.Accuracy()
    brier = 0.0
    for event in events:
        liked, p = event.action == "like", event.prediction_before_update
        loss.update(liked, {True: p, False: 1 - p})
        accuracy.update(liked, p >= 0.5)
        brier += (float(liked) - p) ** 2
    return {
        "count": len(events),
        "log_loss": loss.get() if events else None,
        "accuracy_at_0_5": accuracy.get() if events else None,
        "brier": brier / len(events) if events else None,
        "scope": "Active explicit labels, using immutable predictions recorded before each update",
    }
