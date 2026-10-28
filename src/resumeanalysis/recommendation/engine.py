"""Local recommendation orchestration with isolated, replayable River state."""

import copy
import hashlib
import threading
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import Field, TypeAdapter

from resumeanalysis.recommendation.learner import (
    LearnerConfig,
    PreferenceLearner,
    effective_events,
    recorded_metrics,
)
from resumeanalysis.recommendation.schema import (
    FeedbackEvent,
    Identifier,
    Recommendation,
    StrictModel,
    StructuredProfile,
    StructuredVacancy,
)


class RecommendationConfig(StrictModel):
    exploration_fraction: float = Field(default=0.1, ge=0, le=0.5)
    exploration_seed: int = 42
    include_unknown_constraints: bool = True


class RecommendationEngine:
    def __init__(self, feature_builder, repository, *, learner_config=None, config=None):
        self.builder, self.repository = feature_builder, repository
        self.learner_config = learner_config or LearnerConfig()
        self.config = config or RecommendationConfig()
        self._cache = {}
        self._lock = threading.RLock()
        self._identity = {
            "feature_identity": self.builder.identity,
            "feature_schema_version": self.builder.schema_version,
            "model_version": self.learner_config.model_version,
            "learner_config": self.learner_config.model_dump(),
        }

    @staticmethod
    def _user(user_id):
        return TypeAdapter(Identifier).validate_python(user_id, strict=True)

    def _state(self, session, user_id):
        state, events = session.initialize(self._identity), session.events()
        cached = self._cache.get(user_id)
        if cached is not None and cached[0] == state["revision"]:
            learner = cached[1]
        else:
            learner = PreferenceLearner.replay(
                effective_events(events, after_sequence=state["reset_after"]), self.learner_config
            )
        return state, events, learner

    def recommend(
        self,
        user_id: str,
        profile: StructuredProfile,
        vacancies: list[StructuredVacancy],
        limit: int = 20,
    ) -> list[Recommendation]:
        user_id = self._user(user_id)
        profile = StructuredProfile.model_validate(profile.model_dump(), strict=True)
        if profile.user_id != user_id:
            raise ValueError("Profile belongs to a different user")
        if profile.review_status == "needs_review":
            raise ValueError("Confirm or correct extracted profile facts before recommendation")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Recommendation limit must be between 1 and 100")
        vacancies = [
            StructuredVacancy.model_validate(v.model_dump(), strict=True) for v in vacancies
        ]
        if len({v.vacancy_id for v in vacancies}) != len(vacancies):
            raise ValueError("Vacancy IDs must be unique within a candidate pool")
        if not vacancies:
            return []
        # Embeddings do not require the SQLite writer lock. They never fit a user model.
        built = [(vacancy, self.builder.build(profile, vacancy)) for vacancy in vacancies]
        with self._lock:
            with self.repository.transaction(user_id) as session:
                state, events, learner = self._state(session, user_id)
                rejected = {e.vacancy_id for e in effective_events(events) if e.action == "dislike"}
                candidates = []
                for vacancy, features in built:
                    if vacancy.vacancy_id in rejected or features.constraints.status == "blocked":
                        continue
                    if (
                        features.constraints.status == "unknown"
                        and not self.config.include_unknown_constraints
                    ):
                        continue
                    preference = learner.predict_proba_one(features.snapshot.values)
                    weight = self.learner_config.preference_weight * min(
                        1.0, learner.count / self.learner_config.warmup_feedback
                    )
                    ranking = (1 - weight) * features.snapshot.content_score + weight * preference
                    candidates.append((vacancy, features, preference, float(ranking)))
                candidates.sort(key=lambda row: (-row[3], row[0].vacancy_id))
                count = min(limit, len(candidates))
                explore_count = (
                    max(1, int(count * self.config.exploration_fraction))
                    if count and self.config.exploration_fraction and len(candidates) > count
                    else 0
                )
                selected = [(row, False) for row in candidates[: count - explore_count]]
                remaining = candidates[count - explore_count :]
                remaining.sort(
                    key=lambda row: hashlib.sha256(
                        f"{self.config.exploration_seed}:{user_id}:{state['revision']}:{row[0].vacancy_id}".encode()
                    ).digest()
                )
                selected += [(row, True) for row in remaining[:explore_count]]
                results = []
                for rank, ((vacancy, features, preference, ranking), exploration) in enumerate(
                    selected, 1
                ):
                    identifier = str(uuid4())
                    session.save_snapshot(
                        identifier, vacancy.vacancy_id, features.snapshot, preference
                    )
                    explanation = (
                        "Cold-start ranking from explicit preferences and qualification evidence. "
                        if not learner.count
                        else f"Ranking combines content evidence with {learner.count} recent explicit feedback labels. "
                    )
                    explanation += (
                        "Preference is an uncalibrated interest score, not a qualification verdict."
                    )
                    if exploration:
                        explanation += " This slot explores a different eligible candidate."
                    if features.constraints.status == "unknown":
                        explanation += " Some hard constraints need confirmation."
                    results.append(
                        Recommendation(
                            recommendation_id=identifier,
                            vacancy_id=vacancy.vacancy_id,
                            qualification=features.qualification,
                            preference_score=preference,
                            hard_constraints=features.constraints,
                            ranking_score=ranking,
                            rank=rank,
                            exploration=exploration,
                            explanation=explanation,
                            profile_version=profile.version,
                            feature_schema_version=self.builder.schema_version,
                            model_version=self.learner_config.model_version,
                        )
                    )
            self._cache[user_id] = (state["revision"], learner)
            return results

    def record_feedback(
        self,
        user_id,
        vacancy_id,
        action,
        reason=None,
        *,
        event_id=None,
        recommendation_id=None,
        corrects_event_id=None,
    ):
        user_id, vacancy_id = self._user(user_id), self._user(vacancy_id)
        if action not in {"like", "dislike"}:
            raise ValueError("Only explicit like/dislike actions are learning labels")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 2000):
            raise ValueError("Feedback reason must be a bounded string")
        if event_id is not None:
            event_id = self._user(event_id)
        with self._lock:
            with self.repository.transaction(user_id) as session:
                state, events, learner = self._state(session, user_id)
                by_id = {event.event_id: event for event in events}
                if event_id in by_id:
                    previous = by_id[event_id]
                    if (
                        previous.vacancy_id,
                        previous.action,
                        previous.reason,
                        previous.corrects_event_id,
                    ) != (vacancy_id, action, reason, corrects_event_id) or (
                        recommendation_id is not None
                        and previous.recommendation_id != recommendation_id
                    ):
                        raise ValueError("Event ID was reused with a different payload")
                    return previous
                active = [e for e in effective_events(events) if e.vacancy_id == vacancy_id]
                if active and corrects_event_id is None:
                    previous = active[-1]
                    if previous.action == action and previous.reason == reason:
                        return previous
                    raise ValueError("Changing feedback requires corrects_event_id")
                if corrects_event_id is not None:
                    if not active or active[-1].event_id != corrects_event_id:
                        raise ValueError(
                            "Correction must reference the active feedback for this vacancy"
                        )
                    original = active[-1]
                    if (
                        recommendation_id is not None
                        and recommendation_id != original.recommendation_id
                    ):
                        raise ValueError("A correction must preserve the original feature snapshot")
                    recommendation_id = original.recommendation_id
                saved = (
                    session.snapshot(recommendation_id)
                    if recommendation_id is not None
                    else session.latest_snapshot(vacancy_id)
                )
                if saved is None or saved["vacancy_id"] != vacancy_id:
                    raise ValueError(
                        "Feedback requires a recommendation previously shown to this user"
                    )
                features = saved["features"]
                if features.identity != self.builder.identity:
                    raise ValueError("Feedback feature identity is incompatible with this learner")
                event = FeedbackEvent(
                    user_id=user_id,
                    vacancy_id=vacancy_id,
                    event_id=event_id or str(uuid4()),
                    action=action,
                    reason=reason,
                    timestamp=datetime.now(timezone.utc),
                    recommendation_id=saved["recommendation_id"],
                    corrects_event_id=corrects_event_id,
                    profile_version=features.profile_version,
                    feature_schema_version=features.schema_version,
                    model_version=self.learner_config.model_version,
                    feature_snapshot=features,
                    prediction_before_update=learner.predict_proba_one(features.values),
                    prediction_at_recommendation=saved["prediction"],
                )
                event = session.append(event)
                active_after = effective_events(
                    [*events, event], after_sequence=state["reset_after"]
                )
                if corrects_event_id or len(active_after) > self.learner_config.history_window:
                    updated = PreferenceLearner.replay(active_after, self.learner_config)
                else:
                    updated = copy.deepcopy(learner)
                    updated.learn_one(features.values, action == "like")
                revision = state["revision"] + 1
                session.update_model(revision, state["reset_after"])
            # A rollback/commit failure cannot publish partially trained in-memory state.
            self._cache[user_id] = (revision, updated)
            return event

    def reset(self, user_id):
        user_id = self._user(user_id)
        with self._lock:
            with self.repository.transaction(user_id) as session:
                state, events, _ = self._state(session, user_id)
                revision = state["revision"] + 1
                session.update_model(revision, max((e.sequence for e in events), default=0))
            self._cache[user_id] = (revision, PreferenceLearner(self.learner_config))

    def replay(self, user_id, *, include_before_reset=False):
        user_id = self._user(user_id)
        with self._lock:
            with self.repository.transaction(user_id) as session:
                state, events, _ = self._state(session, user_id)
                after = 0 if include_before_reset else state["reset_after"]
                learner = PreferenceLearner.replay(
                    effective_events(events, after_sequence=after), self.learner_config
                )
                revision = state["revision"] + 1
                session.update_model(revision, after)
            self._cache[user_id] = (revision, learner)

    def online_metrics(self, user_id):
        user_id = self._user(user_id)
        with self._lock, self.repository.transaction(user_id) as session:
            state, events, _ = self._state(session, user_id)
            return recorded_metrics(effective_events(events, after_sequence=state["reset_after"]))
