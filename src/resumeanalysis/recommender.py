"""Independent River logistic regressions trained only on explicit user feedback."""

import hashlib
import json
import math
from importlib.metadata import version

from river import linear_model, optim
from starlette.concurrency import run_in_threadpool

from resumeanalysis.config import Settings
from resumeanalysis.schemas import FeedbackResponse, RankedVacancy, RankResponse, VacancyText


class UnknownVacancy(ValueError):
    pass


class Recommender:
    def __init__(self, storage, embeddings, settings: Settings):
        self.storage, self.embeddings, self.settings = storage, embeddings, settings

    @property
    def identity(self):
        return json.dumps(
            [
                "river-embedding-logreg-v1",
                version("river"),
                self.embeddings.identity,
                self.settings.learning_rate,
                self.settings.l2,
            ]
        )

    def _new_model(self):
        return linear_model.LogisticRegression(
            optimizer=optim.SGD(self.settings.learning_rate),
            intercept_lr=self.settings.learning_rate,
            l2=self.settings.l2,
        )

    def _vectors(self, texts):
        vectors = self.embeddings.encode_many(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding batch length mismatch")
        dimension = len(vectors[0]) if vectors else 0
        if vectors and not 1 <= dimension <= 4096:
            raise RuntimeError("Invalid embedding dimension")
        if any(len(v) != dimension or not all(math.isfinite(x) for x in v) for v in vectors):
            raise RuntimeError("Invalid embedding vectors")
        return [dict(enumerate(v)) for v in vectors]

    @staticmethod
    def _personalized(history):
        return {row["liked"] for row in history} == {0, 1}

    async def _save(self, session, model, dimension, revision, identity):
        state = json.dumps(
            {
                "identity": identity,
                "revision": revision,
                "dimension": dimension,
                "weights": [float(model.weights.get(i, 0)) for i in range(dimension)],
                "intercept": float(model.intercept),
                "iterations": model.optimizer.n_iterations,
            },
            allow_nan=False,
            sort_keys=True,
        )
        await session.save_model(state, hashlib.sha256(state.encode()).hexdigest(), revision)

    def _replay(self, history, dimension):
        model = self._new_model()
        vectors = self._vectors(
            [VacancyText(title=r["title"], description=r["description"]).text for r in history]
        )
        if any(len(v) != dimension for v in vectors):
            raise RuntimeError("History and ranking embeddings differ in dimension")
        for row, features in zip(history, vectors, strict=True):
            model.learn_one(features, bool(row["liked"]))
        return model

    async def _model(self, session, dimension, identity):
        row, history = session.state(), await session.history()
        try:
            raw = row["model_json"]
            if not raw or hashlib.sha256(raw.encode()).hexdigest() != row["model_checksum"]:
                raise ValueError("Missing or corrupt model")
            state = json.loads(raw)
            if (
                state["identity"] != identity
                or state["revision"] != row["revision"]
                or state["dimension"] != dimension
                or state["iterations"] != len(history)
            ):
                raise ValueError("Stale model")
            weights, intercept = state["weights"], state["intercept"]
            if len(weights) != dimension or not all(
                type(x) in (int, float) and math.isfinite(x) for x in [*weights, intercept]
            ):
                raise ValueError("Invalid numeric state")
            model = self._new_model()
            # River is pinned. Constant-rate SGD has no momentum/accumulator state.
            # Only numeric weights/intercept/iteration count are restored, never objects.
            model._weights.update(dict(enumerate(weights)))
            model.intercept = intercept
            model.optimizer.n_iterations = len(history)
        except (ValueError, TypeError, KeyError, OverflowError):
            model = await run_in_threadpool(self._replay, history, dimension)
            await self._save(session, model, dimension, row["revision"], identity)
        return model, history, row["revision"]

    @staticmethod
    def _ranked(vacancies, vectors, model, personalized):
        results = []
        for vacancy, features in zip(vacancies, vectors, strict=True):
            score = float(model.predict_proba_one(features)[True]) if personalized else 0.5
            results.append(RankedVacancy(**vacancy.model_dump(), score=score))
        results.sort(key=lambda v: -v.score)
        return RankResponse(personalized=personalized, vacancies=results)

    async def rank(self, request):
        vectors = await run_in_threadpool(self._vectors, [v.text for v in request.vacancies])
        identity = await run_in_threadpool(lambda: self.identity)
        async with self.storage.transaction(request.user_id) as session:
            model, history, _ = await self._model(session, len(vectors[0]), identity)
            for vacancy in request.vacancies:
                await session.save_vacancy(vacancy.vacancy_id, vacancy)
        # This request owns its model snapshot; scoring never mutates shared state.
        return await run_in_threadpool(
            self._ranked,
            request.vacancies,
            vectors,
            model,
            self._personalized(history),
        )

    async def feedback(self, request):
        identity = await run_in_threadpool(lambda: self.identity)
        features = (
            (await run_in_threadpool(self._vectors, [request.vacancy.text]))[0]
            if request.vacancy is not None
            else None
        )
        async with self.storage.transaction(request.user_id) as session:
            previous = await session.feedback(request.vacancy_id)
            vacancy = request.vacancy
            if vacancy is None:
                # Keep the originally reacted-to text unless explicitly replaced.
                saved = previous or await session.vacancy(request.vacancy_id)
                if saved is None:
                    raise UnknownVacancy("Supply vacancy data or rank it for this user first")
                vacancy = VacancyText(title=saved["title"], description=saved["description"])
                features = (await run_in_threadpool(self._vectors, [vacancy.text]))[0]
            dimension = len(features)
            model, history, revision = await self._model(session, dimension, identity)
            updated = previous is None or (
                previous["liked"],
                previous["title"],
                previous["description"],
            ) != (request.liked, vacancy.title, vacancy.description)
            if updated:
                await session.save_feedback(request.vacancy_id, vacancy, request.liked)
                history = await session.history()
                if previous is None:
                    await run_in_threadpool(model.learn_one, features, request.liked)
                else:
                    model = await run_in_threadpool(self._replay, history, dimension)
                revision += 1
                await self._save(session, model, dimension, revision, identity)
            return FeedbackResponse(
                updated=updated,
                revision=revision,
                feedback_count=len(history),
                personalized=self._personalized(history),
            )
