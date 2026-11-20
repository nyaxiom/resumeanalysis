"""Independent River logistic regressions trained only on explicit user feedback."""

import hashlib
import json
import math
from importlib.metadata import version

from river import linear_model, optim

from resumeanalysis.config import Settings
from resumeanalysis.schemas import FeedbackResponse, RankedVacancy, RankResponse, VacancyText
from resumeanalysis.storage import Storage


class UnknownVacancy(ValueError):
    pass


class Recommender:
    def __init__(self, storage: Storage, embeddings, settings: Settings):
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
    def _history(db, user_id):
        return db.execute(
            "SELECT * FROM feedback WHERE user_id=? ORDER BY sequence", (user_id,)
        ).fetchall()

    @staticmethod
    def _personalized(history):
        return {row["liked"] for row in history} == {0, 1}

    def _save(self, db, user_id, model, dimension, revision):
        state = json.dumps(
            {
                "identity": self.identity,
                "revision": revision,
                "dimension": dimension,
                "weights": [float(model.weights.get(i, 0)) for i in range(dimension)],
                "intercept": float(model.intercept),
                "iterations": model.optimizer.n_iterations,
            },
            allow_nan=False,
            sort_keys=True,
        )
        db.execute(
            "UPDATE users SET revision=?,model_json=?,model_checksum=? WHERE user_id=?",
            (revision, state, hashlib.sha256(state.encode()).hexdigest(), user_id),
        )

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

    def _model(self, db, user_id, dimension):
        db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
        row = db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        history = self._history(db, user_id)
        try:
            raw = row["model_json"]
            if not raw or hashlib.sha256(raw.encode()).hexdigest() != row["model_checksum"]:
                raise ValueError("Missing or corrupt model")
            state = json.loads(raw)
            if (
                state["identity"] != self.identity
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
            model = self._replay(history, dimension)
            self._save(db, user_id, model, dimension, row["revision"])
        return model, history, row["revision"]

    @staticmethod
    def _store_vacancy(db, user_id, vacancy_id, vacancy):
        db.execute(
            "INSERT INTO vacancies VALUES (?, ?, ?, ?) ON CONFLICT(user_id,vacancy_id) "
            "DO UPDATE SET title=excluded.title,description=excluded.description",
            (user_id, vacancy_id, vacancy.title, vacancy.description),
        )

    def rank(self, request):
        vectors = self._vectors([v.text for v in request.vacancies])
        with self.storage.transaction() as db:
            model, history, _ = self._model(db, request.user_id, len(vectors[0]))
            personalized = self._personalized(history)
            results = []
            for vacancy, features in zip(request.vacancies, vectors, strict=True):
                self._store_vacancy(db, request.user_id, vacancy.vacancy_id, vacancy)
                score = float(model.predict_proba_one(features)[True]) if personalized else 0.5
                results.append(RankedVacancy(**vacancy.model_dump(), score=score))
        # Python's stable sort preserves input order for cold start and tied scores.
        results.sort(key=lambda v: -v.score)
        return RankResponse(personalized=personalized, vacancies=results)

    def feedback(self, request):
        with self.storage.transaction() as db:
            previous = db.execute(
                "SELECT * FROM feedback WHERE user_id=? AND vacancy_id=?",
                (request.user_id, request.vacancy_id),
            ).fetchone()
            vacancy = request.vacancy
            if vacancy is None:
                # A correction keeps the original reacted-to text, even if /rank
                # has since seen a changed listing. Send explicit text to replace it.
                saved = (
                    previous
                    or db.execute(
                        "SELECT * FROM vacancies WHERE user_id=? AND vacancy_id=?",
                        (request.user_id, request.vacancy_id),
                    ).fetchone()
                )
                if saved is None:
                    raise UnknownVacancy("Supply vacancy data or rank it for this user first")
                vacancy = VacancyText(title=saved["title"], description=saved["description"])
            features = self._vectors([vacancy.text])[0]
            dimension = len(features)
            model, history, revision = self._model(db, request.user_id, dimension)
            updated = previous is None or (
                previous["liked"],
                previous["title"],
                previous["description"],
            ) != (request.liked, vacancy.title, vacancy.description)
            if updated:
                db.execute(
                    "INSERT INTO feedback(user_id,vacancy_id,title,description,liked) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id,vacancy_id) DO UPDATE SET "
                    "title=excluded.title,description=excluded.description,liked=excluded.liked",
                    (
                        request.user_id,
                        request.vacancy_id,
                        vacancy.title,
                        vacancy.description,
                        int(request.liked),
                    ),
                )
                self._store_vacancy(db, request.user_id, request.vacancy_id, vacancy)
                history = self._history(db, request.user_id)
                if previous is None:
                    model.learn_one(features, request.liked)
                else:
                    model = self._replay(history, dimension)
                revision += 1
                self._save(db, request.user_id, model, dimension, revision)
            return FeedbackResponse(
                updated=updated,
                revision=revision,
                feedback_count=len(history),
                personalized=self._personalized(history),
            )
