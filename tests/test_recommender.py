import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import VACANCIES, FakeEmbeddings, feedback, rank
from fastapi.testclient import TestClient
from river import linear_model

from resumeanalysis.main import create_app
from resumeanalysis.recommender import Recommender
from resumeanalysis.schemas import FeedbackRequest
from resumeanalysis.storage import Storage


def snapshot(client, user="alice"):
    db = client.app.state.recommender.storage.connection
    row = db.execute("SELECT model_json FROM users WHERE user_id=?", (user,)).fetchone()
    return json.loads(row[0])


def test_cold_start_then_real_likes_and_dislikes(client, monkeypatch):
    calls = []
    original = linear_model.LogisticRegression.learn_one

    def learn(model, features, liked, *args, **kwargs):
        calls.append((features, liked))
        return original(model, features, liked, *args, **kwargs)

    monkeypatch.setattr(linear_model.LogisticRegression, "learn_one", learn)
    result = rank(client)
    assert result["personalized"] is False
    assert [v["vacancy_id"] for v in result["vacancies"]] == ["sales", "python"]
    assert [v["score"] for v in result["vacancies"]] == [0.5, 0.5]
    assert calls == []
    assert len(client.app.state.recommender.embeddings.calls[0]) == 2
    assert feedback(client, "python", True) == {
        "updated": True,
        "revision": 1,
        "feedback_count": 1,
        "personalized": False,
    }
    assert calls == [({0: 1.0, 1: 0.0}, True)]
    assert snapshot(client)["weights"][0] > 0
    assert rank(client) == result  # Still neutral with only one class.
    assert feedback(client, "sales", False)["personalized"] is True
    assert calls[-1] == ({0: 0.0, 1: 1.0}, False)
    assert snapshot(client)["weights"][1] < 0
    ordered = rank(client)
    assert ordered["personalized"] is True
    assert ordered["vacancies"][0]["vacancy_id"] == "python"
    assert ordered["vacancies"][0]["score"] > ordered["vacancies"][1]["score"]


def test_dislike_first_remains_neutral(client):
    rank(client)
    feedback(client, "sales", False)
    assert snapshot(client)["weights"][1] < 0
    assert not rank(client)["personalized"]


def test_user_models_and_stored_vacancies_are_isolated(client):
    rank(client)
    feedback(client, "python", True)
    feedback(client, "sales", False)
    before = snapshot(client)
    denied = client.post(
        "/feedback", json={"user_id": "bob", "vacancy_id": "python", "liked": True}
    )
    assert denied.status_code == 404
    assert not rank(client, "bob")["personalized"]
    feedback(client, "python", False, "bob")
    feedback(client, "sales", True, "bob")
    assert rank(client, "bob")["vacancies"][0]["vacancy_id"] == "sales"
    assert snapshot(client) == before
    assert rank(client)["vacancies"][0]["vacancy_id"] == "python"


def test_restart_restores_model_and_continues_learning(settings):
    with TestClient(create_app(settings, embeddings=FakeEmbeddings())) as first:
        rank(first)
        feedback(first, "python", True)
        feedback(first, "sales", False)
        expected, state = rank(first), snapshot(first)
    encoder = FakeEmbeddings()
    with TestClient(create_app(settings, embeddings=encoder)) as restarted:
        assert rank(restarted) == expected
        assert snapshot(restarted) == state
        assert len(encoder.calls) == 1  # Restored state did not replay training.
        feedback(
            restarted, "python2", True, vacancy={"title": "Python", "description": "Python API"}
        )
        after = rank(restarted)
        with restarted.app.state.recommender.storage.transaction() as db:
            db.execute("UPDATE users SET model_json=NULL")
        assert rank(restarted) == after  # Same result as fresh replay of the full history.


def test_identical_feedback_does_not_train_again(client, monkeypatch):
    rank(client)
    feedback(client, "python", True)
    original = snapshot(client)

    def forbidden(*args, **kwargs):
        raise AssertionError("Duplicate feedback called learn_one")

    monkeypatch.setattr(linear_model.LogisticRegression, "learn_one", forbidden)
    assert feedback(client, "python", True)["updated"] is False
    data = {k: v for k, v in VACANCIES[1].items() if k != "vacancy_id"}
    assert feedback(client, "python", True, vacancy=data)["revision"] == 1
    assert snapshot(client) == original


def test_correction_replays_current_labels_and_cold_start(client):
    rank(client)
    feedback(client, "python", True)
    feedback(client, "sales", False)
    result = feedback(client, "python", False)
    assert result == {"updated": True, "revision": 3, "feedback_count": 2, "personalized": False}
    assert snapshot(client)["iterations"] == 2
    assert rank(client)["vacancies"][0]["vacancy_id"] == "sales"
    assert not rank(client)["personalized"]
    feedback(client, "sales", True)
    assert rank(client)["vacancies"][0]["vacancy_id"] == "sales"
    rank(client, "fresh")
    feedback(client, "python", False, "fresh")
    feedback(client, "sales", True, "fresh")
    assert rank(client) == rank(client, "fresh")


def test_feedback_with_inline_data_and_changed_text(client):
    data = {"title": "Python", "description": "Python systems"}
    feedback(client, "direct", True, vacancy=data)
    assert snapshot(client)["weights"][0] > 0
    feedback(client, "direct", True, vacancy={"title": "Sales", "description": "Customer calls"})
    assert snapshot(client)["weights"][0] == 0
    assert snapshot(client)["weights"][1] > 0
    assert snapshot(client)["iterations"] == 1


def test_correction_preserves_reacted_to_text(client):
    rank(client)
    feedback(client, "python", True)
    rank(client, vacancies=[{"vacancy_id": "python", "title": "Sales", "description": "Calls"}])
    feedback(client, "python", False)
    assert snapshot(client)["weights"][0] < 0
    assert snapshot(client)["weights"][1] == 0


@pytest.mark.parametrize("corruption", [None, "not-json", "[]", "wrong-checksum", "NaN"])
def test_missing_or_corrupted_snapshot_rebuilt_from_feedback(client, corruption):
    rank(client)
    feedback(client, "python", True)
    feedback(client, "sales", False)
    expected = rank(client)
    with client.app.state.recommender.storage.transaction() as db:
        checksum = hashlib.sha256((corruption or "").encode()).hexdigest()
        if corruption == "wrong-checksum":
            checksum = "bad"
        db.execute("UPDATE users SET model_json=?,model_checksum=?", (corruption, checksum))
    assert rank(client) == expected
    assert snapshot(client)["iterations"] == 2


def test_embedding_version_change_reembeds_feedback(settings):
    with TestClient(create_app(settings, embeddings=FakeEmbeddings())) as first:
        rank(first)
        feedback(first, "python", True)
        feedback(first, "sales", False)

    class NewEmbeddings(FakeEmbeddings):
        identity = "changed-model-with-three-dimensions"

        def encode_many(self, texts):
            return [[*v[::-1], 0.0] for v in super().encode_many(texts)]

    encoder = NewEmbeddings()
    with TestClient(create_app(settings, embeddings=encoder)) as changed:
        assert rank(changed)["vacancies"][0]["vacancy_id"] == "python"
        assert len(encoder.calls) == 2  # Rank and re-embedding stored feedback.
        assert snapshot(changed)["dimension"] == 3
        assert snapshot(changed)["iterations"] == 2
        assert encoder.identity in snapshot(changed)["identity"]


def test_failure_rolls_back_feedback_and_model(client, monkeypatch):
    rank(client)
    feedback(client, "python", True)
    before = snapshot(client)
    original = linear_model.LogisticRegression.learn_one

    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(linear_model.LogisticRegression, "learn_one", fail)
    result = client.post(
        "/feedback", json={"user_id": "alice", "vacancy_id": "sales", "liked": False}
    )
    assert result.status_code == 503
    assert snapshot(client) == before
    monkeypatch.setattr(linear_model.LogisticRegression, "learn_one", original)
    assert feedback(client, "sales", False)["feedback_count"] == 2


def test_duplicate_requests_from_two_connections_train_once(settings):
    stores = [Storage(settings.database_path), Storage(settings.database_path)]
    services = [Recommender(store, FakeEmbeddings(), settings) for store in stores]
    request = FeedbackRequest(
        user_id="alice",
        vacancy_id="p",
        liked=True,
        vacancy={"title": "Python", "description": "Python systems"},
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda service: service.feedback(request), services))
        assert sorted(r.updated for r in results) == [False, True]
        assert [r.revision for r in results] == [1, 1]
        state = json.loads(
            stores[0].connection.execute("SELECT model_json FROM users").fetchone()[0]
        )
        assert state["iterations"] == 1
    finally:
        for store in stores:
            store.close()


def test_old_database_is_not_modified(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE preference_users(user_id TEXT)")
        db.execute("INSERT INTO preference_users VALUES ('existing-user')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="new database"):
        Storage(path)
    assert path.read_bytes() == before
