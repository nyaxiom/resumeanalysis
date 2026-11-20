import os

import pytest
from fastapi.testclient import TestClient

from resumeanalysis.config import Settings
from resumeanalysis.main import create_app

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

VACANCIES = [
    {"vacancy_id": "sales", "title": "Sales", "description": "Customer sales"},
    {"vacancy_id": "python", "title": "Python developer", "description": "Build Python services"},
]


class FakeEmbeddings:
    identity = "deterministic-test-v1"

    def __init__(self):
        self.calls = []

    def encode_many(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] if "Python" in text else [0.0, 1.0] for text in texts]


class FakeLLM:
    output = "I build Python services and would bring this experience to your developer role."

    def __init__(self):
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.output


@pytest.fixture
def settings(tmp_path):
    return Settings(database_path=tmp_path / "service.sqlite3", embedding_cache_dir=None)


@pytest.fixture
def client(settings):
    app = create_app(settings, embeddings=FakeEmbeddings(), letter_backend=FakeLLM())
    with TestClient(app) as client:
        yield client


def rank(client, user="alice", vacancies=None):
    response = client.post("/rank", json={"user_id": user, "vacancies": vacancies or VACANCIES})
    assert response.status_code == 200, response.text
    return response.json()


def feedback(client, vacancy_id, liked, user="alice", vacancy=None):
    body = {"user_id": user, "vacancy_id": vacancy_id, "liked": liked}
    if vacancy is not None:
        body["vacancy"] = vacancy
    response = client.post("/feedback", json=body)
    assert response.status_code == 200, response.text
    return response.json()
