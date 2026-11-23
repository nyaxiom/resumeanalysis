import asyncio
import os
from uuid import uuid4

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

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
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()  # Inference must be in a worker thread.
        self.calls.append(list(texts))
        return [[1.0, 0.0] if "Python" in text else [0.0, 1.0] for text in texts]


class FakeLLM:
    output = "I build Python services and would bring this experience to your developer role."

    def __init__(self):
        self.calls = []

    def generate(self, messages):
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        self.calls.append(messages)
        return self.output


@pytest.fixture
def settings(tmp_path):
    return Settings(embedding_cache_dir=None)


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


@pytest.fixture
def empty_database_url(monkeypatch):
    admin = os.getenv("RESUMEANALYSIS_TEST_DATABASE_URL")
    if not admin:
        pytest.skip("Set RESUMEANALYSIS_TEST_DATABASE_URL for PostgreSQL integration tests")
    parsed = make_url(admin)
    if parsed.drivername != "postgresql+asyncpg" or not (parsed.database or "").startswith(
        "resumeanalysis_test"
    ):
        pytest.fail("Integration tests require a dedicated resumeanalysis_test database")
    name = "resumeanalysis_test_" + uuid4().hex

    async def manage(create):
        connection = await asyncpg.connect(
            parsed.set(drivername="postgresql").render_as_string(hide_password=False)
        )
        try:
            version = await connection.fetchval("SHOW server_version_num")
            assert 160000 <= int(version) < 170000, "Integration tests require PostgreSQL 16"
            if create:
                await connection.execute(f'CREATE DATABASE "{name}"')
            else:
                await connection.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        finally:
            await connection.close()

    asyncio.run(manage(True))
    url = parsed.set(database=name).render_as_string(hide_password=False)
    monkeypatch.setenv("RESUMEANALYSIS_DATABASE_URL", url)
    try:
        yield url
    finally:
        asyncio.run(manage(False))


@pytest.fixture
def db_settings(empty_database_url):
    command.upgrade(Config("alembic.ini"), "head")
    return Settings(database_url=empty_database_url, embedding_cache_dir=None)


def db(client, statement):
    """Run test inspection/corruption on the application's own async event loop."""

    async def execute():
        async with client.app.state.recommender.storage.sessions.begin() as session:
            result = await session.execute(statement)
            return result.scalar_one_or_none() if statement.is_select else None

    return client.portal.call(execute)
