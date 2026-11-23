import asyncio
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import anyio
import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from conftest import FakeEmbeddings, db, feedback, rank
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from resumeanalysis.config import Settings
from resumeanalysis.main import create_app
from resumeanalysis.models import MLFeedback, MLUser
from resumeanalysis.storage import UserStorage

pytestmark = pytest.mark.integration


def postgres(url, callback):
    async def run():
        connection = await asyncpg.connect(
            make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)
        )
        try:
            return await callback(connection)
        finally:
            await connection.close()

    return asyncio.run(run())


def test_migrations_repeat_and_leave_backend_tables_untouched(empty_database_url):
    async def sentinel(connection):
        await connection.execute("CREATE TABLE mails(id integer PRIMARY KEY, subject text)")
        await connection.execute("INSERT INTO mails VALUES (1, 'preserve')")
        await connection.execute("CREATE TABLE alembic_version(version_num varchar(32))")
        await connection.execute("INSERT INTO alembic_version VALUES ('backend_revision')")

    postgres(empty_database_url, sentinel)
    output = io.StringIO()
    config = Config("alembic.ini", stdout=output)
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    command.current(config)
    assert "0001_ml_preferences" in output.getvalue()
    command.check(config)  # ORM/migration agreement; ignores foreign tables.

    async def inspect(connection):
        assert (
            await connection.fetchval("SELECT version_num FROM ml_alembic_version")
            == "0001_ml_preferences"
        )
        assert (
            await connection.fetchval("SELECT version_num FROM alembic_version")
            == "backend_revision"
        )
        assert await connection.fetchval("SELECT subject FROM mails WHERE id=1") == "preserve"
        constraints = await connection.fetch(
            "SELECT conname FROM pg_constraint WHERE conrelid='ml_feedback'::regclass"
        )
        assert "ml_feedback_user_vacancy_unique" in {r[0] for r in constraints}

    postgres(empty_database_url, inspect)
    command.downgrade(config, "base")

    async def after(connection):
        assert await connection.fetchval("SELECT to_regclass('ml_users')") is None
        assert await connection.fetchval("SELECT to_regclass('ml_vacancies')") is None
        assert await connection.fetchval("SELECT to_regclass('ml_feedback')") is None
        assert await connection.fetchval("SELECT subject FROM mails WHERE id=1") == "preserve"
        assert (
            await connection.fetchval("SELECT version_num FROM alembic_version")
            == "backend_revision"
        )

    postgres(empty_database_url, after)
    command.upgrade(config, "head")
    command.check(config)


def test_startup_does_not_create_tables_and_missing_schema_is_redacted(empty_database_url):
    app = create_app(Settings(database_url=empty_database_url), embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200
        response = client.post(
            "/rank",
            json={
                "user_id": "secret-user",
                "vacancies": [
                    {
                        "vacancy_id": "hh.ru:123",
                        "title": "Private title",
                        "description": "Private text",
                    },
                ],
            },
        )
        assert response.status_code == 503
        assert "migrations" in response.text
        for hidden in ("secret-user", "Private", "INSERT", "asyncpg", empty_database_url):
            assert hidden not in response.text

    async def inspect(connection):
        return await connection.fetchval("SELECT to_regclass('ml_users')")

    assert postgres(empty_database_url, inspect) is None


def test_database_failure_rolls_back_feedback_and_numeric_state(db_settings, monkeypatch):
    with TestClient(create_app(db_settings, embeddings=FakeEmbeddings())) as client:
        rank(client)
        feedback(client, "python", True)
        before = db(client, select(MLUser.model_json))

        async def fail(*args):
            raise OperationalError(
                "private SQL", {"resume": "secret"}, Exception("private password")
            )

        monkeypatch.setattr(UserStorage, "save_model", fail)
        response = client.post(
            "/feedback", json={"user_id": "alice", "vacancy_id": "sales", "liked": False}
        )
        assert response.status_code == 503 and "private" not in response.text
        assert db(client, select(MLUser.model_json)) == before
        assert db(client, select(func.count()).select_from(MLFeedback)) == 1


def test_same_user_concurrent_distinct_labels_and_corrections(db_settings):
    apps = [create_app(db_settings, embeddings=FakeEmbeddings()) for _ in range(2)]
    with TestClient(apps[0]) as one, TestClient(apps[1]) as two:

        def send(i):
            return feedback(
                [one, two][i % 2],
                f"hh.ru:{i}",
                bool(i % 2),
                vacancy={
                    "title": "Python" if i % 2 else "Sales",
                    "description": "Full vacancy description",
                },
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(send, range(8)))
        assert sorted(r["revision"] for r in results) == list(range(1, 9))
        assert db(one, select(func.count()).select_from(MLFeedback)) == 8
        assert json.loads(db(one, select(MLUser.model_json)))["iterations"] == 8
        # Both processes try to apply the same correction: one update, one no-op.
        with ThreadPoolExecutor(max_workers=2) as executor:
            corrections = list(
                executor.map(lambda client: feedback(client, "hh.ru:0", True), [one, two])
            )
        assert sorted(r["updated"] for r in corrections) == [False, True]
        assert [r["revision"] for r in corrections] == [9, 9]
        expected = rank(one)
        db(one, update(MLUser).values(model_json=None))
        assert rank(two) == expected


def test_row_lock_blocks_same_user_but_not_other_users(db_settings):
    acquired, release = threading.Event(), threading.Event()
    with (
        TestClient(create_app(db_settings, embeddings=FakeEmbeddings())) as one,
        TestClient(create_app(db_settings, embeddings=FakeEmbeddings())) as two,
    ):
        rank(one)

        async def hold_lock():
            async with one.app.state.recommender.storage.transaction("alice"):
                acquired.set()
                await anyio.to_thread.run_sync(release.wait, 10)

        holder = one.portal.start_task_soon(hold_lock)
        assert acquired.wait(5)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                blocked = executor.submit(rank, two, "alice")
                other = executor.submit(rank, two, "bob")
                assert other.result(timeout=5)["personalized"] is False
                assert not blocked.done()
                release.set()
                assert blocked.result(timeout=5)["personalized"] is False
        finally:
            release.set()
            holder.result(timeout=5)


def test_source_namespaced_vacancy_ids_do_not_collide(db_settings):
    with TestClient(create_app(db_settings, embeddings=FakeEmbeddings())) as client:
        feedback(
            client, "hh.ru:123", True, vacancy={"title": "Python", "description": "Python APIs"}
        )
        feedback(
            client,
            "career.habr.com:123",
            False,
            vacancy={"title": "Sales", "description": "Sales calls"},
        )
        assert db(client, select(func.count()).select_from(MLFeedback)) == 2
        assert (
            db(client, select(MLFeedback.liked).where(MLFeedback.vacancy_id == "hh.ru:123")) is True
        )


def test_pool_disposed_at_application_shutdown(db_settings):
    app = create_app(db_settings, embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        rank(client)
        pool = app.state.recommender.storage.engine.pool
        assert pool.checkedin() >= 1
    assert pool.checkedin() == 0
