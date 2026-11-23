"""Async PostgreSQL storage with one transaction/row lock per user's update."""

from contextlib import asynccontextmanager

from asyncpg import PostgresError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from resumeanalysis.models import MLFeedback, MLUser, MLVacancy


class StorageUnavailable(RuntimeError):
    """Redacted database failure safe to expose through the service boundary."""


class UserStorage:
    """Operations bound to the row-locked user; never shared across requests."""

    def __init__(self, session, user):
        self.session, self.user = session, user

    def state(self):
        return {
            "revision": self.user.revision,
            "model_json": self.user.model_json,
            "model_checksum": self.user.model_checksum,
        }

    async def history(self):
        rows = await self.session.scalars(
            select(MLFeedback)
            .where(MLFeedback.user_id == self.user.user_id)
            .order_by(MLFeedback.sequence)
        )
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row):
        if row is None:
            return None
        fields = ["vacancy_id", "title", "description"]
        if isinstance(row, MLFeedback):
            fields += ["liked", "sequence"]
        return {field: getattr(row, field) for field in fields}

    async def feedback(self, vacancy_id):
        row = await self.session.scalar(
            select(MLFeedback).where(
                MLFeedback.user_id == self.user.user_id,
                MLFeedback.vacancy_id == vacancy_id,
            )
        )
        return self._record(row)

    async def vacancy(self, vacancy_id):
        return self._record(await self.session.get(MLVacancy, (self.user.user_id, vacancy_id)))

    async def save_vacancy(self, vacancy_id, vacancy):
        await self.session.execute(
            insert(MLVacancy)
            .values(
                user_id=self.user.user_id,
                vacancy_id=vacancy_id,
                title=vacancy.title,
                description=vacancy.description,
            )
            .on_conflict_do_update(
                index_elements=[MLVacancy.user_id, MLVacancy.vacancy_id],
                set_={"title": vacancy.title, "description": vacancy.description},
            )
        )

    async def save_feedback(self, vacancy_id, vacancy, liked):
        await self.save_vacancy(vacancy_id, vacancy)
        await self.session.execute(
            insert(MLFeedback)
            .values(
                user_id=self.user.user_id,
                vacancy_id=vacancy_id,
                title=vacancy.title,
                description=vacancy.description,
                liked=liked,
            )
            .on_conflict_do_update(
                index_elements=[MLFeedback.user_id, MLFeedback.vacancy_id],
                set_={"title": vacancy.title, "description": vacancy.description, "liked": liked},
            )
        )

    async def save_model(self, state, checksum, revision):
        self.user.model_json = state
        self.user.model_checksum = checksum
        self.user.revision = revision
        await self.session.flush()


class Storage:
    def __init__(self, database_url: str):
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            hide_parameters=True,
            connect_args={
                "timeout": 10,
                "command_timeout": 30,
                "server_settings": {"application_name": "resumeanalysis", "lock_timeout": "30s"},
            },
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        # No connection, DDL, or migrations at construction/startup.

    @asynccontextmanager
    async def transaction(self, user_id):
        try:
            async with self.sessions.begin() as session:
                # The unique insert serializes first-use races; SELECT FOR UPDATE
                # serializes all subsequent mutations across processes/workers.
                await session.execute(
                    insert(MLUser).values(user_id=user_id).on_conflict_do_nothing()
                )
                user = await session.scalar(
                    select(MLUser).where(MLUser.user_id == user_id).with_for_update()
                )
                yield UserStorage(session, user)
        except (SQLAlchemyError, PostgresError, OSError, TimeoutError):
            raise StorageUnavailable(
                "PostgreSQL unavailable; check connectivity and migrations"
            ) from None

    async def close(self):
        await self.engine.dispose()
