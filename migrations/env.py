"""Async migrations with a separate version table and ML-only autogeneration."""

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from resumeanalysis.config import Settings
from resumeanalysis.models import Base


def include_name(name, type_, parent_names):
    # Never propose dropping ApplyPilot's tables or its alembic_version table.
    return type_ != "table" or name in Base.metadata.tables


def configure(**kwargs):
    context.configure(
        target_metadata=Base.metadata,
        version_table="ml_alembic_version",
        include_name=include_name,
        compare_type=True,
        **kwargs,
    )
    with context.begin_transaction():
        context.run_migrations()


async def online(url):
    engine = create_async_engine(url, poolclass=NullPool, hide_parameters=True)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(lambda conn: configure(connection=conn))
    finally:
        await engine.dispose()


url = Settings.from_env().database_url.get_secret_value()
if context.is_offline_mode():
    configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
else:
    asyncio.run(online(url))
