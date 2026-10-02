"""Test fixtures.

By default tests use SQLite, so they run anywhere. Set TEST_DATABASE_URL to run
the same tests against real Postgres, e.g.

    TEST_DATABASE_URL=postgresql+asyncpg://memory:memory@localhost:5432/memory_game pytest
"""

import os

import pytest_asyncio

from app.db import create_tables, make_engine, make_session_factory
from app.models import Base
from app.repository import GameRepository


async def make_test_engine(tmp_path, name: str):
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        engine = make_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    else:
        # A file DB so concurrent sessions see each other's commits like Postgres would.
        engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / name}")
    await create_tables(engine)
    return engine


@pytest_asyncio.fixture
async def repo(tmp_path):
    engine = await make_test_engine(tmp_path, "test.db")
    yield GameRepository(make_session_factory(engine))
    await engine.dispose()
