"""Async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.models import Base


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    # expire_on_commit=False so returned ORM objects stay readable after commit.
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_tables(engine: AsyncEngine) -> None:
    """Create tables on startup. Good enough for an assignment; production would use Alembic."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
