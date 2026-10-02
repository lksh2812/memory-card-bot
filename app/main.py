"""FastAPI app. Creates the database tables on startup; routes are added in later steps."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.db import create_tables, make_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = make_engine(settings.database_url)
    await create_tables(engine)
    yield
    await engine.dispose()


app = FastAPI(title="Memory Card Voice Bot", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
