"""FastAPI app: REST APIs for sessions and scores.

Endpoints
    POST  /api/sessions                 create a game session for a player
    GET   /api/sessions/{id}            live game state (served from Redis)
    GET   /api/sessions/{id}/rounds     round-by-round history (Postgres)
    POST  /api/sessions/{id}/end        end a game
    GET   /api/leaderboard              best score per player (Redis sorted set)
    GET   /api/scores/recent            latest finished games

Responses from cached endpoints carry an X-Cache: hit|miss header, handy for
showing the cache working in the demo.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from redis.asyncio import Redis

from app.cache import GameCache
from app.config import settings
from app.db import create_tables, make_engine, make_session_factory
from app.game.engine import GameRules
from app.models import SessionStatus
from app.repository import GameRepository
from app.schemas import CreateSessionRequest, LeaderboardEntry, RecentScore, RoundOut, SessionState
from app.service import GameService

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = make_engine(settings.database_url)
    await create_tables(engine)
    redis = Redis.from_url(settings.redis_url)
    rules = GameRules(start_length=settings.start_length, max_length=settings.max_length, lives=settings.lives)
    app.state.service = GameService(GameRepository(make_session_factory(engine)), GameCache(redis), rules)
    yield
    await redis.aclose()
    await engine.dispose()


app = FastAPI(title="Memory Card Voice Bot", lifespan=lifespan)


def _service(request: Request) -> GameService:
    return request.app.state.service


# ---- sessions --------------------------------------------------------------------


@app.post("/api/sessions", response_model=SessionState, status_code=201)
async def create_session(body: CreateSessionRequest, request: Request):
    return await _service(request).create_session(body.player_name)


@app.get("/api/sessions/{session_id}", response_model=SessionState)
async def get_session(session_id: int, request: Request, response: Response):
    state, hit = await _service(request).get_state(session_id)
    if state is None:
        raise HTTPException(404, "session not found")
    response.headers["X-Cache"] = "hit" if hit else "miss"
    return state


@app.get("/api/sessions/{session_id}/rounds", response_model=list[RoundOut])
async def get_rounds(session_id: int, request: Request):
    rounds = await _service(request).rounds(session_id)
    if rounds is None:
        raise HTTPException(404, "session not found")
    return rounds


@app.post("/api/sessions/{session_id}/end", response_model=SessionState)
async def end_session(session_id: int, request: Request):
    state = await _service(request).finish_session(session_id, SessionStatus.COMPLETED, "ended_by_player")
    if state is None:
        raise HTTPException(404, "session not found")
    return state


# ---- scores ----------------------------------------------------------------------


@app.get("/api/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard(request: Request, response: Response, limit: int = Query(10, ge=1, le=100)):
    rows, hit = await _service(request).leaderboard(limit)
    response.headers["X-Cache"] = "hit" if hit else "miss"
    return rows


@app.get("/api/scores/recent", response_model=list[RecentScore])
async def recent_scores(request: Request, response: Response, limit: int = Query(10, ge=1, le=50)):
    rows, hit = await _service(request).recent_scores(limit)
    response.headers["X-Cache"] = "hit" if hit else "miss"
    return rows


@app.get("/health")
async def health():
    return {"status": "ok"}
