"""FastAPI app: REST APIs, WebRTC signaling, and the static frontend.

Endpoints
    POST  /api/sessions                 create a game session for a player
    GET   /api/sessions/{id}            live game state (served from Redis)
    GET   /api/sessions/{id}/rounds     round-by-round history (Postgres)
    POST  /api/sessions/{id}/end        end a game (also stops a live bot)
    GET   /api/leaderboard              best score per player (Redis sorted set)
    GET   /api/scores/recent            latest finished games
    POST  /api/offer?session_id=...     WebRTC offer -> answer; starts the bot
    PATCH /api/offer?session_id=...     trickle ICE candidates
    GET   /                             the web UI

Responses from cached endpoints carry an X-Cache: hit|miss header, handy for
showing the cache working in the demo.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from redis.asyncio import Redis

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from app.bot.pipeline import BotRegistry, run_bot
from app.cache import GameCache
from app.config import settings
from app.db import create_tables, make_engine, make_session_factory
from app.game.engine import GameRules
from app.models import SessionStatus
from app.repository import GameRepository
from app.schemas import CreateSessionRequest, LeaderboardEntry, RecentScore, RoundOut, SessionState
from app.service import GameService

logging.basicConfig(level=logging.INFO)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = make_engine(settings.database_url)
    await create_tables(engine)
    redis = Redis.from_url(settings.redis_url)
    rules = GameRules(start_length=settings.start_length, max_length=settings.max_length, lives=settings.lives)
    app.state.service = GameService(GameRepository(make_session_factory(engine)), GameCache(redis), rules)
    app.state.bots = BotRegistry()
    app.state.webrtc = SmallWebRTCRequestHandler()
    if not settings.deepgram_api_key:
        logger.warning("DEEPGRAM_API_KEY is not set; voice calls will fail to connect to STT/TTS")
    yield
    await app.state.webrtc.close()
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
    service = _service(request)
    bots: BotRegistry = request.app.state.bots
    # Record the end first: the database is the source of truth even if the bot is gone.
    state = await service.finish_session(session_id, SessionStatus.COMPLETED, "ended_by_player")
    if state is None:
        raise HTTPException(404, "session not found")
    # If a bot is live for this session, let it say goodbye and hang up.
    await bots.end_game(session_id, "ended_by_player")
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


# ---- WebRTC signaling --------------------------------------------------------------


@app.post("/api/offer")
async def offer(request: Request, background_tasks: BackgroundTasks, session_id: int = Query(...)):
    """The browser sends its SDP offer; we answer and start a bot for the session."""
    service = _service(request)
    bots: BotRegistry = request.app.state.bots
    body = await request.json()
    webrtc_request = SmallWebRTCRequest.from_dict(body)

    # A new connection (no pc_id) must be for a session that hasn't started yet.
    # Renegotiation of an existing connection reuses the running bot.
    if not webrtc_request.pc_id:
        state, _ = await service.get_state(session_id)
        if state is None:
            raise HTTPException(404, "session not found")
        if state.status != SessionStatus.CREATED or bots.is_running(session_id):
            raise HTTPException(409, "this session has already been played; start a new one")

    async def on_connection(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection, session_id, service, settings, bots)

    return await request.app.state.webrtc.handle_web_request(webrtc_request, on_connection)


@app.patch("/api/offer")
async def ice_candidates(request: Request, session_id: int | None = Query(None)):
    body = await request.json()
    patch = SmallWebRTCPatchRequest(
        pc_id=body["pc_id"],
        candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
    )
    await request.app.state.webrtc.handle_patch_request(patch)
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---- frontend ------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
