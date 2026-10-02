import pytest
from fakeredis import FakeAsyncRedis
from httpx import ASGITransport, AsyncClient

from app.bot.pipeline import BotRegistry
from app.cache import GameCache
from app.db import make_session_factory
from app.game.engine import GameRules
from app.main import app
from app.repository import GameRepository
from app.service import GameService
from tests.conftest import make_test_engine

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def client(tmp_path):
    engine = await make_test_engine(tmp_path, "api.db")
    redis = FakeAsyncRedis()
    app.state.service = GameService(GameRepository(make_session_factory(engine)), GameCache(redis), GameRules())
    app.state.bots = BotRegistry()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await redis.aclose()
    await engine.dispose()


async def test_session_lifecycle(client):
    r = await client.post("/api/sessions", json={"player_name": "lokesh"})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["status"] == "created"

    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200 and r.headers["x-cache"] == "hit"

    r = await client.post(f"/api/sessions/{sid}/end")
    assert r.json()["status"] == "completed"
    assert r.json()["end_reason"] == "ended_by_player"

    r = await client.get(f"/api/sessions/{sid}/rounds")
    assert r.status_code == 200 and r.json() == []


async def test_ids_are_database_integers(client):
    first = (await client.post("/api/sessions", json={"player_name": "a"})).json()["session_id"]
    second = (await client.post("/api/sessions", json={"player_name": "b"})).json()["session_id"]
    assert isinstance(first, int) and second > first


async def test_validation_and_404s(client):
    assert (await client.post("/api/sessions", json={"player_name": ""})).status_code == 422
    assert (await client.get("/api/sessions/not-a-number")).status_code == 422
    assert (await client.get("/api/sessions/999999")).status_code == 404
    assert (await client.post("/api/sessions/999999/end")).status_code == 404


async def test_leaderboard_and_recent(client):
    sid = (await client.post("/api/sessions", json={"player_name": "lokesh"})).json()["session_id"]
    await client.post(f"/api/sessions/{sid}/end")
    r = await client.get("/api/leaderboard")
    assert r.status_code == 200 and r.json()[0]["player_name"] == "lokesh"
    r = await client.get("/api/scores/recent")
    assert r.json()[0]["session_id"] == sid
    assert r.headers["x-cache"] == "miss"
    r = await client.get("/api/scores/recent")
    assert r.headers["x-cache"] == "hit"


async def test_offer_rejects_finished_session(client):
    sid = (await client.post("/api/sessions", json={"player_name": "lokesh"})).json()["session_id"]
    await client.post(f"/api/sessions/{sid}/end")
    r = await client.post(f"/api/offer?session_id={sid}", json={"sdp": "x", "type": "offer"})
    assert r.status_code == 409
