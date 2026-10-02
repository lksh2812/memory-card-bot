from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from app.cache import GameCache


async def test_session_state_round_trip(cache):
    assert await cache.get_session_state(1) is None
    await cache.put_session_state(1, {"phase": "listening", "score": 30})
    assert await cache.get_session_state(1) == {"phase": "listening", "score": 30}


async def test_leaderboard_is_cold_until_warmed(cache):
    assert await cache.leaderboard(10) is None
    await cache.warm_leaderboard([("ada", 70), ("bob", 40)])
    assert await cache.leaderboard(10) == [("ada", 70), ("bob", 40)]


async def test_leaderboard_keeps_best_score(cache):
    await cache.warm_leaderboard([])
    await cache.submit_score("ada", 70)
    await cache.submit_score("ada", 30)
    assert await cache.leaderboard(10) == [("ada", 70)]


async def test_new_score_invalidates_recent_scores(cache):
    await cache.put_recent_scores([{"player_name": "ada", "score": 30}])
    assert await cache.get_recent_scores() is not None
    await cache.submit_score("bob", 40)
    assert await cache.get_recent_scores() is None


async def test_recent_sequences_are_capped(cache):
    for i in range(60):
        await cache.remember_sequence(7, f"seq{i}")
    recent = await cache.recent_sequences(7)
    assert len(recent) == 50
    assert "seq59" in recent and "seq0" not in recent


async def test_redis_down_is_a_miss_not_a_crash():
    down = GameCache(Redis(port=1, socket_connect_timeout=0.1, retry=Retry(NoBackoff(), 0)))
    await down.put_session_state(1, {"phase": "listening"})
    assert await down.get_session_state(1) is None
    assert await down.leaderboard(10) is None
    assert await down.recent_sequences(7) == set()
    await down.remember_sequence(7, "a|b|c")
