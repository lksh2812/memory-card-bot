"""Redis cache. Three uses, each with a different pattern:

1. Live session state (write-through)
   The bot writes a small JSON snapshot on every state change; GET /sessions/{id}
   reads it without touching Postgres. The UI can poll this cheaply, and it is the
   hottest read path in the system. Key: mcb:session:{id}, TTL 2h.

2. Leaderboard (sorted set, cache-aside with warm-up)
   ZADD with GT keeps each player's best score; reads are ZREVRANGE, O(log n).
   If Redis was flushed, the first read rebuilds it from Postgres.
   Key: mcb:leaderboard.

3. Recently used sequences (capped list per player)
   The game avoids giving a player the exact sequence they saw recently.
   Key: mcb:player:{id}:recent, last 50 entries, TTL 7d.

Redis is never the source of truth. Every method swallows Redis errors and
returns a miss, so the game keeps working (slower) if Redis is down.
"""

import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

SESSION_TTL = 2 * 60 * 60
RECENT_SEQ_TTL = 7 * 24 * 60 * 60
RECENT_SEQ_MAX = 50
RECENT_SCORES_TTL = 30

LEADERBOARD_KEY = "mcb:leaderboard"
LEADERBOARD_WARM_KEY = "mcb:leaderboard:warm"
RECENT_SCORES_KEY = "mcb:recent_scores"


def _session_key(session_id: int) -> str:
    return f"mcb:session:{session_id}"


def _recent_seq_key(player_id: int) -> str:
    return f"mcb:player:{player_id}:recent"


class GameCache:
    def __init__(self, redis: Redis):
        self._r = redis

    # ---- 1. live session state --------------------------------------------------

    async def put_session_state(self, session_id: int, state: dict[str, Any]) -> None:
        try:
            await self._r.set(_session_key(session_id), json.dumps(state, default=str), ex=SESSION_TTL)
        except RedisError as e:
            log.warning("cache write failed for session %s: %s", session_id, e)

    async def get_session_state(self, session_id: int) -> dict[str, Any] | None:
        try:
            raw = await self._r.get(_session_key(session_id))
        except RedisError as e:
            log.warning("cache read failed for session %s: %s", session_id, e)
            return None
        return json.loads(raw) if raw else None

    # ---- 2. leaderboard -----------------------------------------------------------

    async def submit_score(self, player_name: str, score: int) -> None:
        try:
            # GT: only overwrite when the new score is higher (keeps best score).
            await self._r.zadd(LEADERBOARD_KEY, {player_name: score}, gt=True)
            await self._r.delete(RECENT_SCORES_KEY)
        except RedisError as e:
            log.warning("leaderboard update failed: %s", e)

    async def leaderboard(self, limit: int) -> list[tuple[str, int]] | None:
        """Top scores, or None when the cache is cold and must be rebuilt."""
        try:
            if not await self._r.exists(LEADERBOARD_WARM_KEY):
                return None
            rows = await self._r.zrevrange(LEADERBOARD_KEY, 0, limit - 1, withscores=True)
        except RedisError as e:
            log.warning("leaderboard read failed: %s", e)
            return None
        return [(name.decode() if isinstance(name, bytes) else name, int(score)) for name, score in rows]

    async def warm_leaderboard(self, rows: list[tuple[str, int]]) -> None:
        try:
            async with self._r.pipeline(transaction=True) as pipe:
                if rows:
                    pipe.zadd(LEADERBOARD_KEY, dict(rows), gt=True)
                pipe.set(LEADERBOARD_WARM_KEY, "1")
                await pipe.execute()
        except RedisError as e:
            log.warning("leaderboard warm-up failed: %s", e)

    # ---- recent scores (short TTL read cache) -------------------------------------

    async def get_recent_scores(self) -> list[dict] | None:
        try:
            raw = await self._r.get(RECENT_SCORES_KEY)
        except RedisError:
            return None
        return json.loads(raw) if raw else None

    async def put_recent_scores(self, rows: list[dict]) -> None:
        try:
            await self._r.set(RECENT_SCORES_KEY, json.dumps(rows, default=str), ex=RECENT_SCORES_TTL)
        except RedisError:
            pass

    # ---- 3. recently used sequences --------------------------------------------------

    async def recent_sequences(self, player_id: int) -> set[str]:
        try:
            items = await self._r.lrange(_recent_seq_key(player_id), 0, -1)
        except RedisError:
            return set()
        return {i.decode() if isinstance(i, bytes) else i for i in items}

    async def remember_sequence(self, player_id: int, signature: str) -> None:
        key = _recent_seq_key(player_id)
        try:
            async with self._r.pipeline(transaction=True) as pipe:
                pipe.lpush(key, signature)
                pipe.ltrim(key, 0, RECENT_SEQ_MAX - 1)
                pipe.expire(key, RECENT_SEQ_TTL)
                await pipe.execute()
        except RedisError as e:
            log.warning("recent sequence write failed: %s", e)
