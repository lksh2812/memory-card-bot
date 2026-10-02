"""Use cases shared by the REST API and the voice bot.

Both sides go through this class so there is exactly one implementation of
"start a game", "score an answer" and "end a game", and the cache is updated
in one place.
"""

import random
from datetime import UTC, datetime

from app.cache import GameCache
from app.game.cards import generate_sequence, sequence_signature
from app.game.engine import GameRules
from app.models import GameSession, Round, RoundStatus, SessionStatus
from app.repository import GameRepository, ScoreResult
from app.schemas import LastResult, LeaderboardEntry, RecentScore, RoundOut, SessionState


class GameService:
    def __init__(self, repo: GameRepository, cache: GameCache, rules: GameRules):
        self.repo = repo
        self.cache = cache
        self.rules = rules
        self._rng = random.Random()

    # ---- state snapshots ---------------------------------------------------------

    def snapshot_from_row(self, game: GameSession, phase: str | None = None) -> SessionState:
        if phase is None:
            phase = {
                SessionStatus.CREATED: "waiting_for_call",
                SessionStatus.ACTIVE: "playing",
            }.get(game.status, "finished")
        return SessionState(
            session_id=game.id,
            player_name=game.player.name,
            status=game.status,
            phase=phase,
            round_number=game.rounds_played,
            sequence_length=self.rules.length_for(game.rounds_cleared),
            score=game.score,
            lives=game.lives,
            max_lives=self.rules.lives,
            rounds_cleared=game.rounds_cleared,
            best_length=game.best_length,
            won=game.won,
            end_reason=game.end_reason,
            updated_at=datetime.now(UTC),
        )

    async def publish(self, state: SessionState) -> None:
        state.updated_at = datetime.now(UTC)
        await self.cache.put_session_state(state.session_id, state.model_dump(mode="json"))

    async def get_state(self, session_id: int) -> tuple[SessionState | None, bool]:
        """Cache-first read. Returns (state, cache_hit)."""
        cached = await self.cache.get_session_state(session_id)
        if cached:
            return SessionState(**cached), True
        game = await self.repo.get_session(session_id)
        if game is None:
            return None, False
        state = self.snapshot_from_row(game)
        await self.publish(state)
        return state, False

    # ---- lifecycle ----------------------------------------------------------------

    async def create_session(self, player_name: str) -> SessionState:
        game = await self.repo.create_session(player_name.strip(), lives=self.rules.lives)
        state = self.snapshot_from_row(game)
        await self.publish(state)
        return state

    async def start_session(self, session_id: int) -> GameSession | None:
        """Called by the bot when the call connects. Returns None if not startable."""
        if not await self.repo.activate_session(session_id):
            return None
        return await self.repo.get_session(session_id)

    async def new_round(self, game: GameSession, round_number: int, length: int) -> Round:
        recent = await self.cache.recent_sequences(game.player_id)
        cards = generate_sequence(length, self._rng, avoid=recent)
        rnd = await self.repo.create_round(game.id, round_number, cards)
        await self.cache.remember_sequence(game.player_id, sequence_signature(cards))
        return rnd

    async def record_answer(
        self, round_id: int, transcript: str, heard: list[str], correct: bool, points: int
    ) -> ScoreResult:
        return await self.repo.record_answer(round_id, transcript, heard, correct, points)

    async def finish_session(
        self, session_id: int, status: str, reason: str, won: bool = False
    ) -> SessionState | None:
        """End a game (idempotent) and refresh the cache and leaderboard."""
        changed = await self.repo.end_session(session_id, status, reason, won)
        game = await self.repo.get_session(session_id)
        if game is None:
            return None
        state = self.snapshot_from_row(game)
        cached = await self.cache.get_session_state(session_id)
        if cached and cached.get("last_result"):
            state.last_result = LastResult(**cached["last_result"])
        await self.publish(state)
        if changed:
            await self.cache.submit_score(game.player.name, game.score)
        return state

    # ---- read models -----------------------------------------------------------------

    async def rounds(self, session_id: int) -> list[RoundOut] | None:
        game = await self.repo.get_session(session_id, with_rounds=True)
        if game is None:
            return None
        out = []
        for r in game.rounds:
            revealed = r.status in (RoundStatus.CORRECT, RoundStatus.INCORRECT, RoundStatus.ABANDONED)
            out.append(
                RoundOut(
                    round_number=r.round_number,
                    length=r.length,
                    status=r.status,
                    times_presented=r.times_presented,
                    cards=r.cards if revealed else None,
                    transcript=r.response.transcript if r.response else None,
                    heard=r.response.heard if r.response else None,
                    points=r.response.points if r.response else 0,
                )
            )
        return out

    async def leaderboard(self, limit: int) -> tuple[list[LeaderboardEntry], bool]:
        rows = await self.cache.leaderboard(limit)
        hit = rows is not None
        if rows is None:
            # Cold cache: rebuild from Postgres, then serve.
            rows = await self.repo.best_scores(limit=1000)
            await self.cache.warm_leaderboard(rows)
            rows = rows[:limit]
        return [LeaderboardEntry(rank=i + 1, player_name=n, best_score=s) for i, (n, s) in enumerate(rows)], hit

    async def recent_scores(self, limit: int) -> tuple[list[RecentScore], bool]:
        cached = await self.cache.get_recent_scores()
        if cached is not None:
            return [RecentScore(**r) for r in cached][:limit], True
        sessions = await self.repo.recent_sessions(limit=50)
        rows = [
            RecentScore(
                session_id=s.id,
                player_name=s.player.name,
                score=s.score,
                rounds_cleared=s.rounds_cleared,
                best_length=s.best_length,
                status=s.status,
                won=s.won,
                ended_at=s.ended_at,
            )
            for s in sessions
        ]
        await self.cache.put_recent_scores([r.model_dump(mode="json") for r in rows])
        return rows[:limit], False
