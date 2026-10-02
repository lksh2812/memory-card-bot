"""All SQL lives here. Each method is one short transaction.

The interesting one is `record_answer`, which is how we avoid double scoring:

    UPDATE rounds SET status = 'correct'
     WHERE id = :round_id AND status = 'awaiting'
    RETURNING id

Only the first caller finds the round in 'awaiting' and gets a row back. A second
call for the same round (a duplicate transcript, a retry, a race between two
tasks) matches zero rows, so it never inserts a response or touches the score.
The insert into responses and the score update happen in the same transaction,
so a crash can't leave a scored round without its points or the reverse.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import case, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import selectinload

from app.models import GameSession, Player, Response, Round, RoundStatus, SessionStatus


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ScoreResult:
    applied: bool  # False means this round was already scored; nothing changed
    score: int
    lives: int
    rounds_cleared: int
    status: str


class GameRepository:
    def __init__(self, session_factory: async_sessionmaker):
        self._sf = session_factory

    # ---- players and sessions -------------------------------------------------

    async def create_session(self, player_name: str, lives: int) -> GameSession:
        async with self._sf() as db, db.begin():
            player = await db.scalar(select(Player).where(Player.name == player_name))
            if player is None:
                player = Player(name=player_name)
                db.add(player)
                await db.flush()
            game = GameSession(player_id=player.id, lives=lives)
            db.add(game)
        return await self.get_session(game.id)  # type: ignore[return-value]

    async def get_session(self, session_id: int, with_rounds: bool = False) -> GameSession | None:
        options = [selectinload(GameSession.player)]
        if with_rounds:
            options.append(selectinload(GameSession.rounds).selectinload(Round.response))
        async with self._sf() as db:
            return await db.scalar(
                select(GameSession).where(GameSession.id == session_id).options(*options)
            )

    async def activate_session(self, session_id: int) -> bool:
        """created -> active. Returns False if the session isn't in 'created'."""
        async with self._sf() as db, db.begin():
            result = await db.execute(
                update(GameSession)
                .where(GameSession.id == session_id, GameSession.status == SessionStatus.CREATED)
                .values(status=SessionStatus.ACTIVE, started_at=_now())
            )
            return result.rowcount == 1

    async def end_session(self, session_id: int, status: str, reason: str, won: bool = False) -> bool:
        """Close an open session. Idempotent: a second call is a no-op returning False."""
        async with self._sf() as db, db.begin():
            result = await db.execute(
                update(GameSession)
                .where(GameSession.id == session_id, GameSession.status.in_(SessionStatus.OPEN))
                .values(status=status, end_reason=reason, won=won, ended_at=_now())
            )
            if result.rowcount == 1:
                # Rounds left mid-flight shouldn't look like they're still waiting.
                await db.execute(
                    update(Round)
                    .where(
                        Round.session_id == session_id,
                        Round.status.in_((RoundStatus.PRESENTING, RoundStatus.AWAITING)),
                    )
                    .values(status=RoundStatus.ABANDONED)
                )
            return result.rowcount == 1

    # ---- rounds -----------------------------------------------------------------

    async def create_round(self, session_id: int, round_number: int, cards: list[str]) -> Round:
        async with self._sf() as db, db.begin():
            rnd = Round(session_id=session_id, round_number=round_number, cards=cards, length=len(cards))
            db.add(rnd)
            await db.execute(
                update(GameSession)
                .where(GameSession.id == session_id)
                .values(
                    rounds_played=round_number,
                    best_length=case(
                        (GameSession.best_length < len(cards), len(cards)),
                        else_=GameSession.best_length,
                    ),
                )
            )
        return rnd

    async def mark_round_awaiting(self, round_id: int) -> bool:
        """presenting -> awaiting, once the user has heard every card."""
        async with self._sf() as db, db.begin():
            result = await db.execute(
                update(Round)
                .where(Round.id == round_id, Round.status == RoundStatus.PRESENTING)
                .values(status=RoundStatus.AWAITING)
            )
            return result.rowcount == 1

    async def mark_round_presenting_again(self, round_id: int) -> None:
        """A replay: back to presenting, and count it."""
        async with self._sf() as db, db.begin():
            await db.execute(
                update(Round)
                .where(Round.id == round_id, Round.status.in_((RoundStatus.PRESENTING, RoundStatus.AWAITING)))
                .values(status=RoundStatus.PRESENTING, times_presented=Round.times_presented + 1)
            )

    async def record_answer(
        self,
        round_id: int,
        transcript: str,
        heard: list[str],
        correct: bool,
        points: int,
    ) -> ScoreResult:
        """Score a round exactly once. See the module docstring."""
        new_status = RoundStatus.CORRECT if correct else RoundStatus.INCORRECT
        try:
            async with self._sf() as db, db.begin():
                claimed = await db.scalar(
                    update(Round)
                    .where(Round.id == round_id, Round.status == RoundStatus.AWAITING)
                    .values(status=new_status, evaluated_at=_now())
                    .returning(Round.session_id)
                )
                if claimed is None:
                    return await self._current_score(round_id, applied=False)

                db.add(
                    Response(
                        round_id=round_id,
                        transcript=transcript,
                        heard=heard,
                        is_correct=correct,
                        points=points,
                    )
                )
                values: dict = {"score": GameSession.score + points}
                if correct:
                    values["rounds_cleared"] = GameSession.rounds_cleared + 1
                else:
                    values["lives"] = GameSession.lives - 1
                row = (
                    await db.execute(
                        update(GameSession)
                        .where(GameSession.id == claimed)
                        .values(**values)
                        .returning(
                            GameSession.score,
                            GameSession.lives,
                            GameSession.rounds_cleared,
                            GameSession.status,
                        )
                    )
                ).one()
                return ScoreResult(True, row.score, row.lives, row.rounds_cleared, row.status)
        except IntegrityError:
            # UNIQUE(responses.round_id) fired: someone else scored this round.
            return await self._current_score(round_id, applied=False)

    async def _current_score(self, round_id: int, applied: bool) -> ScoreResult:
        async with self._sf() as db:
            game = await db.scalar(
                select(GameSession).join(Round, Round.session_id == GameSession.id).where(Round.id == round_id)
            )
            return ScoreResult(applied, game.score, game.lives, game.rounds_cleared, game.status)

    # ---- read models --------------------------------------------------------------

    async def best_scores(self, limit: int) -> list[tuple[str, int]]:
        """Best finished score per player, used to (re)build the Redis leaderboard."""
        async with self._sf() as db:
            rows = await db.execute(
                select(Player.name, func.max(GameSession.score).label("best"))
                .join(GameSession, GameSession.player_id == Player.id)
                .where(GameSession.status.in_((SessionStatus.COMPLETED, SessionStatus.ABANDONED)))
                .group_by(Player.name)
                .order_by(desc("best"))
                .limit(limit)
            )
            return [(r.name, r.best) for r in rows]

    async def recent_sessions(self, limit: int) -> list[GameSession]:
        async with self._sf() as db:
            rows = await db.scalars(
                select(GameSession)
                .where(GameSession.ended_at.is_not(None))
                .options(selectinload(GameSession.player))
                .order_by(GameSession.ended_at.desc())
                .limit(limit)
            )
            return list(rows)
