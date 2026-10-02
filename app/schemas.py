"""Response models for the REST API."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    player_name: str = Field(min_length=1, max_length=64)


class LastResult(BaseModel):
    round_number: int
    correct: bool
    expected: list[str]
    heard: list[str]
    points: int


class SessionState(BaseModel):
    """What the UI shows. Served from Redis while a game is live.

    Note what's missing: the cards of the round in progress. Exposing them here
    would let anyone with devtools open read the answer.
    """

    session_id: int
    player_name: str
    status: str
    phase: str
    round_number: int = 0
    sequence_length: int = 0
    score: int = 0
    lives: int = 0
    max_lives: int = 0
    rounds_cleared: int = 0
    best_length: int = 0
    won: bool = False
    end_reason: str | None = None
    last_result: LastResult | None = None
    updated_at: datetime | None = None


class RoundOut(BaseModel):
    round_number: int
    length: int
    status: str
    times_presented: int
    cards: list[str] | None  # hidden until the round is evaluated
    transcript: str | None = None
    heard: list[str] | None = None
    points: int = 0


class LeaderboardEntry(BaseModel):
    rank: int
    player_name: str
    best_score: int


class RecentScore(BaseModel):
    session_id: int
    player_name: str
    score: int
    rounds_cleared: int
    best_length: int
    status: str
    won: bool
    ended_at: datetime | None
