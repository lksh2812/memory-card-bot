"""Game rules as a small, pure state machine.

The voice bot and the database both follow these rules, but the rules
themselves know nothing about either. That keeps difficulty and scoring easy to
test and easy to change in the next round of the interview.

Lifecycle of a game:

    ACTIVE --(correct)--> next round, sequence one card longer
           --(wrong, lives left)--> same length again with new cards
           --(wrong, no lives)--> COMPLETED (lost)
           --(cleared max length)--> COMPLETED (won)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameRules:
    start_length: int = 3
    max_length: int = 10
    lives: int = 3
    points_per_card: int = 10

    def length_for(self, rounds_cleared: int) -> int:
        """Difficulty curve: one more card for every round cleared."""
        return min(self.start_length + rounds_cleared, self.max_length)

    def points_for(self, length: int) -> int:
        return length * self.points_per_card


@dataclass
class GameState:
    """Mutable game state for one session. Mirrors the game_sessions row."""

    rules: GameRules
    score: int = 0
    lives: int = 0
    rounds_cleared: int = 0
    round_number: int = 0
    finished: bool = False
    won: bool = False

    @classmethod
    def new(cls, rules: GameRules) -> "GameState":
        return cls(rules=rules, lives=rules.lives)

    @property
    def next_length(self) -> int:
        return self.rules.length_for(self.rounds_cleared)


@dataclass(frozen=True)
class RoundOutcome:
    correct: bool
    points: int
    score: int
    lives: int
    rounds_cleared: int
    game_over: bool
    won: bool


def apply_result(state: GameState, length: int, correct: bool) -> RoundOutcome:
    """Apply one evaluated answer to the game state and return what happened."""
    if state.finished:
        raise ValueError("game already finished")
    points = 0
    if correct:
        points = state.rules.points_for(length)
        state.score += points
        state.rounds_cleared += 1
        if length >= state.rules.max_length:
            state.finished = True
            state.won = True
    else:
        state.lives -= 1
        if state.lives <= 0:
            state.finished = True
    return RoundOutcome(
        correct=correct,
        points=points,
        score=state.score,
        lives=state.lives,
        rounds_cleared=state.rounds_cleared,
        game_over=state.finished,
        won=state.won,
    )
