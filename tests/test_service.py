import asyncio

import pytest

from app.models import RoundStatus, SessionStatus

pytestmark = pytest.mark.asyncio


async def _started_round(service, length=3):
    state = await service.create_session("lokesh")
    game = await service.start_session(state.session_id)
    rnd = await service.new_round(game, round_number=1, length=length)
    await service.repo.mark_round_awaiting(rnd.id)
    return game, rnd


async def test_create_and_read_state_is_cached(service):
    state = await service.create_session("lokesh")
    assert state.status == SessionStatus.CREATED
    again, hit = await service.get_state(state.session_id)
    assert hit is True
    assert again.player_name == "lokesh"


async def test_start_only_once(service):
    state = await service.create_session("lokesh")
    assert await service.start_session(state.session_id) is not None
    assert await service.start_session(state.session_id) is None


async def test_correct_answer_is_scored(service):
    game, rnd = await _started_round(service)
    result = await service.record_answer(rnd.id, "a b c", rnd.cards, True, 30)
    assert result.applied
    assert result.score == 30
    assert result.rounds_cleared == 1
    assert result.lives == 3


async def test_wrong_answer_costs_a_life(service):
    game, rnd = await _started_round(service)
    result = await service.record_answer(rnd.id, "nope", [], False, 0)
    assert result.applied and result.lives == 2 and result.score == 0


async def test_same_answer_twice_is_scored_once(service):
    game, rnd = await _started_round(service)
    first = await service.record_answer(rnd.id, "x", rnd.cards, True, 30)
    second = await service.record_answer(rnd.id, "x", rnd.cards, True, 30)
    assert first.applied and not second.applied
    assert second.score == 30  # unchanged


async def test_concurrent_duplicates_are_scored_once(service):
    """Ten tasks race to score the same round; exactly one wins."""
    game, rnd = await _started_round(service)
    results = await asyncio.gather(
        *[service.record_answer(rnd.id, "x", rnd.cards, True, 30) for _ in range(10)]
    )
    assert sum(r.applied for r in results) == 1
    final = await service.repo.get_session(game.id)
    assert final.score == 30


async def test_cannot_score_round_still_being_presented(service):
    state = await service.create_session("lokesh")
    game = await service.start_session(state.session_id)
    rnd = await service.new_round(game, 1, 3)  # never marked awaiting
    result = await service.record_answer(rnd.id, "x", rnd.cards, True, 30)
    assert not result.applied


async def test_finish_is_idempotent_and_updates_leaderboard(service):
    game, rnd = await _started_round(service)
    await service.record_answer(rnd.id, "x", rnd.cards, True, 30)
    first = await service.finish_session(game.id, SessionStatus.COMPLETED, "out_of_lives")
    second = await service.finish_session(game.id, SessionStatus.ABANDONED, "disconnected")
    assert first.status == second.status == SessionStatus.COMPLETED
    board, _ = await service.leaderboard(10)
    assert board[0].player_name == "lokesh" and board[0].best_score == 30


async def test_leaderboard_rebuilds_from_db_when_cache_is_cold(service):
    game, rnd = await _started_round(service)
    await service.record_answer(rnd.id, "x", rnd.cards, True, 30)
    await service.finish_session(game.id, SessionStatus.COMPLETED, "done")
    await service.cache._r.flushall()
    board, hit = await service.leaderboard(10)
    assert hit is False and board[0].best_score == 30
    _, hit = await service.leaderboard(10)
    assert hit is True


async def test_leaderboard_keeps_best_score(service):
    for score_points in (50, 20):
        game, rnd = await _started_round(service)
        await service.record_answer(rnd.id, "x", rnd.cards, True, score_points)
        await service.finish_session(game.id, SessionStatus.COMPLETED, "done")
    board, _ = await service.leaderboard(10)
    assert board[0].best_score == 50


async def test_round_cards_hidden_until_evaluated(service):
    game, rnd = await _started_round(service)
    rounds = await service.rounds(game.id)
    assert rounds[0].cards is None
    await service.record_answer(rnd.id, "x", rnd.cards, True, 30)
    rounds = await service.rounds(game.id)
    assert rounds[0].cards == rnd.cards


async def test_ending_mid_round_abandons_open_round(service):
    game, rnd = await _started_round(service)
    await service.finish_session(game.id, SessionStatus.ABANDONED, "disconnected")
    rounds = await service.rounds(game.id)
    assert rounds[0].status == RoundStatus.ABANDONED


async def test_sequences_not_repeated_for_same_player(service):
    state = await service.create_session("lokesh")
    game = await service.start_session(state.session_id)
    seen = set()
    for n in range(1, 30):
        rnd = await service.new_round(game, n, 3)
        sig = "|".join(rnd.cards)
        assert sig not in seen
        seen.add(sig)
