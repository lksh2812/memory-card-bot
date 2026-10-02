import asyncio

from app.models import RoundStatus, SessionStatus

CARDS = ["apple", "river", "tiger"]


async def _awaiting_round(repo):
    game = await repo.create_session("ada", lives=3)
    await repo.activate_session(game.id)
    rnd = await repo.create_round(game.id, 1, CARDS)
    assert await repo.mark_round_awaiting(rnd.id)
    return game, rnd


async def test_same_player_name_reuses_player(repo):
    first = await repo.create_session("ada", lives=3)
    second = await repo.create_session("ada", lives=3)
    assert first.player_id == second.player_id
    assert first.id != second.id


async def test_activate_only_once(repo):
    game = await repo.create_session("ada", lives=3)
    assert await repo.activate_session(game.id)
    assert not await repo.activate_session(game.id)


async def test_correct_answer_is_scored(repo):
    _, rnd = await _awaiting_round(repo)
    result = await repo.record_answer(rnd.id, "apple river tiger", CARDS, True, 30)
    assert result.applied
    assert (result.score, result.rounds_cleared, result.lives) == (30, 1, 3)


async def test_wrong_answer_costs_a_life(repo):
    _, rnd = await _awaiting_round(repo)
    result = await repo.record_answer(rnd.id, "apple", ["apple"], False, 0)
    assert result.applied
    assert (result.score, result.lives) == (0, 2)


async def test_same_answer_twice_is_scored_once(repo):
    _, rnd = await _awaiting_round(repo)
    first = await repo.record_answer(rnd.id, "apple river tiger", CARDS, True, 30)
    second = await repo.record_answer(rnd.id, "apple river tiger", CARDS, True, 30)
    assert first.applied and not second.applied
    assert second.score == 30


async def test_concurrent_duplicates_are_scored_once(repo):
    _, rnd = await _awaiting_round(repo)
    results = await asyncio.gather(
        *[repo.record_answer(rnd.id, "apple river tiger", CARDS, True, 30) for _ in range(10)]
    )
    assert sum(r.applied for r in results) == 1
    game = await repo.get_session(rnd.session_id)
    assert game.score == 30 and game.rounds_cleared == 1


async def test_cannot_score_round_still_being_presented(repo):
    game = await repo.create_session("ada", lives=3)
    rnd = await repo.create_round(game.id, 1, CARDS)
    result = await repo.record_answer(rnd.id, "apple river tiger", CARDS, True, 30)
    assert not result.applied
    assert result.score == 0


async def test_replay_counts_presentations(repo):
    game, rnd = await _awaiting_round(repo)
    await repo.mark_round_presenting_again(rnd.id)
    game = await repo.get_session(game.id, with_rounds=True)
    assert game.rounds[0].times_presented == 2
    assert game.rounds[0].status == RoundStatus.PRESENTING


async def test_end_session_is_idempotent_and_abandons_open_round(repo):
    game, _ = await _awaiting_round(repo)
    assert await repo.end_session(game.id, SessionStatus.ABANDONED, "ended_by_api")
    assert not await repo.end_session(game.id, SessionStatus.COMPLETED, "out_of_lives")
    game = await repo.get_session(game.id, with_rounds=True)
    assert game.status == SessionStatus.ABANDONED
    assert game.rounds[0].status == RoundStatus.ABANDONED


async def test_best_scores_keeps_each_players_best(repo):
    for points in (30, 70):
        _, rnd = await _awaiting_round(repo)
        await repo.record_answer(rnd.id, "x", CARDS, True, points)
        await repo.end_session(rnd.session_id, SessionStatus.COMPLETED, "out_of_lives")
    assert await repo.best_scores(10) == [("ada", 70)]
