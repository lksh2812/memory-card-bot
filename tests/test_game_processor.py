"""Drive MemoryGameProcessor with real Pipecat frames, no audio or API keys needed.

These tests stand in for the rest of the pipeline: we send the frames the user
aggregator and the output transport would send (a finished user turn, bot
stopped speaking, an interruption) and check what the processor says and stores.
"""

import random

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    InterruptionFrame,
    LLMContextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test

from app.bot.frames import StartGameFrame
from app.bot.game_processor import MemoryGameProcessor, Phase
from app.bot.host import HostVoice
from app.game.cards import generate_sequence

pytestmark = pytest.mark.asyncio

SEED = 42
WAIT = SleepFrame(sleep=0.3)


def user_said(text: str) -> LLMContextFrame:
    return LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": text}]))


async def make_game(service, rounds_to_predict=6):
    state = await service.create_session("lokesh")
    service._rng = random.Random(SEED)
    # Same seed, same sequences: lets the test know the cards in advance.
    rng = random.Random(SEED)
    lengths = [3, 4, 5, 6, 7, 8][:rounds_to_predict]
    predicted = [generate_sequence(n, rng) for n in lengths]
    game = MemoryGameProcessor(
        service=service,
        session_id=state.session_id,
        host=HostVoice(llm=None),  # template lines, no network
        rules=service.rules,
        nudge_after=60,
        give_up_after=60,
    )
    return state.session_id, game, predicted


def spoken(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, TTSSpeakFrame)]


async def test_correct_answer_scores_and_moves_to_harder_round(service):
    sid, game, cards = await make_game(service)
    down, _ = await run_test(
        game,
        frames_to_send=[
            StartGameFrame(), WAIT,
            BotStoppedSpeakingFrame(), WAIT,
            user_said("um " + ", ".join(cards[0])), WAIT,
        ],
    )
    lines = spoken(down)
    assert "Round 1. 3 cards." in lines[0]
    assert all(c.capitalize() in lines[0] for c in cards[0])
    assert "Round 2. 4 cards." in lines[1]
    row = await service.repo.get_session(sid)
    assert row.score == 30 and row.rounds_cleared == 1 and row.lives == 3


async def test_wrong_answer_reveals_cards_and_costs_a_life(service):
    sid, game, cards = await make_game(service)
    down, _ = await run_test(
        game,
        frames_to_send=[
            StartGameFrame(), WAIT,
            BotStoppedSpeakingFrame(), WAIT,
            user_said(" ".join(reversed(cards[0]))), WAIT,
        ],
    )
    feedback = spoken(down)[1]
    assert "The cards were" in feedback and "2 lives left" in feedback
    row = await service.repo.get_session(sid)
    assert row.lives == 2 and row.score == 0


async def test_interrupting_the_cards_replays_them_without_scoring(service):
    sid, game, cards = await make_game(service)
    down, _ = await run_test(
        game,
        frames_to_send=[
            StartGameFrame(), WAIT,
            InterruptionFrame(), WAIT,  # user talks over the cards
            BotStoppedSpeakingFrame(), WAIT,
            user_said("wait wait sorry"), WAIT,  # their turn ends
        ],
    )
    lines = spoken(down)
    assert "again from the top" in lines[1]
    assert all(c.capitalize() in lines[1] for c in cards[0])  # same cards replayed
    rounds = await service.rounds(sid)
    assert len(rounds) == 1 and rounds[0].times_presented == 2
    assert rounds[0].status == "presenting"  # nothing scored


async def test_answer_during_presentation_is_ignored(service):
    """A turn that arrives before the cards finished playing must not be scored."""
    sid, game, cards = await make_game(service)
    await run_test(
        game,
        frames_to_send=[StartGameFrame(), WAIT, user_said(" ".join(cards[0])), WAIT],
    )
    row = await service.repo.get_session(sid)
    assert row.score == 0
    assert game._phase == Phase.PRESENTING


async def test_duplicate_answer_is_not_double_scored(service):
    sid, game, cards = await make_game(service)
    answer = " ".join(cards[0])
    await run_test(
        game,
        frames_to_send=[
            StartGameFrame(), WAIT,
            BotStoppedSpeakingFrame(), WAIT,
            user_said(answer), user_said(answer), WAIT,
        ],
    )
    row = await service.repo.get_session(sid)
    assert row.score == 30


async def test_repeat_request_is_allowed_once_per_round(service):
    sid, game, cards = await make_game(service)
    down, _ = await run_test(
        game,
        frames_to_send=[
            StartGameFrame(), WAIT,
            BotStoppedSpeakingFrame(), WAIT,
            user_said("can you repeat that"), WAIT,
            BotStoppedSpeakingFrame(), WAIT,
            user_said("say that again"), WAIT,
        ],
    )
    lines = spoken(down)
    assert "one more time" in lines[1]
    assert "only repeat once" in lines[2]


async def test_three_misses_end_the_game_and_the_call(service):
    sid, game, cards = await make_game(service)
    frames = [StartGameFrame(), WAIT]
    for _ in range(3):
        frames += [BotStoppedSpeakingFrame(), WAIT, user_said("rocket rocket"), WAIT]
    frames += [BotStoppedSpeakingFrame(), WAIT]  # final line finished playing
    down, _ = await run_test(game, frames_to_send=frames)
    assert "Game over" in spoken(down)[-1] or "end of the game" in spoken(down)[-1]
    assert any(isinstance(f, EndWorkerFrame) for f in down)
    row = await service.repo.get_session(sid)
    assert row.status == "completed" and row.end_reason == "out_of_lives" and row.lives == 0


async def test_quit_command_ends_game(service):
    sid, game, cards = await make_game(service)
    await run_test(
        game,
        frames_to_send=[
            StartGameFrame(), WAIT,
            BotStoppedSpeakingFrame(), WAIT,
            user_said("I want to quit"), WAIT,
        ],
    )
    row = await service.repo.get_session(sid)
    assert row.status == "completed" and row.end_reason == "player_quit"


async def test_second_start_signal_does_not_restart_or_end_the_game(service):
    sid, game, cards = await make_game(service)
    down, _ = await run_test(
        game,
        frames_to_send=[StartGameFrame(), WAIT, StartGameFrame(), WAIT],
    )
    assert len(spoken(down)) == 1
    row = await service.repo.get_session(sid)
    assert row.status == "active" and row.rounds_played == 1


async def test_live_state_is_published_to_cache(service):
    sid, game, cards = await make_game(service)
    await run_test(
        game,
        frames_to_send=[StartGameFrame(), WAIT, BotStoppedSpeakingFrame(), WAIT],
    )
    state, hit = await service.get_state(sid)
    assert hit and state.phase == "listening" and state.round_number == 1
    assert state.sequence_length == 3
