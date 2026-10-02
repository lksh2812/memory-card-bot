"""MemoryGameProcessor: the custom Pipecat frame processor that runs the game.

Where it sits:

    transport.input -> STT -> user aggregator -> [MemoryGameProcessor] -> TTS -> transport.output

The user aggregator does turn detection (VAD + our sequence-aware strategy) and,
when the user finishes a turn, pushes an LLMContextFrame downstream. Normally an
LLM consumes that frame. Here this processor consumes it instead: it reads what
the user said, validates it in plain Python, writes the result to the database,
and pushes a TTSSpeakFrame with exactly what the bot should say. The LLM is only
called from HostVoice for flavor lines.

Phases:

    CONNECTING  -> waiting for the browser to be ready
    PRESENTING  -> bot is reading the cards
    AWAITING    -> cards fully heard, waiting for the answer
    EVALUATING  -> scoring the answer (DB write + host line)
    FINISHED    -> game over, final line playing, then the pipeline ends

Interruption rules:
* User talks over the cards (PRESENTING) -> the round is unfair, so we don't
  evaluate anything. After their turn ends we replay the same cards.
* User talks over anything else -> Pipecat already stopped the audio; nothing
  else to do.
* Short backchannels ("okay", "hmm") never count as interruptions because the
  turn start strategy needs at least two words while the bot is speaking.
"""

import asyncio
from enum import Enum

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from app.bot.frames import EndGameFrame, StartGameFrame
from app.bot.host import HostVoice
from app.game.engine import GameRules, GameState, apply_result
from app.game.matching import Evaluation, detect_intent, evaluate
from app.models import GameSession, Round, SessionStatus
from app.schemas import LastResult, SessionState
from app.service import GameService


class Phase(str, Enum):
    CONNECTING = "connecting"
    PRESENTING = "presenting"
    AWAITING = "listening"
    EVALUATING = "evaluating"
    FINISHED = "finished"


def _say_cards(cards: list[str]) -> str:
    # Full stops make the TTS pause between cards, which helps the listener.
    return ". ".join(card.capitalize() for card in cards) + "."


def _describe_mistake(ev: Evaluation) -> str:
    if "missing_cards" in ev.notes:
        return f"they stopped after {len(ev.heard)} of {len(ev.expected)} cards"
    if "extra_cards" in ev.notes:
        return "they added an extra card at the end"
    if "wrong_order" in ev.notes:
        return "they had the right cards in the wrong order"
    if not ev.heard:
        return "they didn't say any of the cards"
    return f"the first {ev.correct_prefix} were right, then a wrong card"


def _latest_user_text(context: LLMContext) -> str:
    """The text of the user's most recent turn (it may span several messages)."""
    parts: list[str] = []
    for message in reversed(context.messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            break
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return " ".join(reversed(parts)).strip()


class MemoryGameProcessor(FrameProcessor):
    def __init__(
        self,
        service: GameService,
        session_id: int,
        host: HostVoice,
        rules: GameRules,
        nudge_after: float = 12.0,
        give_up_after: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._service = service
        self._session_id = session_id
        self._host = host
        self._rules = rules
        self._nudge_after = nudge_after
        self._give_up_after = give_up_after

        self._phase = Phase.CONNECTING
        self._game: GameSession | None = None
        self._state = GameState.new(rules)
        self._player = "friend"
        self._round: Round | None = None
        self._round_number = 0
        self._best_length = 0
        self._presentation_interrupted = False
        self._repeat_used = False
        self._no_card_turns = 0
        self._last_result: LastResult | None = None
        self._timer_task: asyncio.Task | None = None

    # ---- read by the turn strategy ---------------------------------------------------

    def expected_card_count(self) -> int | None:
        """How many cards the user should say right now, if we're waiting for an answer."""
        if self._phase == Phase.AWAITING and self._round is not None:
            return self._round.length
        return None

    # ---- frame routing ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartGameFrame):
            # Only the first one counts; a reconnecting client can signal ready twice.
            if self._phase == Phase.CONNECTING:
                await self._start_game()
            return
        if isinstance(frame, EndGameFrame):
            await self._end_early(frame.reason)
            return
        if isinstance(frame, LLMContextFrame):
            # A completed user turn. Consumed here; there is no LLM downstream.
            if not frame.speculation:
                await self._on_user_turn(_latest_user_text(frame.context))
            return

        if isinstance(frame, InterruptionFrame):
            self._on_interruption()
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._cancel_timer()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._on_bot_stopped_speaking()

        # Everything else (audio, interruptions, speaking events) passes through.
        await self.push_frame(frame, direction)

    # ---- game flow ------------------------------------------------------------------------

    async def _start_game(self):
        game = await self._service.start_session(self._session_id)
        if game is None:
            logger.warning(f"session {self._session_id} is not startable (already started or ended)")
            await self._speak("This game has already finished. Start a new one from the page.")
            self._phase = Phase.FINISHED
            return
        self._game = game
        self._player = game.player.name
        intro = await self._host.intro(self._player)
        await self._present_new_round(prefix=intro)

    async def _present_new_round(self, prefix: str):
        assert self._game is not None
        self._round_number += 1
        length = self._state.next_length
        self._round = await self._service.new_round(self._game, self._round_number, length)
        self._best_length = max(self._best_length, length)
        self._repeat_used = False
        self._no_card_turns = 0
        logger.info(f"round {self._round_number}: {self._round.cards}")
        await self._present(f"{prefix} Round {self._round_number}. {length} cards.")

    async def _present(self, lead_in: str):
        """Speak a lead-in followed by the cards, as one utterance."""
        assert self._round is not None
        self._phase = Phase.PRESENTING
        self._presentation_interrupted = False
        await self._cancel_timer()
        await self._publish()
        await self._speak(f"{lead_in} {_say_cards(self._round.cards)} Your turn!")

    def _on_interruption(self):
        if self._phase == Phase.PRESENTING and not self._presentation_interrupted:
            logger.info("user interrupted while the cards were being read; will replay")
            self._presentation_interrupted = True

    async def _on_bot_stopped_speaking(self):
        if self._phase == Phase.PRESENTING:
            if self._presentation_interrupted:
                # Wait for the user's turn to finish, then replay (see _on_user_turn).
                # If no turn ever arrives (a cough that cut the audio), replay anyway.
                self._start_timer(self._replay_if_still_interrupted(delay=3.0))
                return
            assert self._round is not None
            await self._service.repo.mark_round_awaiting(self._round.id)
            self._phase = Phase.AWAITING
            await self._publish()
            self._start_timer(self._nudge_then_give_up())
        elif self._phase == Phase.FINISHED:
            # The final line has played. End the pipeline gracefully.
            await self.push_frame(EndWorkerFrame(reason="game_over"))

    async def _on_user_turn(self, text: str):
        logger.info(f"user turn in phase={self._phase.value}: {text!r}")
        if self._phase == Phase.PRESENTING:
            if self._presentation_interrupted:
                await self._replay("No worries, here they are again from the top.")
            return
        if self._phase == Phase.AWAITING:
            await self._evaluate(text)
            return
        # CONNECTING / EVALUATING / FINISHED: nothing to answer; ignore.

    async def _replay(self, lead_in: str):
        assert self._round is not None
        await self._service.repo.mark_round_presenting_again(self._round.id)
        await self._present(f"{lead_in} {self._round.length} cards.")

    async def _evaluate(self, text: str, timed_out: bool = False):
        assert self._round is not None
        if self._phase != Phase.AWAITING:
            return
        ev = evaluate(self._round.cards, text)

        if not ev.heard and not timed_out:
            intent = detect_intent(text)
            if intent == "quit":
                await self._end_early("player_quit")
                return
            if intent == "repeat":
                if not self._repeat_used:
                    self._repeat_used = True
                    await self._replay("Sure, one more time.")
                else:
                    await self._speak("I can only repeat once per round. Give it your best shot!")
                    self._start_timer(self._nudge_then_give_up())
                return
            self._no_card_turns += 1
            if self._no_card_turns < 2:
                await self._speak(f"I didn't catch any cards there. Say the {self._round.length} cards in order.")
                self._start_timer(self._nudge_then_give_up())
                return
            # Second turn with no cards at all: score it as a miss and move on.

        # Claim the round before the first await. The silence timer and a real
        # answer can both reach this point; only the first one gets past here.
        # (The database check in record_answer is the second, stronger guard.)
        self._phase = Phase.EVALUATING
        await self._cancel_timer()
        await self._publish()

        points = self._rules.points_for(self._round.length) if ev.correct else 0
        result = await self._service.record_answer(self._round.id, text, ev.heard, ev.correct, points)
        if not result.applied:
            # The database already has an answer for this round. Never score twice.
            logger.warning(f"round {self._round.id} was already scored; ignoring duplicate answer")
            return

        outcome = apply_result(self._state, self._round.length, ev.correct)
        # The database is the source of truth; keep the in-memory mirror aligned with it.
        self._state.score, self._state.lives = result.score, result.lives
        self._state.rounds_cleared = result.rounds_cleared
        self._last_result = LastResult(
            round_number=self._round_number,
            correct=ev.correct,
            expected=ev.expected,
            heard=ev.heard,
            points=points,
        )

        if ev.correct:
            parts = [await self._host.correct(self._player, points, result.score, result.rounds_cleared)]
        else:
            how = "they took too long" if timed_out else _describe_mistake(ev)
            parts = [
                await self._host.wrong(self._player, result.lives, how),
                f"The cards were {', '.join(ev.expected)}.",
            ]
            if not outcome.game_over:
                parts.append(
                    f"{result.lives} {'life' if result.lives == 1 else 'lives'} left."
                )

        if outcome.game_over:
            reason = "won" if outcome.won else "out_of_lives"
            await self._finish(SessionStatus.COMPLETED, reason, won=outcome.won, lead_in=" ".join(parts))
        else:
            await self._present_new_round(prefix=" ".join(parts))

    async def _end_early(self, reason: str):
        """Player said quit, or the session was ended from the API."""
        if self._phase == Phase.FINISHED:
            return
        await self._finish(SessionStatus.COMPLETED, reason, won=False, lead_in="Okay, let's wrap it up there.")

    async def _finish(self, status: str, reason: str, won: bool, lead_in: str):
        await self._cancel_timer()
        self._phase = Phase.FINISHED
        await self._service.finish_session(self._session_id, status, reason, won=won)
        closing = await self._host.game_over(self._player, self._state.score, self._state.rounds_cleared, won)
        await self._publish(status=status, end_reason=reason, won=won)
        await self._speak(f"{lead_in} {closing}")

    async def handle_disconnect(self):
        """The browser left. Called from the transport's disconnect handler."""
        await self._cancel_timer()
        if self._phase != Phase.FINISHED:
            self._phase = Phase.FINISHED
            await self._service.finish_session(self._session_id, SessionStatus.ABANDONED, "disconnected")

    # ---- timers ------------------------------------------------------------------------------

    async def _nudge_then_give_up(self):
        """If the user goes quiet: nudge once, then count the round as missed."""
        await asyncio.sleep(self._nudge_after)
        if self._phase != Phase.AWAITING or self._round is None:
            return
        await self._speak(f"Take your time. Say the {self._round.length} cards in order whenever you're ready.")
        await asyncio.sleep(self._give_up_after)
        if self._phase == Phase.AWAITING:
            logger.info("no answer in time; scoring the round as missed")
            self._timer_task = None  # we're running inside it; don't cancel ourselves
            await self._evaluate("", timed_out=True)

    async def _replay_if_still_interrupted(self, delay: float):
        await asyncio.sleep(delay)
        if self._phase == Phase.PRESENTING and self._presentation_interrupted:
            self._timer_task = None
            await self._replay("Let me read those again.")

    def _start_timer(self, coro):
        if self._timer_task is not None:
            self._timer_task.cancel()
        self._timer_task = self.create_task(coro, "game_timer")

    async def _cancel_timer(self):
        task, self._timer_task = self._timer_task, None
        if task is not None and task is not asyncio.current_task():
            await self.cancel_task(task)

    # ---- output helpers ------------------------------------------------------------------------

    async def _speak(self, text: str):
        await self.push_frame(TTSSpeakFrame(text=text))

    async def _publish(self, status: str | None = None, end_reason: str | None = None, won: bool = False):
        """Push live state to Redis (for the REST API) and to the browser (RTVI)."""
        state = SessionState(
            session_id=self._session_id,
            player_name=self._player,
            status=status or SessionStatus.ACTIVE,
            phase=self._phase.value,
            round_number=self._round_number,
            sequence_length=self._round.length if self._round else self._state.next_length,
            score=self._state.score,
            lives=self._state.lives,
            max_lives=self._rules.lives,
            rounds_cleared=self._state.rounds_cleared,
            best_length=self._best_length,
            won=won,
            end_reason=end_reason,
            last_result=self._last_result,
        )
        await self._service.publish(state)
        await self.push_frame(
            RTVIServerMessageFrame(data={"type": "game_state", "state": state.model_dump(mode="json")})
        )

    async def cleanup(self):
        await self._cancel_timer()
        await super().cleanup()
