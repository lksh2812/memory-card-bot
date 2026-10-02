"""Turn-end detection that knows how many cards the user should say.

The problem (also the example in Curelink's "Anatomy of a voice AI pipeline"
post): people pause mid-answer. "Apple... river... um... tiger" has gaps long
enough for a normal voice agent to decide the user is done and evaluate a
half-finished answer. A longer fixed timeout fixes that, but then every turn
feels slow.

We have information a general assistant doesn't: while waiting for an answer we
know exactly how many cards to expect. So the silence window adapts:

* not waiting for an answer       -> normal timeout (0.6s)
* answer has all expected cards   -> short timeout (0.35s), respond quickly
* answer still incomplete         -> patient timeout (2.0s), let them think

It extends Pipecat's built-in SpeechTimeoutUserTurnStopStrategy and only changes
how long the timer is, so all the VAD and STT edge cases in the base class are
kept.
"""

from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import TranscriptionFrame
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

from app.game.matching import count_cards


class SequenceAwareTurnStopStrategy(SpeechTimeoutUserTurnStopStrategy):
    def __init__(
        self,
        expected_cards: Callable[[], int | None],
        default_timeout: float = 0.6,
        complete_timeout: float = 0.35,
        patient_timeout: float = 2.0,
        **kwargs,
    ):
        super().__init__(user_speech_timeout=default_timeout, **kwargs)
        self._expected_cards = expected_cards
        self._default_timeout = default_timeout
        self._complete_timeout = complete_timeout
        self._patient_timeout = patient_timeout

    def _pick_timeout(self) -> float:
        expected = self._expected_cards()
        if not expected:
            return self._default_timeout
        heard = count_cards(self._text)
        timeout = self._complete_timeout if heard >= expected else self._patient_timeout
        logger.debug(f"turn-end: heard {heard}/{expected} cards -> wait {timeout}s")
        return timeout

    async def _restart_user_speech_timer(self):
        # Every time the base class (re)starts the silence timer, choose its length.
        self._user_speech_timeout = self._pick_timeout()
        await super()._restart_user_speech_timer()

    async def _handle_transcription(self, frame: TranscriptionFrame):
        await super()._handle_transcription(frame)
        # The transcript often lands after VAD already said "silence" and a long
        # timer started. If that transcript completes the answer, shorten the wait.
        if (
            self._user_speech_timeout_task is not None
            and self._user_speech_timeout == self._patient_timeout
            and self._pick_timeout() == self._complete_timeout
        ):
            await self._restart_user_speech_timer()
