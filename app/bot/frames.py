"""Custom frames for driving the game from outside the processor.

Event handlers (client ready, API "end session") run outside the pipeline. Rather
than calling the processor directly from those handlers, we queue a frame. The
frame travels down the pipeline and the processor handles it in order with
everything else, so there is a single place where game state changes.
"""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class StartGameFrame(DataFrame):
    """The browser is connected and ready: start round one."""


@dataclass
class EndGameFrame(DataFrame):
    """End the game early (sent when POST /sessions/{id}/end is called)."""

    reason: str = "ended_by_player"
