"""The game host's voice: short, lively lines around the game.

The LLM only writes flavor text ("Ooh, so close!"). It never sees the cards,
never decides correctness, and never reads the sequence. Correctness comes from
app.game.matching and the cards are spoken verbatim by the game processor.

Two safety nets keep the voice snappy and safe:
* a hard timeout, after which we use a template line instead
* a filter that rejects any LLM line mentioning a card word, so the host can't
  accidentally confuse the player with an extra card name
"""

import asyncio
import random
import re

from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

from app.game.matching import extract_cards

SYSTEM_PROMPT = """You are Max, the upbeat host of a voice memory game called Memory Lane.
Your lines are spoken aloud by text-to-speech, so:
- one or two short sentences, under 30 words total
- plain words only: no emojis, lists, markdown, or stage directions
- warm and playful like a game-show host, never sarcastic or mean
- never name any card, object, or word from the game, and never invent cards
- do not ask questions and do not explain the rules unless asked
Reply with only the line to speak."""

TEMPLATES: dict[str, list[str]] = {
    "intro": [
        "Hey {player}, welcome to Memory Lane! I'll read you some cards, you say them back in the same order. Let's go!",
        "Welcome to Memory Lane, {player}! Listen closely, then repeat the cards in order. Here we go!",
    ],
    "correct": [
        "Nailed it! Plus {points} points.",
        "Perfect recall! That's {points} points.",
        "Spot on! {points} points for you.",
    ],
    "wrong": [
        "Ooh, not quite.",
        "So close, but not this time.",
        "Ah, that one slipped away.",
    ],
    "game_over_lost": [
        "That's the end of the game, {player}. You finished with {score} points. Great effort!",
        "Game over, {player}! Final score: {score} points. Thanks for playing!",
    ],
    "game_over_won": [
        "Incredible, {player}! You cleared every level with {score} points. You're a memory champion!",
    ],
}


def template(kind: str, **facts) -> str:
    return random.choice(TEMPLATES[kind]).format(**facts)


def _clean(line: str) -> str:
    line = re.sub(r"[*_#`~>\[\]\"]", "", line).strip()
    return re.sub(r"\s+", " ", line)


class HostVoice:
    def __init__(self, llm: LLMService | None, timeout: float = 2.0):
        self._llm = llm
        self._timeout = timeout

    async def line(self, kind: str, situation: str, **facts) -> str:
        """Ask the LLM for a host line, falling back to a template."""
        fallback = template(kind, **facts)
        if self._llm is None:
            return fallback
        prompt = f"Situation: {situation}\nWrite the host's line."
        try:
            text = await asyncio.wait_for(
                self._llm.run_inference(
                    LLMContext(messages=[{"role": "user", "content": prompt}]),
                    max_tokens=400,
                    system_instruction=SYSTEM_PROMPT,
                ),
                timeout=self._timeout,
            )
        except Exception as e:  # timeout, rate limit, network: the game must go on
            logger.warning(f"host line fell back to template ({kind}): {e!r}")
            return fallback
        text = _clean(text or "")
        if not text or len(text) > 220 or extract_cards(text):
            logger.info(f"host line rejected ({kind}): {text!r}")
            return fallback
        return text

    async def intro(self, player: str) -> str:
        return await self.line(
            "intro",
            f"A player named {player} just joined. Welcome them and say round one is starting. "
            "Mention they should repeat the cards in the same order.",
            player=player,
        )

    async def correct(self, player: str, points: int, score: int, streak: int) -> str:
        return await self.line(
            "correct",
            f"{player} repeated the sequence perfectly, earning {points} points (total {score}, "
            f"{streak} rounds cleared). Celebrate briefly and mention the points. The next round gets harder.",
            points=points,
        )

    async def wrong(self, player: str, lives: int, how: str) -> str:
        return await self.line(
            "wrong",
            f"{player} got it wrong ({how}). They have {lives} lives left. "
            "React with light sympathy in one short sentence. Do not mention lives or the cards.",
        )

    async def game_over(self, player: str, score: int, rounds: int, won: bool) -> str:
        kind = "game_over_won" if won else "game_over_lost"
        what = "cleared every level and won" if won else "ran out of lives"
        return await self.line(
            kind,
            f"The game is over: {player} {what}, with {score} points after clearing {rounds} rounds. "
            "Wrap up warmly and say their final score.",
            player=player,
            score=score,
        )
