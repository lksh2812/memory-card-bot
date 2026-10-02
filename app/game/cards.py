"""The card deck and sequence generation.

The deck is hardcoded on purpose (the brief allows it). Words were picked to be
easy to say, two or three syllables, and phonetically far apart, so speech-to-text
rarely confuses one card for another. Short words like "pear" or "flower" were
avoided because they have homophones ("pair", "flour").
"""

import random

DECK: tuple[str, ...] = (
    "anchor", "apple", "banana", "blanket", "bottle", "button", "cactus", "camera",
    "candle", "carrot", "castle", "cookie", "compass", "diamond", "dolphin", "dragon",
    "feather", "garden", "guitar", "hammer", "helmet", "island", "jacket", "ladder",
    "lemon", "magnet", "mirror", "monkey", "orange", "panda", "parrot", "pencil",
    "penguin", "piano", "pillow", "planet", "pumpkin", "rabbit", "river", "rocket",
    "spider", "tiger", "tomato", "turtle", "umbrella", "violin", "volcano", "wallet",
    "window", "zebra",
)

DECK_SET = frozenset(DECK)


def sequence_signature(cards: list[str]) -> str:
    """Stable string form of a sequence, used to detect repeats across games."""
    return "|".join(cards)


def generate_sequence(
    length: int,
    rng: random.Random | None = None,
    avoid: set[str] | None = None,
    max_attempts: int = 10,
) -> list[str]:
    """Pick `length` distinct cards.

    `avoid` holds signatures of sequences this player saw recently (served from
    Redis). We retry a few times to dodge them; with 50 cards a collision is
    already unlikely, so this is a cheap guarantee rather than a hot loop.
    Cards are distinct within a sequence, which also keeps validation simple:
    a repeated word in the answer can only be a stutter, never a real card.
    """
    if length > len(DECK):
        raise ValueError(f"sequence length {length} exceeds deck size {len(DECK)}")
    rng = rng or random.Random()
    avoid = avoid or set()
    cards = rng.sample(DECK, length)
    for _ in range(max_attempts):
        if sequence_signature(cards) not in avoid:
            break
        cards = rng.sample(DECK, length)
    return cards
