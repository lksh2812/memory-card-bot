"""Turning a raw transcript into the list of cards the user said.

This is the core validation the brief asks to keep in backend code rather than
in an LLM prompt. It is deterministic and fast, and it never "decides" a game
outcome based on model output.

Speech-to-text output is messy, so the pipeline is:

1. lowercase, strip punctuation, split into tokens
2. map each token to a deck card: exact match, plural ("apples"), a word that
   STT split in two ("pen guin"), or a close fuzzy match ("guitarr")
3. drop everything that isn't a card (fillers like "um", "then", "I think")
4. collapse a card said twice in a row (a stutter: "apple apple river")

Cards in a sequence are always distinct, so step 4 can never hide a real answer.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.game.cards import DECK, DECK_SET

# Tokens that should never be fuzzy matched to a card. Without this list a short
# word can drift into a card at a low threshold.
STOPWORDS = frozenset(
    "a an and the then so um umm uh uhh er ah hmm mm oh okay ok yeah yes no "
    "i im it its was is were are think like well next after first last finally "
    "me my let see that this those these one two three".split()
)

FUZZY_THRESHOLD = 0.8
MIN_FUZZY_LEN = 4

REPEAT_PHRASES = (
    "repeat", "again", "one more time", "didn't hear", "didnt hear",
    "didn't catch", "didnt catch", "say that", "pardon",
)
QUIT_PHRASES = (
    "quit", "stop the game", "end the game", "i give up", "exit the game",
    "i'm done", "im done", "let's stop", "lets stop",
)

_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _match_token(token: str) -> str | None:
    """Map one token to a card, or None if it isn't one."""
    if token in DECK_SET:
        return token
    if token in STOPWORDS:
        return None
    # Plurals: "apples" -> "apple", "tomatoes" -> "tomato".
    for suffix in ("es", "s"):
        if token.endswith(suffix) and token[: -len(suffix)] in DECK_SET:
            return token[: -len(suffix)]
    if len(token) < MIN_FUZZY_LEN:
        return None
    best, best_score = None, 0.0
    for card in DECK:
        score = SequenceMatcher(None, token, card).ratio()
        if score > best_score:
            best, best_score = card, score
    return best if best_score >= FUZZY_THRESHOLD else None


def extract_cards(text: str) -> list[str]:
    """Return the cards found in `text`, in the order they were said."""
    tokens = tokenize(text)
    cards: list[str] = []
    i = 0
    while i < len(tokens):
        # STT sometimes splits one word into two ("pen guin", "um brella").
        if i + 1 < len(tokens):
            joined = tokens[i] + tokens[i + 1]
            if joined in DECK_SET and tokens[i] not in DECK_SET:
                cards.append(joined)
                i += 2
                continue
        card = _match_token(tokens[i])
        if card is not None:
            cards.append(card)
        i += 1
    # Collapse stutters: "apple apple river" -> "apple river".
    collapsed: list[str] = []
    for card in cards:
        if not collapsed or collapsed[-1] != card:
            collapsed.append(card)
    return collapsed


def detect_intent(text: str) -> str | None:
    """Spot a spoken command. Only used when the user said no cards at all."""
    lowered = " ".join(tokenize(text))
    if any(p in lowered for p in QUIT_PHRASES):
        return "quit"
    if any(p in lowered for p in REPEAT_PHRASES):
        return "repeat"
    return None


def count_cards(text: str) -> int:
    """How many cards a partial transcript already contains (used for turn-taking)."""
    return len(extract_cards(text))


@dataclass(frozen=True)
class Evaluation:
    expected: list[str]
    heard: list[str]
    correct: bool
    # How many cards were right from the start before the first mistake.
    correct_prefix: int
    # Index of the first wrong or missing card, None when correct.
    first_error_index: int | None = None
    notes: list[str] = field(default_factory=list)


def evaluate(expected: list[str], transcript: str) -> Evaluation:
    """Compare what the user said against the expected sequence.

    Order matters and every card must be present. Extra cards make the answer
    wrong too, otherwise listing the whole deck would always pass.
    """
    heard = extract_cards(transcript)
    prefix = 0
    for exp, got in zip(expected, heard):
        if exp != got:
            break
        prefix += 1
    correct = heard == expected
    notes: list[str] = []
    if not correct:
        if len(heard) < len(expected) and prefix == len(heard):
            notes.append("missing_cards")
        elif len(heard) > len(expected) and prefix == len(expected):
            notes.append("extra_cards")
        elif sorted(heard) == sorted(expected):
            notes.append("wrong_order")
        else:
            notes.append("wrong_card")
    return Evaluation(
        expected=list(expected),
        heard=heard,
        correct=correct,
        correct_prefix=prefix,
        first_error_index=None if correct else prefix,
        notes=notes,
    )
