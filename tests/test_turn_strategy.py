"""The sequence-aware turn strategy picks the right silence window."""

from app.bot.turn_strategy import SequenceAwareTurnStopStrategy


def make(expected):
    return SequenceAwareTurnStopStrategy(expected_cards=lambda: expected)


def test_default_timeout_when_not_waiting_for_an_answer():
    s = make(None)
    s._text = "hello there"
    assert s._pick_timeout() == 0.6


def test_patient_while_answer_is_incomplete():
    s = make(3)
    s._text = "apple river"
    assert s._pick_timeout() == 2.0


def test_quick_once_all_cards_are_heard():
    s = make(3)
    s._text = "apple, um, river and tiger"
    assert s._pick_timeout() == 0.35
