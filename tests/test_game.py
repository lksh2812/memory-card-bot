import random

import pytest

from app.game.cards import DECK, generate_sequence, sequence_signature
from app.game.engine import GameRules, GameState, apply_result
from app.game.matching import detect_intent, evaluate, extract_cards


class TestExtractCards:
    def test_plain_sequence(self):
        assert extract_cards("apple river tiger") == ["apple", "river", "tiger"]

    def test_punctuation_case_and_fillers(self):
        text = "Um, okay so I think it was... Apple, then RIVER, and uh tiger."
        assert extract_cards(text) == ["apple", "river", "tiger"]

    def test_plurals(self):
        assert extract_cards("apples tomatoes") == ["apple", "tomato"]

    def test_split_word_from_stt(self):
        assert extract_cards("pen guin um brella") == ["penguin", "umbrella"]

    def test_small_misspelling_is_fuzzy_matched(self):
        assert extract_cards("guitarr violen") == ["guitar", "violin"]

    def test_unrelated_words_are_ignored(self):
        assert extract_cards("banana car house rocket") == ["banana", "rocket"]

    def test_stutter_is_collapsed(self):
        assert extract_cards("apple apple river") == ["apple", "river"]

    def test_short_words_do_not_fuzzy_match(self):
        assert extract_cards("and the then") == []


class TestEvaluate:
    expected = ["apple", "river", "tiger"]

    def test_correct(self):
        result = evaluate(self.expected, "apple, river, tiger")
        assert result.correct
        assert result.first_error_index is None
        assert result.correct_prefix == 3

    def test_wrong_order(self):
        result = evaluate(self.expected, "river apple tiger")
        assert not result.correct
        assert result.correct_prefix == 0
        assert "wrong_order" in result.notes

    def test_missing_card(self):
        result = evaluate(self.expected, "apple river")
        assert not result.correct
        assert result.correct_prefix == 2
        assert result.first_error_index == 2
        assert "missing_cards" in result.notes

    def test_extra_card_is_wrong(self):
        result = evaluate(self.expected, "apple river tiger zebra")
        assert not result.correct
        assert "extra_cards" in result.notes

    def test_wrong_card(self):
        result = evaluate(self.expected, "apple rocket tiger")
        assert not result.correct
        assert result.first_error_index == 1
        assert "wrong_card" in result.notes

    def test_empty_answer(self):
        result = evaluate(self.expected, "")
        assert not result.correct
        assert result.heard == []


class TestIntent:
    @pytest.mark.parametrize("text", ["can you repeat that", "say that again please", "sorry I didn't catch it"])
    def test_repeat(self, text):
        assert detect_intent(text) == "repeat"

    @pytest.mark.parametrize("text", ["I want to quit", "let's stop", "end the game"])
    def test_quit(self, text):
        assert detect_intent(text) == "quit"

    def test_no_intent(self):
        assert detect_intent("hmm let me think") is None


class TestSequences:
    def test_cards_are_distinct_and_from_deck(self):
        seq = generate_sequence(8, random.Random(1))
        assert len(seq) == len(set(seq)) == 8
        assert all(card in DECK for card in seq)

    def test_avoids_recent_sequences(self):
        rng = random.Random(7)
        first = generate_sequence(3, random.Random(7))
        second = generate_sequence(3, rng, avoid={sequence_signature(first)})
        assert second != first

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            generate_sequence(len(DECK) + 1)


class TestEngine:
    def test_difficulty_grows_and_caps(self):
        rules = GameRules(start_length=3, max_length=5)
        assert [rules.length_for(n) for n in range(5)] == [3, 4, 5, 5, 5]

    def test_correct_answer_scores_and_advances(self):
        state = GameState.new(GameRules())
        outcome = apply_result(state, length=3, correct=True)
        assert outcome.points == 30
        assert state.score == 30
        assert state.next_length == 4
        assert not outcome.game_over

    def test_wrong_answers_cost_lives_until_game_over(self):
        state = GameState.new(GameRules(lives=2))
        assert not apply_result(state, 3, False).game_over
        outcome = apply_result(state, 3, False)
        assert outcome.game_over and not outcome.won
        assert state.next_length == 3  # a miss does not make it harder

    def test_clearing_max_length_wins(self):
        state = GameState.new(GameRules(start_length=3, max_length=4))
        apply_result(state, 3, True)
        outcome = apply_result(state, 4, True)
        assert outcome.game_over and outcome.won

    def test_cannot_play_after_finish(self):
        state = GameState.new(GameRules(lives=1))
        apply_result(state, 3, False)
        with pytest.raises(ValueError):
            apply_result(state, 3, True)
