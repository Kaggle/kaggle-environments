"""Tests for Go proxy scoring."""

import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.go import go_proxy


# OpenSpiel Go action layout for 9x9:
#   action = row * board_size + col  (row 0 = bottom, col 0 = left)
#   pass = board_size * board_size (81 for 9x9)
# Columns: A=0 B=1 C=2 D=3 E=4 F=5 G=6 H=7 J=8


def _gtp_to_action(gtp_vertex, board_size=9):
    """Convert a GTP vertex like 'E5' or 'pass' to an OpenSpiel action."""
    col_map = {c: i for i, c in enumerate("ABCDEFGHJKLMNOPQRSTUVWXYZ"[:board_size])}
    if gtp_vertex.lower() == "pass":
        return board_size * board_size
    col = col_map[gtp_vertex[0].upper()]
    row = int(gtp_vertex[1:]) - 1
    return row * board_size + col


class GoProxyTest(absltest.TestCase):

    def test_game_is_registered(self):
        game = pyspiel.load_game("go_proxy")
        self.assertIsInstance(game, go_proxy.GoGame)

    def test_non_terminal_state_has_no_scoring(self):
        """Non-terminal states should not include a scoring key."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        state.apply_action(_gtp_to_action("E5"))
        d = state.state_dict()
        self.assertNotIn("scoring", d)

    def test_terminal_state_has_scoring(self):
        """Terminal states must include a scoring key."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        state.apply_action(_gtp_to_action("pass"))
        state.apply_action(_gtp_to_action("pass"))
        self.assertTrue(state.is_terminal())
        d = state.state_dict()
        self.assertIn("scoring", d)

    def test_scoring_in_json(self):
        """Scoring should flow through to_json/observation_json."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        state.apply_action(_gtp_to_action("pass"))
        state.apply_action(_gtp_to_action("pass"))
        obs = json.loads(state.observation_json(0))
        self.assertIn("scoring", obs)
        self.assertEqual(obs["scoring"]["scoring_method"], "tromp-taylor")

    def test_empty_board_white_wins_by_komi(self):
        """Empty board + double pass: White wins by komi."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        state.apply_action(_gtp_to_action("pass"))
        state.apply_action(_gtp_to_action("pass"))
        scoring = state.state_dict()["scoring"]
        self.assertEqual(scoring["black_stones"], 0)
        self.assertEqual(scoring["white_stones"], 0)
        self.assertEqual(scoring["black_territory"], 0)
        self.assertEqual(scoring["white_territory"], 0)
        self.assertEqual(scoring["dame"], 81)
        self.assertEqual(scoring["winner"], "W")
        self.assertEqual(scoring["winning_margin"], 7.5)

    def test_katago_verified_position(self):
        """Verified against KataGo final_score: W+5.5.

        9x9, komi=5.5. Moves: E5 C3 E3 G5 E7 G7 C5 G3 pass pass.
        4 black stones, 4 white stones, no enclosed territory.
        """
        game = go_proxy.GoGame({"board_size": 9, "komi": 5.5})
        state = game.new_initial_state()
        for move in ["E5", "C3", "E3", "G5", "E7", "G7", "C5", "G3", "pass", "pass"]:
            state.apply_action(_gtp_to_action(move))
        self.assertTrue(state.is_terminal())
        scoring = state.state_dict()["scoring"]
        self.assertEqual(scoring["black_stones"], 4)
        self.assertEqual(scoring["white_stones"], 4)
        self.assertEqual(scoring["black_territory"], 0)
        self.assertEqual(scoring["white_territory"], 0)
        self.assertEqual(scoring["winner"], "W")
        self.assertEqual(scoring["winning_margin"], 5.5)
        self.assertEqual(scoring["black_score"], 4.0)
        self.assertEqual(scoring["white_score"], 9.5)

    def test_scoring_fields(self):
        """All expected fields are present in the scoring dict."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        state.apply_action(_gtp_to_action("pass"))
        state.apply_action(_gtp_to_action("pass"))
        scoring = state.state_dict()["scoring"]
        expected_keys = {
            "black_stones", "white_stones",
            "black_territory", "white_territory",
            "dame", "komi",
            "black_score", "white_score",
            "winner", "winning_margin",
            "scoring_method",
        }
        self.assertEqual(set(scoring.keys()), expected_keys)
        self.assertEqual(scoring["scoring_method"], "tromp-taylor")
        self.assertEqual(scoring["komi"], 7.5)

    def test_territory_scoring(self):
        """Black surrounds territory in the corner.

        Board (relevant part):
          A1=B, B1=B, C1=., A2=B, B2=B, A3=B, B3=B
        Black encloses C1 (empty, reaches only Black via A1,B1 neighbors).
        Actually let's build a clearer case: fill a wall.
        """
        # Black plays a line across row 1: A1-H1, creating territory below
        # is impossible on row 1 (it's the bottom). Instead, let's test
        # a simple corner enclosure.
        #
        # Black: A2, B1 -> encloses A1
        # White: somewhere far away
        game = go_proxy.GoGame({"board_size": 9, "komi": 0.5})
        state = game.new_initial_state()
        moves = [
            "A2",  # B
            "J9",  # W
            "B1",  # B
            "H9",  # W
            "pass",  # B
            "pass",  # W
        ]
        for move in moves:
            state.apply_action(_gtp_to_action(move))
        self.assertTrue(state.is_terminal())
        scoring = state.state_dict()["scoring"]
        self.assertEqual(scoring["black_stones"], 2)
        self.assertEqual(scoring["white_stones"], 2)
        # A1 is enclosed by Black (neighbors: A2=B, B1=B)
        self.assertEqual(scoring["black_territory"], 1)
        self.assertEqual(scoring["winner"], "B")
        self.assertEqual(scoring["black_score"], 3.0)
        self.assertEqual(scoring["white_score"], 2.0 + 0.5)

    def test_scoring_points_sum_to_board_size(self):
        """stones + territory + dame must equal board_size^2."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        for move in ["E5", "C3", "E3", "G5", "E7", "G7", "C5", "G3", "pass", "pass"]:
            state.apply_action(_gtp_to_action(move))
        scoring = state.state_dict()["scoring"]
        total = (
            scoring["black_stones"] + scoring["white_stones"]
            + scoring["black_territory"] + scoring["white_territory"]
            + scoring["dame"]
        )
        self.assertEqual(total, 81)

    def test_winner_matches_openspiel_returns(self):
        """Scoring winner must agree with OpenSpiel returns() for random games."""
        random.seed(12345)
        for _ in range(10):
            game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
            state = game.new_initial_state()
            for _ in range(300):
                if state.is_terminal():
                    break
                state.apply_action(random.choice(state.legal_actions()))
            if not state.is_terminal():
                continue
            scoring = state.state_dict()["scoring"]
            returns = state.returns()
            expected_winner = "B" if returns[0] > returns[1] else "W"
            self.assertEqual(
                scoring["winner"],
                expected_winner,
                f"Winner mismatch: scoring={scoring['winner']}, "
                f"returns={returns}",
            )

    def test_different_komi_values(self):
        """Scoring uses the correct komi from game parameters."""
        for komi in [0.5, 5.5, 6.5, 7.5]:
            game = go_proxy.GoGame({"board_size": 9, "komi": komi})
            state = game.new_initial_state()
            state.apply_action(_gtp_to_action("pass"))
            state.apply_action(_gtp_to_action("pass"))
            scoring = state.state_dict()["scoring"]
            self.assertEqual(scoring["komi"], komi)
            self.assertEqual(scoring["winning_margin"], komi)

    def test_handicap_adjusts_score(self):
        """Handicap >= 2 adds points to White's score.

        Handicap stone placement uses fixed 19x19 positions, so this test
        must use board_size=19. With 2 handicap stones + double pass, all
        empty points are Black territory (only border Black). The handicap
        bonus goes to White: white_score includes + handicap.
        """
        game = go_proxy.GoGame({"board_size": 19, "komi": 0.5, "handicap": 2})
        state = game.new_initial_state()
        while not state.is_terminal():
            state.apply_action(19 * 19)  # pass
        scoring = state.state_dict()["scoring"]
        self.assertEqual(scoring["black_stones"], 2)
        self.assertEqual(scoring["white_stones"], 0)
        # All 359 empty points are Black territory (only border Black)
        self.assertEqual(scoring["black_territory"], 359)
        self.assertEqual(scoring["black_score"], 361.0)
        # White gets komi (0.5) + handicap (2) = 2.5
        self.assertEqual(scoring["white_score"], 2.5)
        # Without handicap adjustment, white_score would be only 0.5
        self.assertEqual(scoring["winner"], "B")

    def test_draw_with_zero_komi(self):
        """Equal scores with komi=0 should be a draw, not default to White."""
        game = go_proxy.GoGame({"board_size": 9, "komi": 0.0})
        state = game.new_initial_state()
        # One black stone, one white stone, no territory -> 1 vs 1 + 0 komi
        moves = [
            "E5",   # B
            "D5",   # W
            "pass",  # B
            "pass",  # W
        ]
        for move in moves:
            state.apply_action(_gtp_to_action(move))
        self.assertTrue(state.is_terminal())
        scoring = state.state_dict()["scoring"]
        self.assertEqual(scoring["black_score"], scoring["white_score"])
        self.assertEqual(scoring["winner"], "draw")
        self.assertEqual(scoring["winning_margin"], 0.0)


if __name__ == "__main__":
    absltest.main()
