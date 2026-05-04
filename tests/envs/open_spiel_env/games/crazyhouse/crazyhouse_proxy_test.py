"""Tests for the proxied Crazyhouse game."""

import json

import pyspiel
from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.crazyhouse import crazyhouse_proxy


def _play(state, san):
    """Apply the legal action whose action_to_string matches ``san``."""
    for action in state.legal_actions():
        if state.action_to_string(state.current_player(), action) == san:
            state.apply_action(action)
            return
    raise AssertionError(f"move {san!r} not legal here")


class CrazyhouseProxyTest(absltest.TestCase):
    def test_game_is_registered(self):
        game = pyspiel.load_game("crazyhouse_proxy")
        self.assertIsInstance(game, crazyhouse_proxy.CrazyhouseGame)

    def test_random_sim(self):
        game = crazyhouse_proxy.CrazyhouseGame()
        pyspiel.random_sim_test(game, num_sims=3, serialize=False, verbose=False)

    def test_initial_state_to_json(self):
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        d = json.loads(state.to_json())

        # Standard chess starting position.
        self.assertEqual(d["fen"].split()[0], "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        self.assertEqual(d["side_to_move"], "w")
        self.assertEqual(d["castling_rights"], "KQkq")
        self.assertEqual(d["en_passant"], "-")
        self.assertEqual(d["halfmove_clock"], 0)
        self.assertEqual(d["fullmove_number"], 1)
        self.assertEqual(d["current_player"], "white")
        self.assertFalse(d["is_terminal"])
        self.assertIsNone(d["winner"])
        self.assertEqual(d["pockets"], {"white": {}, "black": {}})

    def test_board_layout_matches_fen(self):
        """Board[0] is rank 8 (Black back rank); board[7] is rank 1 (White)."""
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        d = state.state_dict()
        self.assertEqual(d["board"][0], list("rnbqkbnr"))
        self.assertEqual(d["board"][1], list("pppppppp"))
        self.assertEqual(d["board"][6], list("PPPPPPPP"))
        self.assertEqual(d["board"][7], list("RNBQKBNR"))
        for rank in d["board"][2:6]:
            self.assertEqual(rank, ["."] * 8)

    def test_observation_string_is_json(self):
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        # Both observation_string and __str__ produce parseable JSON.
        self.assertEqual(json.loads(state.observation_string(0)), state.state_dict())
        self.assertEqual(json.loads(str(state)), state.state_dict())

    def test_current_player_labels(self):
        """OpenSpiel uses 1=White, 0=Black for crazyhouse — the proxy
        converts those raw ids into the human-readable colour labels."""
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()

        self.assertEqual(state.current_player(), 1)
        self.assertEqual(state.state_dict()["current_player"], "white")

        _play(state, "e4")
        self.assertEqual(state.current_player(), 0)
        self.assertEqual(state.state_dict()["current_player"], "black")

    def test_capture_populates_pocket(self):
        """After 1.e4 d5 2.exd5 Black has lost a pawn; White's pocket gains
        a pawn. After 2...Qxd5 Black recaptures and gains a pawn too."""
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        _play(state, "e4")
        _play(state, "d5")
        _play(state, "exd5")
        d = state.state_dict()
        self.assertEqual(d["pockets"]["white"], {"P": 1})
        self.assertEqual(d["pockets"]["black"], {})

        _play(state, "Qxd5")
        d = state.state_dict()
        self.assertEqual(d["pockets"]["white"], {"P": 1})
        self.assertEqual(d["pockets"]["black"], {"P": 1})
        # FEN bracketed pocket section preserves both players.
        self.assertIn("[Pp]", d["fen"])

    def test_drop_consumes_pocket(self):
        """Dropping P@e4 from a pocketed pawn removes it from the pocket."""
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        for san in ["e4", "d5", "exd5", "Qxd5"]:
            _play(state, san)
        # White has 1 pawn in pocket; drop it on e4.
        _play(state, "P@e4")
        d = state.state_dict()
        self.assertEqual(d["pockets"]["white"], {})
        self.assertEqual(d["board"][4][4], "P")  # e4 == board[4][4] (rank 4, file e)

    def test_terminal_winner(self):
        """Fool's mate produces a Black win; the proxy reports it."""
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        # 1. f3 e5 2. g4 Qh4#
        for san in ["f3", "e5", "g4", "Qh4#"]:
            _play(state, san)
        self.assertTrue(state.is_terminal())
        d = state.state_dict()
        self.assertTrue(d["is_terminal"])
        self.assertEqual(d["winner"], "black")
        # Once terminal, current_player is the OpenSpiel terminal sentinel.
        self.assertNotIn(d["current_player"], ("white", "black"))


if __name__ == "__main__":
    absltest.main()
