"""Tests for the Coin Game Arena env (4-player 2v2 OpenSpiel game)."""

import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env.games.coin_game_arena import (
    coin_game_arena_game,
)


def _new_state(seed=0, episode_length=20, **params):
    p = {"seed": seed, "episode_length": episode_length, **params}
    return pyspiel.load_game("coin_game_arena", p).new_initial_state()


def _step_random(state, rng):
    legal = state.legal_actions()
    state.apply_action(rng.choice(legal))


# Sequential player order: teams interleave, seats alternate per team.
_PLAYER_ORDER = [0, 2, 1, 3]


class StructureTest(absltest.TestCase):
    """Tests for basic game shape: 4 players, 2 teams, sequential turns."""

    def test_four_players(self):
        g = pyspiel.load_game("coin_game_arena")
        self.assertEqual(g.num_players(), 4)

    def test_sequential_dynamics(self):
        g = pyspiel.load_game("coin_game_arena")
        self.assertEqual(g.get_type().dynamics, pyspiel.GameType.Dynamics.SEQUENTIAL)
        s = _new_state(seed=1)
        self.assertFalse(s.is_simultaneous_node())
        self.assertEqual(s.current_player(), 0)

    def test_only_acting_player_has_legal_actions(self):
        s = _new_state(seed=1)
        # Step 0: only player 0 acts.
        legal_counts = [len(s.legal_actions(p)) for p in range(4)]
        self.assertEqual(legal_counts, [5, 0, 0, 0])

    def test_player_order_interleaves_teams(self):
        s = _new_state(seed=1)
        observed = []
        for _ in range(8):
            observed.append(s.current_player())
            s.apply_action(s.legal_actions()[0])  # stand-ish, just walk
        # Two full cycles of [0, 2, 1, 3].
        self.assertEqual(observed, _PLAYER_ORDER * 2)


class ObservationTest(absltest.TestCase):
    """Per-player observations only reveal the calling team's board."""

    def test_player_0_sees_only_team_a_board(self):
        s = _new_state(seed=4)
        obs0 = json.loads(s.observation_string(0))
        self.assertEqual(obs0["your_team_id"], 0)
        self.assertEqual(obs0["your_seat"], 0)
        self.assertIn("board", obs0)
        # Mid-game, the cross-board "boards" array is hidden.
        self.assertNotIn("boards", obs0)
        self.assertNotIn("preferences", obs0)

    def test_player_2_sees_only_team_b_board(self):
        s = _new_state(seed=4)
        obs2 = json.loads(s.observation_string(2))
        self.assertEqual(obs2["your_team_id"], 1)
        self.assertEqual(obs2["your_seat"], 0)
        self.assertEqual(obs2["board"]["team_id"], 1)

    def test_teammates_see_same_board_different_pref(self):
        s = _new_state(seed=4)
        obs0 = json.loads(s.observation_string(0))
        obs1 = json.loads(s.observation_string(1))
        # Same physical board state.
        self.assertEqual(obs0["board"]["board"], obs1["board"]["board"])
        self.assertEqual(obs0["board"]["player_positions"], obs1["board"]["player_positions"])
        # Distinct preferences (each player on a board has their own colour).
        self.assertNotEqual(obs0["your_preference"], obs1["your_preference"])

    def test_opponent_team_board_differs(self):
        # Setup is randomized per board, so seeing only one of them is
        # genuinely less info than seeing both.
        s = _new_state(seed=4)
        obs0 = json.loads(s.observation_string(0))
        obs2 = json.loads(s.observation_string(2))
        # The two boards are independently laid out.
        self.assertNotEqual(
            obs0["board"]["player_positions"],
            obs2["board"]["player_positions"],
        )


class TurnHistoryTest(absltest.TestCase):
    """Each board records both teammates' moves so partners can see them."""

    def test_history_includes_both_seats(self):
        s = _new_state(seed=2)
        # Sequential order is [0, 2, 1, 3]: seat 0 plays "right" on each
        # board, then seat 1 plays "left" on each board.
        s.apply_action(3)  # player 0: right
        s.apply_action(3)  # player 2: right
        s.apply_action(2)  # player 1: left
        s.apply_action(2)  # player 3: left

        obs0 = json.loads(s.observation_string(0))  # team A view
        history = obs0["board"]["move_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"seat": 0, "player_id": 0, "action": "right"})
        self.assertEqual(history[1], {"seat": 1, "player_id": 1, "action": "left"})

        obs2 = json.loads(s.observation_string(2))  # team B view
        history_b = obs2["board"]["move_history"]
        self.assertEqual(history_b[0], {"seat": 0, "player_id": 2, "action": "right"})
        self.assertEqual(history_b[1], {"seat": 1, "player_id": 3, "action": "left"})


class DeterminismTest(absltest.TestCase):
    """Same seed -> identical setup; different seeds differ."""

    def test_same_seed_identical(self):
        a = json.loads(_new_state(seed=99).observation_string(0))
        b = json.loads(_new_state(seed=99).observation_string(0))
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        a = json.loads(_new_state(seed=99).observation_string(0))
        b = json.loads(_new_state(seed=100).observation_string(0))
        self.assertNotEqual(a, b)


class TerminationTest(absltest.TestCase):
    """Episode ends after ``2 * episode_length`` steps (per-board moves)."""

    def test_terminates_on_episode_length(self):
        s = _new_state(seed=3, episode_length=4)
        rng = random.Random(0)
        steps = 0
        while not s.is_terminal() and steps < 100:
            _step_random(s, rng)
            steps += 1
        self.assertTrue(s.is_terminal())
        # 4 moves per board x 2 boards = 8 sequential steps.
        self.assertEqual(steps, 8)


class ScoringTest(absltest.TestCase):
    """Reward formula and team aggregation."""

    def test_zero_returns_when_no_coins_collected(self):
        s = _new_state(seed=1, episode_length=2)
        # All four players "stand" once each (total moves = 2 * 2 = 4).
        for _ in range(4):
            s.apply_action(4)
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.returns(), [0.0, 0.0, 0.0, 0.0])

    def test_team_total_and_winner_at_terminal(self):
        s = _new_state(seed=5, episode_length=100)
        rng = random.Random(0)
        while not s.is_terminal():
            _step_random(s, rng)
        obs0 = json.loads(s.observation_string(0))
        self.assertIn("team_totals", obs0)
        self.assertIn("winning_team", obs0)
        self.assertEqual(len(obs0["team_totals"]), 2)
        # Returns must agree with team totals.
        returns = obs0["returns"]
        self.assertAlmostEqual(returns[0] + returns[1], obs0["team_totals"][0])
        self.assertAlmostEqual(returns[2] + returns[3], obs0["team_totals"][1])
        # Winner consistency.
        a, b = obs0["team_totals"]
        if a > b:
            self.assertEqual(obs0["winning_team"], 0)
        elif b > a:
            self.assertEqual(obs0["winning_team"], 1)
        else:
            self.assertEqual(obs0["winning_team"], "draw")

    def test_reward_formula_matches_hand_calc(self):
        """Hand-verify ``self_pref^2 + other_pref^2 - bad_coins^2``."""
        # Use a small random episode and reproduce returns from collected counts.
        s = _new_state(seed=5, episode_length=100)
        rng = random.Random(0)
        while not s.is_terminal():
            _step_random(s, rng)
        # At terminal, every per-player view reveals both boards and prefs.
        full = json.loads(s.observation_string(0))
        prefs = {int(k): v for k, v in full["preferences"].items()}
        for team in range(2):
            board = full["boards"][team]
            collected = {int(k): v for k, v in board["coins_collected"].items()}
            pids = [team * 2, team * 2 + 1]
            team_prefs = {prefs[p] for p in pids}
            totals = {}
            for color in board["coin_colors"]:
                totals[color] = sum(collected[p].get(color, 0) for p in pids)
            bad = sum(c for col, c in totals.items() if col not in team_prefs)
            for pid in pids:
                self_p = prefs[pid]
                other_p = prefs[1 - (pid % 2) + team * 2]
                expected = (
                    totals[self_p] ** 2
                    + totals[other_p] ** 2
                    - bad ** 2
                )
                self.assertEqual(s.returns()[pid], float(expected))


class TerminalRevealTest(absltest.TestCase):
    """At terminal, both boards and all preferences become visible."""

    def test_terminal_reveal(self):
        s = _new_state(seed=5, episode_length=2)
        for _ in range(4):
            s.apply_action(4)  # stand
        obs0 = json.loads(s.observation_string(0))
        self.assertIn("boards", obs0)
        self.assertEqual(len(obs0["boards"]), 2)
        self.assertIn("preferences", obs0)
        self.assertEqual(len(obs0["preferences"]), 4)
        self.assertIn("returns", obs0)


class KaggleEnvIntegrationTest(absltest.TestCase):
    """Full episode through the kaggle env wrapper with random agents."""

    def test_random_agents_run_to_completion(self):
        env = make(
            "open_spiel_coin_game_arena",
            configuration={
                "openSpielGameParameters": {"seed": 11, "episode_length": 8},
                "includeLegalActions": True,
            },
        )
        env.run(["random", "random", "random", "random"])
        last = env.steps[-1]
        for agent in last:
            self.assertEqual(agent["status"], "DONE")
        # Sum of team rewards = team totals; rewards are floats.
        for agent in last:
            self.assertIsInstance(agent["reward"], float)


if __name__ == "__main__":
    absltest.main()
