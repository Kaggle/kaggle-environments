"""Tests for the Ant Foraging Arena env (4-player 2v2 OpenSpiel game)."""

import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env.games.ant_foraging_arena import (
    ant_foraging_arena_game,  # noqa: F401
)


def _new_state(seed=0, max_turns=50, **params):
    p = {"seed": seed, "max_turns": max_turns, **params}
    return pyspiel.load_game("ant_foraging_arena", p).new_initial_state()


def _step_random(state, rng):
    legal = state.legal_actions()
    state.apply_action(rng.choice(legal))


# Sequential player order: teams interleave, seats alternate per team.
_PLAYER_ORDER = [0, 2, 1, 3]


class StructureTest(absltest.TestCase):
    """Tests for basic game shape: 4 players, 2 teams, sequential turns."""

    def test_four_players(self):
        g = pyspiel.load_game("ant_foraging_arena")
        self.assertEqual(g.num_players(), 4)

    def test_sequential_dynamics(self):
        g = pyspiel.load_game("ant_foraging_arena")
        self.assertEqual(g.get_type().dynamics, pyspiel.GameType.Dynamics.SEQUENTIAL)
        s = _new_state(seed=1)
        self.assertFalse(s.is_simultaneous_node())
        self.assertEqual(s.current_player(), 0)

    def test_only_acting_player_has_legal_actions(self):
        s = _new_state(seed=1)
        # Step 0: only player 0 acts. Five actions: stay/up/down/left/right.
        legal_counts = [len(s.legal_actions(p)) for p in range(4)]
        self.assertEqual(legal_counts, [5, 0, 0, 0])

    def test_player_order_interleaves_teams(self):
        s = _new_state(seed=1)
        observed = []
        for _ in range(8):
            observed.append(s.current_player())
            s.apply_action(0)  # everyone "stays" — no food collisions
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

    def test_player_2_sees_only_team_b_board(self):
        s = _new_state(seed=4)
        obs2 = json.loads(s.observation_string(2))
        self.assertEqual(obs2["your_team_id"], 1)
        self.assertEqual(obs2["your_seat"], 0)
        self.assertEqual(obs2["board"]["team_id"], 1)

    def test_teammates_see_same_board(self):
        s = _new_state(seed=4)
        obs0 = json.loads(s.observation_string(0))
        obs1 = json.loads(s.observation_string(1))
        # Same physical board state.
        self.assertEqual(obs0["board"]["grid"], obs1["board"]["grid"])
        self.assertEqual(obs0["board"]["ant_positions"], obs1["board"]["ant_positions"])
        self.assertEqual(obs0["board"]["food_positions"], obs1["board"]["food_positions"])

    def test_both_teams_share_identical_board_setup(self):
        # Both teams start on identical boards (same nest, food, and ant
        # placement) so AA-vs-BB matches are a fair comparison.
        s = _new_state(seed=4)
        obs0 = json.loads(s.observation_string(0))
        obs2 = json.loads(s.observation_string(2))
        self.assertEqual(obs0["board"]["nest_position"], obs2["board"]["nest_position"])
        self.assertEqual(obs0["board"]["food_positions"], obs2["board"]["food_positions"])
        # Per-seat ant positions match (team A seat s == team B seat s).
        team_a_positions = [obs0["board"]["ant_positions"][str(pid)] for pid in (0, 1)]
        team_b_positions = [obs2["board"]["ant_positions"][str(pid)] for pid in (2, 3)]
        self.assertEqual(team_a_positions, team_b_positions)


class TurnHistoryTest(absltest.TestCase):
    """Each board records both teammates' moves so partners can see them."""

    def test_history_includes_both_seats(self):
        s = _new_state(seed=2)
        # Sequential order is [0, 2, 1, 3]: seat 0 plays "right" on each
        # board, then seat 1 plays "left" on each board.
        s.apply_action(4)  # player 0: right
        s.apply_action(4)  # player 2: right
        s.apply_action(3)  # player 1: left
        s.apply_action(3)  # player 3: left

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
    """Episode ends on max_turns cap OR when either board delivers all food."""

    def test_terminates_on_max_turns_cap(self):
        # All ants stay for the full cap; no food delivered, runs to length.
        s = _new_state(seed=3, max_turns=3, num_food=3)
        steps = 0
        while not s.is_terminal() and steps < 100:
            s.apply_action(0)  # stay
            steps += 1
        self.assertTrue(s.is_terminal())
        # max_turns rounds * num_ants_per_team (2) * num_teams (2) = 12.
        self.assertEqual(steps, 12)
        # No food was delivered.
        self.assertEqual(s.returns(), [0.0, 0.0, 0.0, 0.0])

    def test_terminates_when_any_board_delivers_all_food(self):
        # Hand-walk team A: move player 0 onto the only food then back to
        # the nest. Team B just stays. Game ends as soon as team A clears
        # its board.
        s = _new_state(seed=1, num_food=1, max_turns=100, grid_size=8)
        obs0 = json.loads(s.observation_string(0))
        food_r, food_c = obs0["board"]["food_positions"][0]
        nest_r, nest_c = obs0["board"]["nest_position"]
        ant_r, ant_c = obs0["board"]["ant_positions"]["0"]

        # Players act in order [0, 2, 1, 3]. We move player 0 each turn,
        # and have the other three stay.
        def _step_for_p0(action_value):
            # P0 acts, then P2/P1/P3 each stay.
            s.apply_action(action_value)  # P0
            s.apply_action(0)  # P2 stay
            s.apply_action(0)  # P1 stay
            s.apply_action(0)  # P3 stay

        # Walk P0 from ant -> food, then food -> nest. Use Manhattan moves.
        def _walk(r0, c0, r1, c1):
            nonlocal s
            while r0 != r1:
                action = 2 if r1 > r0 else 1  # DOWN or UP
                _step_for_p0(action)
                r0 += 1 if r1 > r0 else -1
                if s.is_terminal():
                    return
            while c0 != c1:
                action = 4 if c1 > c0 else 3  # RIGHT or LEFT
                _step_for_p0(action)
                c0 += 1 if c1 > c0 else -1
                if s.is_terminal():
                    return

        _walk(ant_r, ant_c, food_r, food_c)
        if not s.is_terminal():
            _walk(food_r, food_c, nest_r, nest_c)
        self.assertTrue(s.is_terminal())
        # Team A delivered the only food; team B delivered nothing.
        self.assertEqual(
            [b["food_collected"] for b in s._boards],
            [1, 0],
        )


class ScoringTest(absltest.TestCase):
    """Cooperative reward: both teammates get the team's food count."""

    def test_zero_returns_when_no_food_collected(self):
        s = _new_state(seed=1, max_turns=1, num_food=3)
        # max_turns=1 -> 4 interleaved steps total; all "stay".
        for _ in range(4):
            s.apply_action(0)
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.returns(), [0.0, 0.0, 0.0, 0.0])

    def test_teammates_receive_same_reward(self):
        # Drive a random game to terminal and confirm teammates share.
        s = _new_state(seed=5, max_turns=20)
        rng = random.Random(0)
        while not s.is_terminal():
            _step_random(s, rng)
        rewards = s.returns()
        self.assertEqual(rewards[0], rewards[1])  # team A
        self.assertEqual(rewards[2], rewards[3])  # team B

    def test_team_totals_match_food_counts(self):
        s = _new_state(seed=5, max_turns=20)
        rng = random.Random(0)
        while not s.is_terminal():
            _step_random(s, rng)
        obs0 = json.loads(s.observation_string(0))
        self.assertIn("team_totals", obs0)
        self.assertIn("winning_team", obs0)
        self.assertEqual(len(obs0["team_totals"]), 2)
        # team_totals reflects each board's food_collected.
        self.assertEqual(
            obs0["team_totals"],
            [b["food_collected"] for b in s._boards],
        )
        # returns[pid] equals team_totals[team_of(pid)].
        returns = obs0["returns"]
        self.assertEqual(returns[0], float(obs0["team_totals"][0]))
        self.assertEqual(returns[2], float(obs0["team_totals"][1]))
        # Winner consistency.
        a, b = obs0["team_totals"]
        if a > b:
            self.assertEqual(obs0["winning_team"], 0)
        elif b > a:
            self.assertEqual(obs0["winning_team"], 1)
        else:
            self.assertEqual(obs0["winning_team"], "draw")


class TerminalRevealTest(absltest.TestCase):
    """At terminal, both boards become visible on every player's view."""

    def test_terminal_reveal(self):
        s = _new_state(seed=5, max_turns=1)
        for _ in range(4):
            s.apply_action(0)  # stay
        obs0 = json.loads(s.observation_string(0))
        self.assertIn("boards", obs0)
        self.assertEqual(len(obs0["boards"]), 2)
        self.assertIn("returns", obs0)
        self.assertIn("team_totals", obs0)


class KaggleEnvIntegrationTest(absltest.TestCase):
    """Full episode through the kaggle env wrapper with random agents."""

    def test_random_agents_run_to_completion(self):
        env = make(
            "open_spiel_ant_foraging_arena",
            configuration={
                "openSpielGameParameters": {"seed": 11, "max_turns": 10},
                "includeLegalActions": True,
            },
        )
        env.run(["random", "random", "random", "random"])
        last = env.steps[-1]
        for agent in last:
            self.assertEqual(agent["status"], "DONE")
        # Teammates share rewards.
        self.assertEqual(last[0]["reward"], last[1]["reward"])
        self.assertEqual(last[2]["reward"], last[3]["reward"])
        for agent in last:
            self.assertIsInstance(agent["reward"], float)


if __name__ == "__main__":
    absltest.main()
