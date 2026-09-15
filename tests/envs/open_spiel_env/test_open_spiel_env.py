"""Framework-level tests for the open_spiel_env wrapper.

Per-game tests live alongside each game in
``tests/envs/open_spiel_env/games/<name>/env_test.py``. This file covers
behavior of the open_spiel_env interpreter itself: env registration, strict
mode, agent error / invalid action handling, game-parameter configuration,
``includeLegalActions``, simultaneous-game dispatch, and serialized state.
"""

import json

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env

# Expected that not all pyspiel registered games can be registered as Kaggle
# envs, but should register at least this many
_REGISTERED_GAMES_THRESHOLD = 50


def _started_arena(configuration):
    """A hanabi_arena env stepped once, so the game is built and readable.

    The interpreter builds ``env.os_game`` on its first step, not at reset,
    so the merged parameters only exist once a step has run.
    """
    open_spiel_env._register_game_envs(["hanabi_arena"])
    env = make("open_spiel_hanabi_arena", configuration, debug=True)
    env.reset()
    env.step([{"submission": -1}] * 4)
    return env


class OpenSpielEnvTest(absltest.TestCase):
    def test_envs_load(self):
        envs = open_spiel_env._register_game_envs([game_type.short_name for game_type in pyspiel.registered_games()])
        self.assertTrue(len(envs) > _REGISTERED_GAMES_THRESHOLD)

    def test_strict_mode_agent_error_keeps_per_player_status(self):
        env = make(
            "open_spiel_dark_hex",
            configuration={"strictMode": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Player 0 signals an internal error; player 1 plays normally.
        env.step([{"submission": open_spiel_env.AGENT_ERROR_ACTION}, {"submission": -1}])
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        # Offender keeps its natural ERROR status; the other agent gets DONE+win.
        self.assertEqual(playthrough["statuses"], ["ERROR", "DONE"])
        # Offender's reward is nulled by core.py because status is ERROR;
        # the other agent receives the winning reward.
        self.assertIsNone(playthrough["rewards"][0])
        self.assertEqual(playthrough["rewards"][1], -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)

    def test_strict_mode_extra_action_field_is_invalid(self):
        env = make(
            "open_spiel_dark_hex",
            configuration={"strictMode": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Extra "thoughts" key violates the strict-mode action schema.
        env.step([{"submission": 0, "thoughts": "..."}, {"submission": -1}])
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        # INVALID still maps to DONE (matches lenient behavior; kaggleazure
        # already accepts INVALID statuses through the open_spiel carveout).
        self.assertEqual(playthrough["statuses"], ["DONE", "DONE"])
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )

    def test_action_string_populated_for_visualizer(self):
        # Visualizers (e.g. goTransformer) read actionString off each player's
        # action dict to render moves. The env must surface it even when the
        # agent only submits {"submission": int}.
        env = make("open_spiel_dark_hex", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup.
        env.step([{"submission": 0}, {"submission": -1}])
        steps = env.toJSON()["steps"]
        played_action = steps[2][0]["action"]
        self.assertIn("actionString", played_action)
        self.assertTrue(played_action["actionString"])

    def test_invalid_action(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        for i in range(5):  # Try repeatedly applying an illegal action
            env.step(
                [
                    {"submission": pyspiel.INVALID_ACTION},
                    {"submission": pyspiel.INVALID_ACTION},
                ]
            )
            if env.done:
                break
        self.assertEqual(i, 1)  # Zeroth step is setup step, should fail next step.
        json = env.toJSON()
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
        self.assertEqual(
            json["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )

    def test_forfeiting_players_without_teams(self):
        # Most games have no teams, so a forfeit stays with the offender and
        # every opponent keeps the winning reward -- the behavior the
        # tic-tac-toe invalid-action test above pins end to end.
        class _NoTeams:
            pass

        self.assertEqual(open_spiel_env._forfeiting_players(_NoTeams(), 0, 4), {0})

    def test_forfeiting_players_with_teams(self):
        class _Teams:
            def team_of(self, player):
                return player // 2

        self.assertEqual(open_spiel_env._forfeiting_players(_Teams(), 3, 4), {2, 3})

    def test_forfeiting_players_falls_back_when_team_of_misbehaves(self):
        # A broken hook must not take the whole episode down with it; scoping
        # the loss too narrowly is the same as the old behavior, which is a
        # safe place to land.
        class _Raises:
            def team_of(self, player):
                raise RuntimeError("boom")

        class _ReturnsNone:
            def team_of(self, player):
                return None

        self.assertEqual(open_spiel_env._forfeiting_players(_Raises(), 1, 4), {1})
        self.assertEqual(open_spiel_env._forfeiting_players(_ReturnsNone(), 1, 4), {1})

    def test_serialized_game_and_state(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        kaggle_state = env.step([{"submission": 0}, {"submission": -1}])
        serialize_game_and_state = kaggle_state[1]["observation"]["serializedGameAndState"]
        game, state = pyspiel.deserialize_game_and_state(serialize_game_and_state)
        self.assertEqual(game.get_type().short_name, "tic_tac_toe")
        self.assertEqual(state.history(), [0])

    def test_agent_error(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        # Setup step
        env.step(
            [
                {"submission": pyspiel.INVALID_ACTION},
                {"submission": pyspiel.INVALID_ACTION},
            ]
        )
        env.step(
            [
                {"submission": open_spiel_env.AGENT_ERROR_ACTION},
                {"submission": pyspiel.INVALID_ACTION},
            ]
        )
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [None, None])
        self.assertEqual(json["statuses"], ["ERROR", "ERROR"])

    def test_initial_actions(self):
        open_spiel_env._register_game_envs(["tic_tac_toe"])
        env = make(
            "open_spiel_tic_tac_toe",
            {"initialActions": [0, 1, 3, 4]},
            debug=True,
        )
        env.reset()
        # Setup step
        env.step(
            [
                {"submission": pyspiel.INVALID_ACTION},
                {"submission": pyspiel.INVALID_ACTION},
            ]
        )
        env.step(
            [
                {"submission": 2},
                {"submission": pyspiel.INVALID_ACTION},
            ]
        )
        env.step(
            [
                {"submission": pyspiel.INVALID_ACTION},
                {"submission": 7},
            ]
        )
        self.assertTrue(env.done)
        json_playthrough = env.toJSON()
        self.assertEqual(json_playthrough["rewards"], [-1, 1])

    def test_game_params_override(self):
        """Test that openSpielGameParameters can override default params."""
        open_spiel_env._register_game_envs(["go"])
        env = make(
            "open_spiel_go",
            {"openSpielGameParameters": {"board_size": 19}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        self.assertEqual(env.os_game.get_parameters()["board_size"], 19)
        # Default komi should be preserved
        self.assertEqual(env.os_game.get_parameters()["komi"], 7.5)

    def test_game_string_override(self):
        """Test that openSpielGameString can specify game params."""
        open_spiel_env._register_game_envs(["go"])
        env = make(
            "open_spiel_go",
            {"openSpielGameString": "go(board_size=13)"},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        self.assertEqual(env.os_game.get_parameters()["board_size"], 13)

    def test_game_string_with_params_override(self):
        """Test that openSpielGameParameters overrides params from game string."""
        open_spiel_env._register_game_envs(["go"])
        env = make(
            "open_spiel_go",
            {
                "openSpielGameString": "go(board_size=19)",
                "openSpielGameParameters": {"komi": 6.5},
            },
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        # board_size from string
        self.assertEqual(env.os_game.get_parameters()["board_size"], 19)
        # komi from params
        self.assertEqual(env.os_game.get_parameters()["komi"], 6.5)

    def test_params_override_string_param(self):
        """Test that explicit params override the same param in game string."""
        open_spiel_env._register_game_envs(["go"])
        env = make(
            "open_spiel_go",
            {
                "openSpielGameString": "go(board_size=9)",
                "openSpielGameParameters": {"board_size": 19},
            },
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        # params should win over string
        self.assertEqual(env.os_game.get_parameters()["board_size"], 19)

    def test_config_seed_reaches_a_self_dealing_game(self):
        """configuration.seed must set a `seed` game parameter, not just chance_rng.

        A game that deals itself (the 2v2 arenas) exposes no chance nodes, so
        env.chance_rng -- all configuration["seed"] otherwise feeds -- never
        touches it. Without this the parameter sits at its default and every
        episode of the tournament runs the identical board.
        """
        env = _started_arena({"seed": 7})
        self.assertEqual(env.os_game.get_parameters()["seed"], 7)
        self.assertIn("seed=7", env.info["openSpielGameStringResolved"])

    def test_config_seed_does_not_invent_a_param(self):
        """Games without a `seed` parameter must be left alone."""
        open_spiel_env._register_game_envs(["go"])
        env = make("open_spiel_go", {"seed": 7}, debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        self.assertNotIn("seed", env.os_game.get_parameters())

    def test_explicit_game_param_seed_beats_config_seed(self):
        """A caller pinning a specific deal keeps it."""
        env = _started_arena({"seed": 7, "openSpielGameParameters": {"seed": 3}})
        self.assertEqual(env.os_game.get_parameters()["seed"], 3)

    def test_config_seed_varies_the_arena_deal(self):
        """The point of the plumbing: different seeds must deal differently."""

        def teammate_hand(seed):
            env = _started_arena({"seed": seed})
            observation = json.loads(env.os_state.observation_string(0))
            return json.dumps(observation["table"]["hands"][1]["cards"])

        self.assertLen({teammate_hand(seed) for seed in (1, 2, 3)}, 3)
        self.assertEqual(teammate_hand(1), teammate_hand(1))

    def test_an_unseeded_run_still_varies_the_deal(self):
        """No configuration seed means an ARBITRARY deal, not deal zero.

        A self-dealing game whose `seed` parameter is left at its default runs
        the identical board every episode, so a tournament would score every
        pairing on one deal -- and whichever side that single board happens to
        favour is baked into the Elo. Draw one instead.
        """

        def teammate_hand():
            env = _started_arena({})
            observation = json.loads(env.os_state.observation_string(0))
            return json.dumps(observation["table"]["hands"][1]["cards"])

        self.assertGreater(len({teammate_hand() for _ in range(6)}), 1)

    def test_a_hiding_game_withholds_the_serialized_state(self):
        """hanabi_arena's blob rebuilds every hidden hand, so agents lose it.

        Every other game keeps shipping it (see
        test_serialized_game_and_state), so the hook is an opt-out, not a
        change of default.
        """
        env = _started_arena({})
        for agent in env.state:
            self.assertNotIn("serializedGameAndState", agent["observation"])
            self.assertIn("observationString", agent["observation"])

    def test_a_raising_hook_is_treated_as_no_opt_out(self):
        """A broken hook must cost the blob, not the episode."""

        class _Boom:
            def hides_state_from_agents(self):
                raise RuntimeError("boom")

        self.assertFalse(open_spiel_env._hides_state_from_agents(_Boom()))
        self.assertFalse(open_spiel_env._hides_state_from_agents(object()))

    def test_resolved_game_string(self):
        """Test that openSpielGameStringResolved shows the actual game config."""
        open_spiel_env._register_game_envs(["go"])
        env = make(
            "open_spiel_go",
            {
                "openSpielGameString": "go(board_size=19)",
                "openSpielGameParameters": {"komi": 6.5},
            },
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        # Original string preserved
        self.assertEqual(env.configuration["openSpielGameString"], "go(board_size=19)")
        # Resolved string shows actual game with merged params
        resolved = env.info["openSpielGameStringResolved"]
        self.assertIn("board_size=19", resolved)
        self.assertIn("komi=6.5", resolved)

    def test_include_legal_actions(self):
        """Test that legalActions is controlled by includeLegalActions config."""
        # Default: legalActions not included
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        obs = env.state[0].observation
        self.assertIsNone(getattr(obs, "legalActions", None))
        self.assertIsNone(getattr(obs, "legalActionStrings", None))
        self.assertIsNotNone(obs.serializedGameAndState)

        # With includeLegalActions=True: legalActions included
        env = make("open_spiel_tic_tac_toe", {"includeLegalActions": True}, debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        obs = env.state[0].observation
        self.assertIsNotNone(obs.legalActions)
        self.assertIsNotNone(obs.legalActionStrings)
        self.assertEqual(len(obs.legalActions), 9)  # All 9 squares available

    # --- Simultaneous move game dispatch ---

    def test_simultaneous_invalid_action(self):
        open_spiel_env._register_game_envs(["goofspiel(num_cards=4,points_order=descending,returns_type=total_points)"])
        env = make("open_spiel_goofspiel", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # Submit an invalid action (999 is not a legal bid).
        env.step([{"submission": 999}, {"submission": 0}])
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
        # Player 0 submitted invalid, so gets the penalty.
        self.assertEqual(json["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)
        self.assertEqual(json["rewards"][1], -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)

    def test_simultaneous_agent_error(self):
        open_spiel_env._register_game_envs(["goofspiel(num_cards=4,points_order=descending,returns_type=total_points)"])
        env = make("open_spiel_goofspiel", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step(
            [
                {"submission": open_spiel_env.AGENT_ERROR_ACTION},
                {"submission": 0},
            ]
        )
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [None, None])
        self.assertEqual(json["statuses"], ["ERROR", "ERROR"])


if __name__ == "__main__":
    absltest.main()
