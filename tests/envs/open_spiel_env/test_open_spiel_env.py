import json
import pathlib

import pokerkit  # noqa: F401
import pyspiel
from absl.testing import absltest
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env

TEST_REPEATED_POKER_GAME_STRING = open_spiel_env.DEFAULT_REPEATED_POKER_GAME_STRING.replace(
    "calcOddsNumSims=1000000",
    "calcOddsNumSims=1",
)


# Expected that not all pyspiel registered games can be registered as Kaggle
# envs (e.g. does not yet support simultaneous move games), but should register
# at least this many
_REGISTERED_GAMES_THRESHOLD = 50


class OpenSpielEnvTest(absltest.TestCase):
    def test_envs_load(self):
        envs = open_spiel_env._register_game_envs([game_type.short_name for game_type in pyspiel.registered_games()])
        self.assertTrue(len(envs) > _REGISTERED_GAMES_THRESHOLD)

    def test_tic_tac_toe_agent_playthrough(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_tic_tac_toe")
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))

    def test_tic_tac_toe_manual_playthrough(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 0}, {"submission": -1}])
        env.step([{"submission": -1}, {"submission": 1}])
        env.step([{"submission": 3}, {"submission": -1}])
        env.step([{"submission": -1}, {"submission": 4}])
        env.step([{"submission": 6}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])

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

    def test_chess_openings_manually_configured(self):
        open_spiel_env._register_game_envs(["chess"])
        openings_path = pathlib.Path(
            open_spiel_env.GAMES_DIR,
            "chess/openings.jsonl",
        )
        self.assertTrue(openings_path.is_file())
        with open(openings_path, "r", encoding="utf-8") as f:
            for line in f:
                opening = json.loads(line)
                config = {
                    "initialActions": opening.pop("initialActions"),
                    "metadata": opening,
                }
                env = make(
                    "open_spiel_chess",
                    config,
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
                obs = env.state[0]["observation"]
                _, state = pyspiel.deserialize_game_and_state(obs["serializedGameAndState"])
                self.assertEqual(str(state), opening["fen"])
                self.assertEqual(str(state), env.toJSON()["configuration"]["metadata"]["fen"])

    def test_chess_openings_configured_with_seed(self):
        open_spiel_env._register_game_envs(["chess"])
        config = {
            "useImage": True,
            "seed": 1,
        }
        env = make(
            "open_spiel_chess",
            config,
            debug=True,
        )
        env.reset()
        # Image config is loaded during setup step.
        self.assertFalse("imageConfig" in env.configuration)
        # Setup step
        env.step(
            [
                {"submission": pyspiel.INVALID_ACTION},
                {"submission": pyspiel.INVALID_ACTION},
            ]
        )
        self.assertTrue("imageConfig" in env.configuration)
        self.assertEqual(env.configuration["imageConfig"]["color"], "blue")
        self.assertEqual(env.configuration["imageConfig"]["pieceSet"], "cardinal")
        self.assertTrue("imageConfig" in env.state[0]["observation"])

    def test_default_repeated_poker(self):
        env = make("open_spiel_repeated_poker", {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING})
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Default repeated_poker now includes hand odds calculations which take
        # ~1-2s per action, so we avoid stepping until the end of the game.
        for i in range(2):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertEqual(len(env.os_state.acpc_hand_histories()), 2)
        state_dict = json.loads(str(env.os_state))
        self.assertTrue("current_universal_poker_json" in state_dict)
        current_hand_dict = json.loads(state_dict["current_universal_poker_json"])
        self.assertTrue("odds" in current_hand_dict)
        self.assertEqual(len(current_hand_dict["odds"]), 4)

    def test_repeated_pokerkit(self):
        pokerkit_game_str = (
            "python_repeated_pokerkit("
            "bet_size_schedule=,"
            "blind_schedule=,"
            "bring_in_schedule=,"
            "first_button_player=-1,"
            "max_num_hands=20,"
            "pokerkit_game_params=python_pokerkit_wrapper("
            "blinds=1 2,"
            "num_players=2,"
            "stack_sizes=200 200,"
            "variant=NoLimitTexasHoldem),"
            "reset_stacks=True,"
            "rotate_dealer=True)"
        )
        envs = open_spiel_env._register_game_envs([pokerkit_game_str])
        env = make(envs["open_spiel_python_repeated_pokerkit"])
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for i in range(20):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [0.0, 0.0])

    def test_default_repeated_pokerkit_loads(self):
        env = make("open_spiel_python_repeated_pokerkit")
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for i in range(100):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [0.0, 0.0])

    def test_poker_set_num_hands(self):
        num_hands = 2
        config = {
            "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
            "setNumHands": num_hands,
        }
        env = make(
            "open_spiel_repeated_poker",
            config,
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for i in range(num_hands):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [0.0, 0.0])

    def test_repeated_poker_preset_hands_replay(self):
        base_env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "setNumHands": 2},
            debug=True,
        )
        base_env.reset()
        base_env.step([{"submission": -1}, {"submission": -1}])
        for i in range(2):
            if i % 2 == 0:
                base_env.step([{"submission": -1}, {"submission": 0}])
            else:
                base_env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(base_env.done)
        base_history = [int(action) for action in base_env.info["actionHistory"]]
        replay_state = base_env.os_game.new_initial_state()
        preset_hands: list[list[int]] = []
        for action in base_history:
            if replay_state.is_chance_node():
                hand_idx = len(replay_state.acpc_hand_histories())
                while len(preset_hands) <= hand_idx:
                    preset_hands.append([])
                preset_hands[hand_idx].append(action)
            replay_state.apply_action(action)
        if preset_hands and preset_hands[0]:
            preset_hands[0].append(preset_hands[0][-1])
        preset_env = make(
            "open_spiel_repeated_poker",
            {
                "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
                "setNumHands": 2,
                "presetHands": [hand[:] for hand in preset_hands],
            },
            debug=True,
        )
        preset_env.reset()
        preset_env.step([{"submission": -1}, {"submission": -1}])
        for i in range(2):
            if i % 2 == 0:
                preset_env.step([{"submission": -1}, {"submission": 0}])
            else:
                preset_env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(preset_env.done)
        self.assertEqual(
            [int(action) for action in preset_env.info["actionHistory"]],
            base_history,
        )
        self.assertEqual(
            preset_env.info["presetHands"],
            [hand[:] for hand in preset_hands],
        )
        self.assertLess(
            preset_env.info["presetHandsState"]["next_index"][0],
            len(preset_env.info["presetHands"][0]),
        )

    def test_repeated_poker_preset_hands_runs_out(self):
        env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "presetHands": [[0]]},
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "presetHands"):
            env.step([{"submission": -1}, {"submission": -1}])

    def test_repeated_poker_preset_hands_conflicts_with_use_openings(self):
        env = make(
            "open_spiel_repeated_poker",
            {
                "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
                "presetHands": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
                "useOpenings": True,
            },
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "useOpenings"):
            env.step([{"submission": -1}, {"submission": -1}])

    def test_repeated_poker_load_preset_hands_loads_file(self):
        env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "loadPresetHands": True, "seed": 0},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        self.assertIn("presetHands", env.configuration)
        preset_hands = env.configuration["presetHands"]
        self.assertEqual(len(preset_hands), 100)
        self.assertTrue(all(len(hand) == 9 for hand in preset_hands))

    def test_repeated_poker_load_preset_hands_requires_seed(self):
        env = make(
            "open_spiel_repeated_poker",
            {"openSpielGameString": TEST_REPEATED_POKER_GAME_STRING, "loadPresetHands": True},
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "seed"):
            env.step([{"submission": -1}, {"submission": -1}])

    def test_repeated_poker_load_preset_hands_conflicts_with_manual(self):
        env = make(
            "open_spiel_repeated_poker",
            {
                "openSpielGameString": TEST_REPEATED_POKER_GAME_STRING,
                "loadPresetHands": True,
                "seed": 0,
                "presetHands": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
            },
            debug=True,
        )
        env.reset()
        with self.assertRaisesRegex(ValueError, "loadPresetHands"):
            env.step([{"submission": -1}, {"submission": -1}])

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


if __name__ == "__main__":
    absltest.main()
