import json
import pathlib
import random

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
# envs, but should register at least this many
_REGISTERED_GAMES_THRESHOLD = 50


class OpenSpielEnvTest(absltest.TestCase):
    def test_envs_load(self):
        envs = open_spiel_env._register_game_envs([game_type.short_name for game_type in pyspiel.registered_games()])
        self.assertTrue(len(envs) > _REGISTERED_GAMES_THRESHOLD)

    def test_dark_hex_agent_playthrough(self):
        env = make(
            "open_spiel_dark_hex",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_dark_hex")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertEqual(sorted(rewards), [-1.0, 1.0])

    def test_dark_hex_manual_playthrough(self):
        env = make("open_spiel_dark_hex", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 1}, {"submission": -1}])  # p0: b1
        env.step([{"submission": -1}, {"submission": 0}])  # p1: a1
        env.step([{"submission": 4}, {"submission": -1}])  # p0: b2
        env.step([{"submission": -1}, {"submission": 3}])  # p1: a2
        env.step([{"submission": 7}, {"submission": -1}])  # p0: b3 (wins)
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "x")

    def test_dark_hex_observation_hides_opponent(self):
        env = make("open_spiel_dark_hex", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 4}, {"submission": -1}])  # p0: b2 (center)
        # After p0's move, p1's view should not see the new x piece.
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p1["board"], [["."] * 3] * 3)
        # p0's view should see their own x piece.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["board"][1][1], "x")

    def test_dark_hex_invalid_action(self):
        env = make("open_spiel_dark_hex", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )

    def test_gin_rummy_agent_playthrough(self):
        env = make(
            "open_spiel_gin_rummy",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_gin_rummy")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_gin_rummy_observation_is_json(self):
        env = make(
            "open_spiel_gin_rummy",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # After dealing, it is player 0's turn (FirstUpcard phase).
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["phase"], "FirstUpcard")
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertFalse(obs_p0["is_terminal"])
        self.assertEqual(obs_p0["knock_card"], 10)
        self.assertEqual(obs_p0["stock_size"], 31)
        self.assertIsNotNone(obs_p0["upcard"])
        # Player 0 sees their own 10-card hand; opponent's hand is hidden.
        self.assertEqual(len(obs_p0["hands"]["0"]), 10)
        self.assertEqual(obs_p0["hands"]["1"], [])
        self.assertIsNotNone(obs_p0["deadwood"]["0"])
        self.assertIsNone(obs_p0["deadwood"]["1"])

    def test_gin_rummy_observation_hides_opponent(self):
        env = make("open_spiel_gin_rummy", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Each player should only see their own hand in their observation.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(len(obs_p0["hands"]["0"]), 10)
        self.assertEqual(obs_p0["hands"]["1"], [])
        self.assertEqual(obs_p1["hands"]["0"], [])
        self.assertEqual(len(obs_p1["hands"]["1"]), 10)

    def test_gin_rummy_invalid_action(self):
        env = make("open_spiel_gin_rummy", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # In FirstUpcard phase only actions 52 (Draw upcard) and 54 (Pass)
        # are legal, so 0 (the As card) is an invalid action.
        env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )

    def test_y_agent_playthrough(self):
        env = make(
            "open_spiel_y",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_y")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertEqual(sorted(rewards), [-1.0, 1.0])

    def test_y_manual_playthrough(self):
        env = make(
            "open_spiel_y",
            configuration={"openSpielGameParameters": {"board_size": 8}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # P0 connects the left column (a1..a8), touching all three sides.
        # Action encoding: row * 8 + col, with rows/cols 0-indexed.
        moves = [0, 1, 8, 2, 16, 3, 24, 4, 32, 5, 40, 6, 48, 7, 56]
        for i, action in enumerate(moves):
            if i % 2 == 0:
                env.step([{"submission": action}, {"submission": -1}])
            else:
                env.step([{"submission": -1}, {"submission": action}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "x")
        self.assertEqual(final_obs["board_size"], 8)
        self.assertEqual(final_obs["last_move"], "a8")
        self.assertEqual(final_obs["board"][0][0], "x")
        self.assertEqual(final_obs["board"][7][0], "x")

    def test_y_invalid_action(self):
        env = make("open_spiel_y", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )

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

    def test_amazons_agent_playthrough(self):
        env = make("open_spiel_amazons", debug=True)
        env.run(["random", "random"])
        json_data = env.toJSON()
        self.assertEqual(json_data["name"], "open_spiel_amazons")
        self.assertTrue(all(status == "DONE" for status in json_data["statuses"]))

    def test_amazons_manual_playthrough(self):
        # Walks player 0 (X) through one full Amazons turn (from -> to -> shoot)
        # by picking the first legal action at each sub-action, and verifies
        # the proxy advances `phase` through all three values and then resets
        # to "from" for player O. Picking from `legalActions` keeps this test
        # robust to pyspiel version differences in the starting layout.
        env = make("open_spiel_amazons", {"includeLegalActions": True}, debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for expected_next_phase in ("to", "shoot", "from"):
            legal = env.state[0]["observation"]["legalActions"]
            env.step([{"submission": legal[0]}, {"submission": -1}])
            obs = json.loads(env.state[0]["observation"]["observationString"])
            if expected_next_phase == "from":
                # After three sub-actions it is player 1's (O) turn.
                obs = json.loads(env.state[1]["observation"]["observationString"])
                self.assertEqual(env.state[0]["status"], "INACTIVE")
                self.assertEqual(env.state[1]["status"], "ACTIVE")
                self.assertEqual(obs["current_player"], "o")
            self.assertEqual(obs["phase"], expected_next_phase)
            self.assertFalse(obs["is_terminal"])
        # An arrow (burned square) was fired somewhere on the board.
        self.assertTrue(any("#" in row for row in obs["board"]))

    def test_amazons_terminal_state(self):
        # Drive an Amazons game to natural termination using only legal random
        # actions, then verify the proxy reports a winner and clears the phase.
        rng = random.Random(0)
        game = pyspiel.load_game("amazons_proxy")
        state = game.new_initial_state()
        while not state.is_terminal():
            state.apply_action(rng.choice(state.legal_actions()))
        obs = json.loads(state.observation_string(0))
        self.assertTrue(obs["is_terminal"])
        self.assertIn(obs["winner"], ("x", "o"))
        self.assertIsNone(obs["phase"])

    def test_amazons_invalid_action(self):
        env = make("open_spiel_amazons", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        json_data = env.toJSON()
        self.assertEqual(json_data["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)
        self.assertEqual(json_data["rewards"][1], -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)

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

    # --- Simultaneous move game tests ---

    def test_goofspiel_agent_playthrough(self):
        open_spiel_env._register_game_envs(["goofspiel(num_cards=4,points_order=descending,returns_type=total_points)"])
        env = make("open_spiel_goofspiel", debug=True)
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_goofspiel")
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
        # Both players should have rewards (total_points mode).
        self.assertEqual(len(json["rewards"]), 2)
        self.assertTrue(all(r is not None for r in json["rewards"]))

    def test_goofspiel_manual_playthrough(self):
        open_spiel_env._register_game_envs(["goofspiel(num_cards=4,points_order=descending,returns_type=total_points)"])
        env = make("open_spiel_goofspiel", debug=True)
        env.reset()
        # Initial setup step.
        env.step([{"submission": -1}, {"submission": -1}])
        # After setup, both players should be ACTIVE (simultaneous node).
        self.assertEqual(env.state[0]["status"], "ACTIVE")
        self.assertEqual(env.state[1]["status"], "ACTIVE")
        # Play all 4 rounds: both players submit actions each step.
        # With descending point order and 4 cards, there are 4 bidding rounds.
        # Legal actions are card indices (0-3 initially).
        for _ in range(4):
            if env.done:
                break
            env.step([{"submission": 0}, {"submission": 0}])
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))

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

    def test_matching_pennies_manual(self):
        open_spiel_env._register_game_envs(["matching_pennies_3p"])
        env = make("open_spiel_matching_pennies_3p", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}, {"submission": -1}])  # Setup.
        # All 3 players should be ACTIVE.
        for pid in range(3):
            self.assertEqual(env.state[pid]["status"], "ACTIVE")
        # All play Heads (action 0).
        env.step([{"submission": 0}, {"submission": 0}, {"submission": 0}])
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
        self.assertEqual(len(json["rewards"]), 3)

    def test_repeated_prisoners_dilemma_agent_playthrough(self):
        """Test repeated Prisoner's Dilemma with random agents for 10 rounds."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_repeated_game")
        self.assertTrue(all(s == "DONE" for s in json["statuses"]))
        self.assertEqual(len(json["rewards"]), 2)
        self.assertTrue(all(r is not None for r in json["rewards"]))

    def test_repeated_prisoners_dilemma_mutual_cooperate(self):
        """Both players cooperate every round. Expected reward: 5 * 10 = 50 each."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        for _ in range(10):
            self.assertFalse(env.done)
            self.assertEqual(env.state[0]["status"], "ACTIVE")
            self.assertEqual(env.state[1]["status"], "ACTIVE")
            env.step([{"submission": 0}, {"submission": 0}])  # Both cooperate.
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [50.0, 50.0])

    def test_repeated_prisoners_dilemma_mutual_defect(self):
        """Both players defect every round. Expected reward: 1 * 10 = 10 each."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        for _ in range(10):
            env.step([{"submission": 1}, {"submission": 1}])  # Both defect.
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [10.0, 10.0])

    def test_repeated_prisoners_dilemma_asymmetric(self):
        """P0 always cooperates, P1 always defects. P0 gets 0, P1 gets 100."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        for _ in range(10):
            env.step([{"submission": 0}, {"submission": 1}])  # P0 cooperate, P1 defect.
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [0.0, 100.0])

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
