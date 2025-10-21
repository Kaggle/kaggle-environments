import json
import pathlib

import pokerkit
from open_spiel.python.games import pokerkit_wrapper
import pyspiel

from absl.testing import absltest

from kaggle_environments import make

from . import open_spiel_env

# Expected that not all pyspiel registered games can be registered as Kaggle
# envs (e.g. does not yet support simultaneous move games), but should register
# at least this many
_REGISTERED_GAMES_THRESHOLD = 50


class OpenSpielEnvTest(absltest.TestCase):
    def test_envs_load(self):
        envs = open_spiel_env._register_game_envs([game_type.short_name for game_type in pyspiel.registered_games()])
        self.assertTrue(len(envs) > _REGISTERED_GAMES_THRESHOLD)

    def test_tic_tac_toe_agent_playthrough(self):
        envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_tic_tac_toe")
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))

    def test_tic_tac_toe_manual_playthrough(self):
        envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
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
        envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
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
        envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        kaggle_state = env.step([{"submission": 0}, {"submission": -1}])
        serialize_game_and_state = kaggle_state[1]["observation"]["serializedGameAndState"]
        game, state = pyspiel.deserialize_game_and_state(serialize_game_and_state)
        self.assertEqual(game.get_type().short_name, "tic_tac_toe_proxy")
        self.assertEqual(state.history(), [0])

    def test_agent_error(self):
        envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
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
        env = make('open_spiel_repeated_poker')
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
        current_hand_dict = json.loads(
            state_dict["current_universal_poker_json"])
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
        envs = open_spiel_env._register_game_envs(
            [pokerkit_game_str]
        )
        env = make(envs['open_spiel_python_repeated_pokerkit'])
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
        env = make('open_spiel_python_repeated_pokerkit')
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
            "setNumHands": num_hands,
        }
        env = make(
            'open_spiel_repeated_poker',
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

if __name__ == "__main__":
    absltest.main()
