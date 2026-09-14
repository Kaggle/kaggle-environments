"""Env-level tests for open_spiel_hanabi."""

import json

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


def _obs(env, player):
    return json.loads(env.state[player]["observation"]["observationString"])


class HanabiEnvTest(absltest.TestCase):
    def test_hanabi_agent_playthrough(self):
        env = make(
            "open_spiel_hanabi",
            configuration={"includeLegalActions": True, "seed": 0},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_hanabi")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Hanabi is cooperative, so both seats always score identically.
        self.assertEqual(playthrough["rewards"][0], playthrough["rewards"][1])

    def test_hanabi_initial_observation(self):
        env = make("open_spiel_hanabi", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        obs = _obs(env, 0)
        self.assertEqual(obs["num_players"], 2)
        self.assertEqual(obs["current_player"], 0)
        self.assertFalse(obs["is_terminal"])
        self.assertEqual(obs["outcome"], "in_progress")
        self.assertEqual(obs["life_tokens"], 3)
        self.assertEqual(obs["info_tokens"], 8)
        self.assertEqual(obs["score"], 0)
        self.assertEqual(obs["max_score"], 25)
        self.assertEqual(obs["fireworks"], {"R": 0, "Y": 0, "G": 0, "W": 0, "B": 0})
        self.assertEqual(obs["discards"], [])
        # Two players draw 5 cards each out of the 50-card deck.
        self.assertEqual(obs["deck_size"], 40)
        self.assertLen(obs["hands"], 2)
        for hand in obs["hands"]:
            self.assertLen(hand["cards"], 5)

    def test_hanabi_observation_hides_own_hand(self):
        """Each player sees every hand but their own -- the core of Hanabi."""
        env = make("open_spiel_hanabi", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for observer in (0, 1):
            obs = _obs(env, observer)
            self.assertEqual(obs["observer"], observer)
            # Hands are keyed by absolute player index, sorted.
            self.assertEqual([hand["player"] for hand in obs["hands"]], [0, 1])
            own = obs["hands"][observer]
            other = obs["hands"][1 - observer]
            self.assertTrue(own["is_observer"])
            self.assertFalse(other["is_observer"])
            # Own cards are hidden; the teammate's are fully visible.
            self.assertTrue(all(card["card"] is None for card in own["cards"]))
            self.assertTrue(all(card["card"] is not None for card in other["cards"]))
        # Each player sees the other's actual hand, so the two views disagree.
        seen_by_0 = [card["card"] for card in _obs(env, 0)["hands"][1]["cards"]]
        seen_by_1 = [card["card"] for card in _obs(env, 1)["hands"][1]["cards"]]
        self.assertTrue(all(card is not None for card in seen_by_0))
        self.assertTrue(all(card is None for card in seen_by_1))

    def test_hanabi_hint_updates_knowledge_and_spends_token(self):
        """Revealing a color tells the teammate about the matching cards."""
        env = make(
            "open_spiel_hanabi",
            configuration={"includeLegalActions": True, "seed": 0},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        before = _obs(env, 0)
        self.assertTrue(all(card["hinted_color"] is None for card in before["hands"][1]["cards"]))

        # Find a legal "Reveal player +1 color C" action and take it.
        hint = next(a for a in before["legal_actions"] if "color" in a["label"])
        color = hint["label"].rsplit(" ", 1)[1].rstrip(")")
        env.step([{"submission": hint["action"]}, {"submission": -1}])

        after = _obs(env, 1)
        self.assertEqual(after["info_tokens"], 7)  # A hint costs one info token.
        self.assertEqual(after["life_tokens"], 3)  # Hints never cost a life.
        hinted = [card for card in after["hands"][1]["cards"] if card["hinted_color"] == color]
        self.assertNotEmpty(hinted)
        for card in hinted:
            # A color hint narrows plausible colors to exactly that color...
            self.assertEqual(card["plausible_colors"], [color])
            # ...and says nothing about rank.
            self.assertIsNone(card["hinted_rank"])
        # Cards the hint skipped are now known NOT to be that color.
        for card in after["hands"][1]["cards"]:
            if card["hinted_color"] is None:
                self.assertNotIn(color, card["plausible_colors"])

    def test_hanabi_cooperative_has_no_winner(self):
        env = make(
            "open_spiel_hanabi",
            configuration={"includeLegalActions": True, "seed": 1},
            debug=True,
        )
        env.run(["random", "random"])
        obs = _obs(env, 0)
        self.assertTrue(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertIn(obs["outcome"], ("lives_exhausted", "deck_exhausted", "perfect_score"))
        # Both players share one outcome.
        self.assertEqual(obs["returns"][0], obs["returns"][1])
        if obs["outcome"] == "lives_exhausted":
            # Bombing out zeroes the score no matter what was played.
            self.assertEqual(obs["life_tokens"], 0)
            self.assertEqual(obs["returns"][0], 0.0)
        else:
            self.assertEqual(obs["returns"][0], obs["score"])

    def test_hanabi_legal_actions_match_env(self):
        env = make(
            "open_spiel_hanabi",
            configuration={"includeLegalActions": True, "seed": 0},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        obs = _obs(env, 0)
        self.assertEqual(
            [a["action"] for a in obs["legal_actions"]],
            env.state[0]["observation"]["legalActions"],
        )
        self.assertEqual(
            [a["label"] for a in obs["legal_actions"]],
            env.state[0]["observation"]["legalActionStrings"],
        )

    def test_proxy_loads_without_parameters(self):
        # pyspiel backfills every spec key, and hanabi's spec defaults are
        # sentinels (0 / "") that hanabi itself rejects with a C++ SPIEL_CHECK
        # -- which aborts the process rather than raising, so a caller cannot
        # guard against it. The proxy must strip them. `_register_game_envs`
        # probes every registered short name this way.
        game = pyspiel.load_game("hanabi_proxy")
        self.assertEqual(game.num_players(), 2)
        self.assertGreater(game.new_initial_state().chance_outcomes(), [])

    def test_proxy_honors_explicit_parameters(self):
        game = pyspiel.load_game(
            "hanabi_proxy",
            {
                "players": 3,
                "colors": 4,
                "ranks": 5,
                "hand_size": 5,
                "max_life_tokens": 3,
                "max_information_tokens": 8,
            },
        )
        self.assertEqual(game.num_players(), 3)
        self.assertEqual(game.get_parameters()["colors"], 4)

    def test_proxy_legal_actions_are_only_the_actors(self):
        # The legal hints against a hand enumerate exactly the colors and
        # ranks IN that hand, so handing the actor's move list to a non-acting
        # observer would spell out the hand that observer may not see.
        game = pyspiel.load_game("hanabi_proxy")
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        actor = int(state.current_player())
        other = (actor + 1) % 2
        self.assertNotEmpty(state.state_dict(actor)["legal_actions"])
        self.assertEmpty(state.state_dict(other)["legal_actions"])

    def test_proxy_score_is_zero_once_the_lives_are_gone(self):
        # HanabiState::Score() returns 0 outright at zero lives and Returns()
        # is that score, so a proxy that just summed the fireworks would
        # report points for a game that banked none.
        game = pyspiel.load_game("hanabi_proxy")
        state = game.new_initial_state()
        while not state.is_terminal():
            while state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            if state.is_terminal():
                break
            # Play blind to burn lives; a rank-1 card may still land first.
            state.apply_action(state.legal_actions()[0])
        observation = state.state_dict(0)
        if observation["outcome"] != "lives_exhausted":
            self.skipTest("this deal did not bomb out")
        self.assertEqual(observation["life_tokens"], 0)
        self.assertEqual(observation["score"], 0)
        self.assertEqual(observation["score"], observation["returns"][0])
        # The stacks that were built are still reported, for display.
        self.assertEqual(observation["fireworks_total"], sum(observation["fireworks"].values()))

    def test_proxy_deck_total_matches_the_engine_at_every_size(self):
        # Every card is in the deck, a hand, a firework, or the discards, so
        # the total is recoverable mid-game -- not just from the root. At
        # ranks=1 it is three copies per color, because NumberCardInstances
        # tests the bottom rank before the top one.
        for colors, ranks in ((5, 5), (2, 1), (3, 1), (2, 3)):
            game = pyspiel.load_game(
                "hanabi_proxy", {"players": 2, "colors": colors, "ranks": ranks, "hand_size": 2}
            )
            state = game.new_initial_state()
            expected = state.state_dict(0)["deck_size"]
            self.assertEqual(state.state_dict(0)["deck_total"], expected)
            while state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            # Still the full deck once the cards have been dealt into hands.
            self.assertEqual(state.state_dict(0)["deck_total"], expected)

    def test_proxy_final_turns_remaining_counts_down_after_the_deck_empties(self):
        # Null while cards remain, then one turn per seat. A tiny deck so
        # discard-only play drains it before the lives run out.
        game = pyspiel.load_game(
            "hanabi_proxy", {"players": 2, "colors": 2, "ranks": 2, "hand_size": 2}
        )
        state = game.new_initial_state()
        seen = []
        while not state.is_terminal():
            while state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            if state.is_terminal():
                break
            observation = state.state_dict(0)
            if observation["deck_size"] > 0:
                self.assertIsNone(observation["final_turns_remaining"])
            else:
                seen.append(observation["final_turns_remaining"])
            discards = [a for a in state.legal_actions() if "Discard" in state.action_to_string(a)]
            state.apply_action(discards[0] if discards else state.legal_actions()[0])
        self.assertTrue(seen, "the game never reached deck exhaustion")
        # Strictly decreasing, starting at no more than one turn per seat.
        self.assertLessEqual(seen[0], 2)
        self.assertEqual(seen, sorted(seen, reverse=True))
        self.assertIsNone(state.state_dict(0)["final_turns_remaining"])  # Terminal.

    def test_hanabi_invalid_action(self):
        env = make("open_spiel_hanabi", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"][0],
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )


if __name__ == "__main__":
    absltest.main()
