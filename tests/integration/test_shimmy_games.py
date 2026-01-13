"""
Integration tests for OpenSpiel games via Shimmy wrapper.

These tests verify that shimmy games can be created and run through
kaggle_environments. They use the 'openspiel_shimmy' environment with
different game_name configurations.

Note: These tests are expected to fail initially until the full shimmy
integration with ProtoAgent and remote game drivers is complete.
"""

import pytest

from kaggle_environments import make


class TestShimmyGameCreation:
    """Tests for creating shimmy game environments."""

    @pytest.mark.parametrize(
        "game_name",
        [
            "tic_tac_toe",
            "connect_four",
            "chess",
            "checkers",
            "hex",
            "kuhn_poker",
            "leduc_poker",
            "nim",
            "pig",
            "breakthrough",
        ],
    )
    def test_create_shimmy_environment(self, game_name: str):
        """Test that shimmy environments can be created with different games."""
        env = make("openspiel_shimmy", configuration={"game_name": game_name})
        assert env is not None
        assert env.name == "openspiel_shimmy"
        assert env.configuration.game_name == game_name


class TestShimmyBoardGames:
    """Integration tests for shimmy board games."""

    def test_tic_tac_toe_episode(self):
        """Run a tic-tac-toe episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "tic_tac_toe"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

        # Check for zero-sum game (winner gets positive, loser gets negative, or draw)
        if all(r is not None for r in rewards):
            sorted_rewards = sorted(rewards)
            loser, winner = sorted_rewards
            assert (loser == 0 and winner == 0) or (loser <= 0 <= winner), "Should be zero-sum or draw"

    def test_connect_four_episode(self):
        """Run a connect four episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "connect_four"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

    def test_chess_episode(self):
        """Run a chess episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "chess"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

    def test_checkers_episode(self):
        """Run a checkers episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "checkers"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

    def test_hex_episode(self):
        """Run a hex episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "hex"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

    def test_breakthrough_episode(self):
        """Run a breakthrough episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "breakthrough"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"


class TestShimmyCardGames:
    """Integration tests for shimmy card games."""

    def test_kuhn_poker_episode(self):
        """Run a Kuhn poker episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "kuhn_poker"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

    def test_leduc_poker_episode(self):
        """Run a Leduc poker episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "leduc_poker"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"


class TestShimmyAbstractGames:
    """Integration tests for shimmy abstract/mathematical games."""

    def test_nim_episode(self):
        """Run a nim episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "nim"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"

    def test_pig_episode(self):
        """Run a pig episode."""
        env = make("openspiel_shimmy", configuration={"game_name": "pig"})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) or r is None for r in rewards), "Rewards should be numeric or None"


class TestShimmyGameRewards:
    """Tests for shimmy game reward structures."""

    @pytest.mark.parametrize(
        "game_name",
        [
            "tic_tac_toe",
            "connect_four",
            "checkers",
            "hex",
            "breakthrough",
        ],
    )
    def test_zero_sum_board_games(self, game_name: str):
        """Test that zero-sum board games produce appropriate rewards."""
        env = make("openspiel_shimmy", configuration={"game_name": game_name})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]

        # Check that rewards are numeric (or None if not yet implemented)
        assert all(isinstance(r, (int, float)) or r is None for r in rewards)

        # If rewards are implemented, check zero-sum property
        if all(r is not None for r in rewards):
            sorted_rewards = sorted(rewards)
            loser, winner = sorted_rewards
            # Either draw (both 0) or loser <= 0 <= winner
            assert (loser == 0 and winner == 0) or (loser <= 0 <= winner), f"Game {game_name} should be zero-sum"

    @pytest.mark.parametrize(
        "game_name",
        [
            "kuhn_poker",
            "leduc_poker",
            "nim",
            "pig",
        ],
    )
    def test_game_produces_numeric_rewards(self, game_name: str):
        """Test that games produce numeric rewards."""
        env = make("openspiel_shimmy", configuration={"game_name": game_name})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]

        # All rewards should be numeric or None
        assert all(isinstance(r, (int, float)) or r is None for r in rewards)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
