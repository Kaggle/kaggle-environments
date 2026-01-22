"""Tests for all Shimmy/OpenSpiel game environments.

This file contains parametrized tests that run against all supported shimmy games.
Each game is tested with random agents to verify basic functionality.

Game lists are mirrored from hearth/tests/remote_game_drivers/test_shimmy_openspiel_drivers.py.
"""

import pytest

from kaggle_environments import make

# Wrapper games need default parameters in the game name
WRAPPER_DEFAULTS = {
    "add_noise": "add_noise(game=tic_tac_toe(),epsilon=0.1,seed=42)",
    "cached_tree": "cached_tree(game=tic_tac_toe())",
    "misere": "misere(game=tic_tac_toe())",
    "normal_form_extensive_game": "normal_form_extensive_game(game=matrix_rps())",
    "repeated_game": "repeated_game(stage_game=matrix_rps(),num_repetitions=3)",
    "repeated_poker": "repeated_poker(max_num_hands=5,reset_stacks=true,universal_poker_game_string=universal_poker())",
    "restricted_nash_response": "restricted_nash_response(game=tic_tac_toe())",
    "start_at": "start_at(game=tic_tac_toe(),history=)",
    "turn_based_simultaneous_game": "turn_based_simultaneous_game(game=matrix_rps())",
    "zerosum": "zerosum(game=matrix_rps())",
}


def _get_full_game_name(game_name: str) -> str:
    """Get full game name with wrapper defaults if needed."""
    return WRAPPER_DEFAULTS.get(game_name, game_name)


def _is_zero_sum_rewards(rewards: list[float | int]) -> bool:
    """Check that rewards are zero-sum (winner positive, loser negative or both zero)."""
    if len(rewards) != 2:
        return False
    sorted_rewards = sorted(rewards)
    loser_score, winner_score = sorted_rewards
    # Either both zero (draw) or loser negative/zero and winner positive/zero
    return (loser_score == 0 and winner_score == 0) or (loser_score <= 0 <= winner_score)


def _is_numeric_rewards(rewards: list) -> bool:
    """Check that all rewards are numeric."""
    return all(isinstance(r, (int, float)) for r in rewards)


def _get_num_players(game_name: str) -> int:
    """Get the number of players for a game using pyspiel."""
    import pyspiel  # type: ignore[import-untyped]

    full_game_name = _get_full_game_name(game_name)
    try:
        game = pyspiel.load_game(full_game_name)
        return game.num_players()
    except Exception:
        return 2  # Default to 2 players


# Dedicated shimmy game modules with custom renderers
DEDICATED_SHIMMY_GAMES = ["shimmy_tic_tac_toe", "shimmy_connect_four"]

# Zero-sum games (winner positive, loser negative or both zero)
ZERO_SUM_GAMES = [
    "breakthrough",
    "checkers",
    "chess",
    "clobber",
    "connect_four",
    "dark_chess",
    "dark_hex",
    "go",
    "havannah",
    "hex",
    "hive",
    "latent_ttt",
    "lines_of_action",
    "nim",
    "nine_mens_morris",
    "pentago",
    "phantom_ttt",
    "quoridor",
    "tic_tac_toe",
    "twixt",
    "ultimate_tic_tac_toe",
    "y",
]

# Games that yield numeric rewards (may not be zero-sum)
NUMERIC_REWARD_GAMES = [
    "2048",
    "add_noise",
    "amazons",
    "backgammon",
    "bargaining",
    "battleship",
    "blackjack",
    "blotto",
    "bridge",
    "bridge_uncontested_bidding",
    "cached_tree",
    "catch",
    "cliff_walking",
    "coin_game",
    "colored_trails",
    "coop_box_pushing",
    "coop_to_1p",
    "coordinated_mp",
    "crazy_eights",
    "cursor_go",
    "dark_hex_ir",
    "deep_sea",
    "dots_and_boxes",
    "dou_dizhu",
    "einstein_wurfelt_nicht",
    "euchre",
    "first_sealed_auction",
    "gin_rummy",
    "goofspiel",
    "hanabi",
    "hearts",
    "kriegspiel",
    "kuhn_poker",
    "laser_tag",
    "leduc_poker",
    "lewis_signaling",
    "liars_dice",
    "liars_dice_ir",
    "maedn",
    "mancala",
    "markov_soccer",
    "matching_pennies_3p",
    "matrix_bos",
    "matrix_brps",
    "matrix_cd",
    "matrix_coordination",
    "matrix_mp",
    "matrix_pd",
    "matrix_rps",
    "matrix_rpsw",
    "matrix_sh",
    "matrix_shapleys_game",
    "mfg_crowd_modelling",
    "mfg_crowd_modelling_2d",
    "mfg_dynamic_routing",
    "mfg_garnet",
    "misere",
    "mnk",
    "morpion_solitaire",
    "negotiation",
    "normal_form_extensive_game",
    "oh_hell",
    "oshi_zumo",
    "othello",
    "oware",
    "pathfinding",
    "phantom_go",
    "phantom_ttt_ir",
    "pig",
    "rbc",
    "repeated_game",
    "repeated_leduc_poker",
    "repeated_poker",
    "restricted_nash_response",
    "sheriff",
    "skat",
    "solitaire",
    "spades",
    "start_at",
    "stones_and_gems",
    "tarok",
    "tiny_bridge_2p",
    "tiny_bridge_4p",
    "tiny_hanabi",
    "trade_comm",
    "turn_based_simultaneous_game",
    "universal_poker",
    "zerosum",
]

# Games that are not supported by OpenSpiel/Shimmy or require special handling
UNSUPPORTED_GAMES = [
    "cribbage",  # No observation tensors implemented in OpenSpiel
    "efg_game",  # Requires external game definition file
    "nfg_game",  # Requires external game definition file
]

# All supported OpenSpiel games via dynamic shimmy environment
ALL_OPENSPIEL_GAMES = ZERO_SUM_GAMES + NUMERIC_REWARD_GAMES


@pytest.mark.parametrize("game_name", DEDICATED_SHIMMY_GAMES)
def test_dedicated_shimmy_game(game_name: str):
    """Test dedicated shimmy game modules with random agents."""
    env = make(game_name)
    env.run(["random", "random"])

    assert env.done, f"{game_name}: Game did not complete"
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]
    assert _is_numeric_rewards(rewards), f"{game_name}: Rewards should be numeric, got {rewards}"


@pytest.mark.parametrize("game_name", DEDICATED_SHIMMY_GAMES)
def test_dedicated_shimmy_multi_episode(game_name: str):
    """Test that dedicated shimmy games correctly reset state between episodes."""
    env = make(game_name)

    for episode in range(3):
        env.run(["random", "random"])
        assert env.done, f"{game_name} episode {episode}: Game did not complete"
        final_state = env.steps[-1]
        rewards = [agent.reward for agent in final_state]
        assert _is_numeric_rewards(rewards), f"{game_name} episode {episode}: Rewards should be numeric, got {rewards}"


@pytest.mark.parametrize("game_name", ZERO_SUM_GAMES)
def test_zero_sum_games(game_name: str):
    """Test zero-sum games yield proper rewards (winner positive, loser negative or both zero)."""
    full_game_name = _get_full_game_name(game_name)
    num_players = _get_num_players(game_name)
    agents = ["random"] * num_players

    env = make("shimmy", info={"game_name": full_game_name})
    env.run(agents)

    assert env.done, f"shimmy:{game_name}: Game did not complete"
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]
    assert _is_zero_sum_rewards(rewards), f"shimmy:{game_name}: Expected zero-sum rewards, got {rewards}"


@pytest.mark.parametrize("game_name", NUMERIC_REWARD_GAMES)
def test_numeric_reward_games(game_name: str):
    """Test games that yield numeric rewards (may not be zero-sum)."""
    full_game_name = _get_full_game_name(game_name)
    num_players = _get_num_players(game_name)
    agents = ["random"] * num_players

    env = make("shimmy", info={"game_name": full_game_name})
    env.run(agents)

    assert env.done, f"shimmy:{game_name}: Game did not complete"
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]
    assert _is_numeric_rewards(rewards), f"shimmy:{game_name}: Rewards should be numeric, got {rewards}"


@pytest.mark.parametrize("game_name", DEDICATED_SHIMMY_GAMES)
def test_shimmy_game_specification(game_name: str):
    """Test that dedicated shimmy games have valid specifications."""
    env = make(game_name)

    assert env.name == game_name
    assert env.specification is not None
    assert "name" in env.specification
    assert "title" in env.specification
    assert "description" in env.specification
    assert env.specification["name"] == game_name


@pytest.mark.parametrize("game_name", DEDICATED_SHIMMY_GAMES)
def test_shimmy_game_configuration(game_name: str):
    """Test that dedicated shimmy games have expected configuration options."""
    env = make(game_name)

    assert hasattr(env.configuration, "episodeSteps")
    assert hasattr(env.configuration, "actTimeout")
    assert hasattr(env.configuration, "runTimeout")


def test_dynamic_shimmy_missing_game_name():
    """Test that dynamic shimmy raises error when game_name is missing."""
    env = make("shimmy")

    with pytest.raises(ValueError, match="No game_name specified"):
        env.run(["random", "random"])
