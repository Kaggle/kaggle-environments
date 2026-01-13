import pytest

from kaggle_environments import make
from kaggle_environments.envs.werewolf.game.consts import EnvInfoKeys, Team
from kaggle_environments.envs.werewolf.game.protocols.vote import TieBreak
from kaggle_environments.envs.werewolf.game.records import GameEndResultsDataEntry

URLS = {
    "gemini": "https://storage.googleapis.com/kaggle-static/game-arena/werewolf/thumbnails/gemini.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png",
}


@pytest.fixture
def deterministic_agents_config():
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager"]
    names = [f"player_{i}" for i in range(len(roles))]
    thumbnails = [
        URLS["gemini"],
        URLS["gemini"],
        URLS["openai"],
        URLS["openai"],
        URLS["openai"],
        URLS["claude"],
        URLS["grok"],
    ]
    agents_config = [
        {"role": role, "id": name, "agent_id": "deterministic", "thumbnail": url}
        for role, name, url in zip(roles, names, thumbnails)
    ]
    return agents_config


@pytest.fixture
def deterministic_config_options():
    options = {
        "discussion_protocol": {
            "name": "RoundRobinDiscussion",
            "params": {"max_rounds": 1, "first_to_speak": "fixed"},
        },
        "day_voting_protocol": {
            "name": "SequentialVoting",
            "params": {"first_to_vote": "random", "tie_break": TieBreak.NO_EXILE},
        },
        "werewolf_night_vote_protocol": {
            "name": "SequentialVoting",
            "params": {"first_to_vote": "random", "tie_break": TieBreak.NO_EXILE},
        },
    }
    return options


def test_game_result(deterministic_agents_config, deterministic_config_options):
    """
    Tests that the deterministic werewolves vote to eliminate the first valid target.
    """
    env = make(
        "werewolf", debug=True, configuration={"agents": deterministic_agents_config, **deterministic_config_options}
    )
    agents = ["deterministic"] * 7
    env.run(agents)

    result = GameEndResultsDataEntry(**env.info[EnvInfoKeys.GAME_END])

    assert len(env.steps) == 26
    assert result.winner_team == Team.VILLAGERS
    assert result.winner_ids == ["player_2", "player_3", "player_4", "player_5", "player_6"]
    assert result.loser_ids == ["player_0", "player_1"]
    assert result.scores == {
        "player_2": 1,
        "player_3": 1,
        "player_4": 1,
        "player_5": 1,
        "player_6": 1,
        "player_0": 0,
        "player_1": 0,
    }
    assert result.elimination_info == [
        {"player_id": "player_0", "eliminated_during_day": 1, "eliminated_during_phase": "Day"},
        {"player_id": "player_1", "eliminated_during_day": 2, "eliminated_during_phase": "Day"},
        {"player_id": "player_2", "eliminated_during_day": 0, "eliminated_during_phase": "Night"},
        {"player_id": "player_3", "eliminated_during_day": 1, "eliminated_during_phase": "Night"},
        {"player_id": "player_4", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_5", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_6", "eliminated_during_day": -1, "eliminated_during_phase": None},
    ]


def test_parallel_discussion_simultaneous_majority_vote(deterministic_agents_config, deterministic_config_options):
    config = {"agents": deterministic_agents_config, **deterministic_config_options}
    config.update(
        {
            "discussion_protocol": {"name": "ParallelDiscussion", "params": {"ticks": 2}},
            "day_voting_protocol": {"name": "SimultaneousMajority", "params": {"tie_break": TieBreak.NO_EXILE}},
            "werewolf_night_vote_protocol": {
                "name": "SimultaneousMajority",
                "params": {"tie_break": TieBreak.NO_EXILE},
            },
        }
    )

    env = make("werewolf", debug=True, configuration=config)
    agents = ["deterministic"] * 7
    env.run(agents)

    result = GameEndResultsDataEntry(**env.info[EnvInfoKeys.GAME_END])

    assert len(env.steps) == 11
    assert result.winner_team == Team.VILLAGERS
    assert result.winner_ids == ["player_2", "player_3", "player_4", "player_5", "player_6"]
    assert result.loser_ids == ["player_0", "player_1"]
    assert result.scores == {
        "player_2": 1,
        "player_3": 1,
        "player_4": 1,
        "player_5": 1,
        "player_6": 1,
        "player_0": 0,
        "player_1": 0,
    }
    assert result.elimination_info == [
        {"player_id": "player_0", "eliminated_during_day": 1, "eliminated_during_phase": "Day"},
        {"player_id": "player_1", "eliminated_during_day": 2, "eliminated_during_phase": "Day"},
        {"player_id": "player_2", "eliminated_during_day": 0, "eliminated_during_phase": "Night"},
        {"player_id": "player_3", "eliminated_during_day": 1, "eliminated_during_phase": "Night"},
        {"player_id": "player_4", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_5", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_6", "eliminated_during_day": -1, "eliminated_during_phase": None},
    ]


def test_round_by_round_bidding_discussion_sequential_vote(deterministic_agents_config, deterministic_config_options):
    config = {"agents": deterministic_agents_config, **deterministic_config_options}
    config.update(
        {
            "discussion_protocol": {
                "name": "RoundByRoundBiddingDiscussion",
                "params": {"bidding": {"name": "UrgencyBiddingProtocol"}, "max_rounds": 2, "bid_result_public": True},
            }
        }
    )
    env = make("werewolf", debug=True, configuration=config)
    agents = ["deterministic"] * 7
    env.run(agents)

    result = GameEndResultsDataEntry(**env.info[EnvInfoKeys.GAME_END])

    assert len(env.steps) == 36
    assert result.winner_team == Team.VILLAGERS
    assert result.winner_ids == ["player_2", "player_3", "player_4", "player_5", "player_6"]
    assert result.loser_ids == ["player_0", "player_1"]
    assert result.scores == {
        "player_2": 1,
        "player_3": 1,
        "player_4": 1,
        "player_5": 1,
        "player_6": 1,
        "player_0": 0,
        "player_1": 0,
    }
    assert result.elimination_info == [
        {"player_id": "player_0", "eliminated_during_day": 1, "eliminated_during_phase": "Day"},
        {"player_id": "player_1", "eliminated_during_day": 2, "eliminated_during_phase": "Day"},
        {"player_id": "player_2", "eliminated_during_day": 0, "eliminated_during_phase": "Night"},
        {"player_id": "player_3", "eliminated_during_day": 1, "eliminated_during_phase": "Night"},
        {"player_id": "player_4", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_5", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_6", "eliminated_during_day": -1, "eliminated_during_phase": None},
    ]


def test_turn_by_turn_bidding(deterministic_agents_config, deterministic_config_options):
    config = {"agents": deterministic_agents_config, **deterministic_config_options}
    config.update(
        {
            "discussion_protocol": {
                "name": "TurnByTurnBiddingDiscussion",
                "params": {"bidding": {"name": "UrgencyBiddingProtocol"}, "max_turns": 10, "bid_result_public": False},
            }
        }
    )
    env = make("werewolf", debug=True, configuration=config)
    agents = ["deterministic"] * 7
    env.run(agents)

    result = GameEndResultsDataEntry(**env.info[EnvInfoKeys.GAME_END])

    assert len(env.steps) == 36
    assert result.winner_team == Team.VILLAGERS
    assert result.winner_ids == ["player_2", "player_3", "player_4", "player_5", "player_6"]
    assert result.loser_ids == ["player_0", "player_1"]
    assert result.scores == {
        "player_2": 1,
        "player_3": 1,
        "player_4": 1,
        "player_5": 1,
        "player_6": 1,
        "player_0": 0,
        "player_1": 0,
    }
    assert result.elimination_info == [
        {"player_id": "player_0", "eliminated_during_day": 1, "eliminated_during_phase": "Day"},
        {"player_id": "player_1", "eliminated_during_day": 2, "eliminated_during_phase": "Day"},
        {"player_id": "player_2", "eliminated_during_day": 0, "eliminated_during_phase": "Night"},
        {"player_id": "player_3", "eliminated_during_day": 1, "eliminated_during_phase": "Night"},
        {"player_id": "player_4", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_5", "eliminated_during_day": -1, "eliminated_during_phase": None},
        {"player_id": "player_6", "eliminated_during_day": -1, "eliminated_during_phase": None},
    ]
