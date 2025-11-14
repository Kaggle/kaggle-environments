import pytest

from kaggle_environments.envs.werewolf.game.consts import RoleConst
from kaggle_environments.envs.werewolf.game.roles import (
    create_players_from_agents_config,
    get_permutation,
    Player,
    Werewolf,
    Doctor,
    Seer,
    Villager,
)


@pytest.fixture
def sample_agents_config():
    """Provides a sample agent configuration list for testing."""
    return [
        {"id": "Player1", "agent_id": "random", "role": "Werewolf", "role_params": {}},
        {"id": "Player2", "agent_id": "random", "role": "Doctor", "role_params": {"allow_self_save": True}},
        {"id": "Player3", "agent_id": "random", "role": "Seer", "role_params": {}},
        {"id": "Player4", "agent_id": "random", "role": "Villager", "role_params": {}},
    ]


def test_create_players_from_agents_config_basic(sample_agents_config):
    """Tests basic player creation from a valid configuration."""
    players = create_players_from_agents_config(sample_agents_config)

    assert isinstance(players, list)
    assert len(players) == len(sample_agents_config)
    assert all(isinstance(p, Player) for p in players)

    # Check player IDs
    assert [p.id for p in players] == ["Player1", "Player2", "Player3", "Player4"]

    # Check role assignment and types
    assert isinstance(players[0].role, Werewolf)
    assert players[0].role.name == RoleConst.WEREWOLF

    assert isinstance(players[1].role, Doctor)
    assert players[1].role.name == RoleConst.DOCTOR
    assert players[1].role.allow_self_save is True

    assert isinstance(players[2].role, Seer)
    assert players[2].role.name == RoleConst.SEER

    assert isinstance(players[3].role, Villager)
    assert players[3].role.name == RoleConst.VILLAGER


def test_create_players_with_duplicate_ids_raises_error(sample_agents_config):
    """Tests that a ValueError is raised when duplicate agent IDs are provided."""
    invalid_config = sample_agents_config + [{"id": "Player1", "agent_id": "random", "role": "Villager", "role_params": {}}]
    with pytest.raises(ValueError, match="Duplicate agent ids found: Player1"):
        create_players_from_agents_config(invalid_config)


def test_get_permutation_is_deterministic():
    """Tests that the get_permutation function is deterministic for the same seed."""
    items = ["a", "b", "c", "d", "e"]
    seed1 = 123
    seed2 = 456

    permutation1_run1 = get_permutation(items, seed1)
    permutation1_run2 = get_permutation(items, seed1)
    permutation2 = get_permutation(items, seed2)

    assert permutation1_run1 == permutation1_run2
    assert permutation1_run1 != permutation2
    assert sorted(permutation1_run1) == sorted(items) # Ensure it's still a permutation


def test_create_players_with_role_shuffling(sample_agents_config):
    """Tests that roles are shuffled deterministically when randomize_roles is True."""
    seed = 10
    # This is the expected order based on the LCG and Fisher-Yates implementation
    expected_permuted_roles = [
        (RoleConst.WEREWOLF, {}),
        (RoleConst.SEER, {}),
        (RoleConst.DOCTOR, {"allow_self_save": True}),
        (RoleConst.VILLAGER, {}),
    ]

    players = create_players_from_agents_config(sample_agents_config, randomize_roles=True, seed=seed)

    # The roles should be assigned to the agents in the new permuted order
    for i, player in enumerate(players):
        expected_role_name, expected_role_params = expected_permuted_roles[i]
        assert player.role.name == expected_role_name
        # Check if role_params match, excluding defaults that might be added by pydantic
        for key, value in expected_role_params.items():
            assert getattr(player.role, key) == value


def test_create_players_no_shuffling_without_flag(sample_agents_config):
    """Tests that roles are not shuffled if randomize_roles is False, even with a seed."""
    seed = 42
    players = create_players_from_agents_config(sample_agents_config, randomize_roles=False, seed=seed)
    
    original_roles = [agent["role"] for agent in sample_agents_config]
    assigned_roles = [p.role.name for p in players]
    
    assert original_roles == assigned_roles
