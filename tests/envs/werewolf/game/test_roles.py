import pytest

from kaggle_environments.envs.werewolf.game.consts import RoleConst
from kaggle_environments.envs.werewolf.game.roles import (
    Doctor,
    Player,
    Seer,
    Villager,
    Werewolf,
    create_players_from_agents_config,
    get_permutation,
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
    invalid_config = sample_agents_config + [
        {"id": "Player1", "agent_id": "random", "role": "Villager", "role_params": {}}
    ]
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
    assert sorted(permutation1_run1) == sorted(items)  # Ensure it's still a permutation


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


def test_shuffle_ids_and_roles_are_uncorrelated(sample_agents_config):
    """
    Tests that when both IDs and roles are shuffled, the assignments are
    uncorrelated and deterministic.
    """
    seed = 44

    # --- First run ---
    players1 = create_players_from_agents_config(
        sample_agents_config, randomize_roles=True, randomize_ids=True, seed=seed
    )

    # --- Second run with same seed ---
    players2 = create_players_from_agents_config(
        sample_agents_config, randomize_roles=True, randomize_ids=True, seed=seed
    )

    # 1. Check for determinism
    player1_map = {p.id: p.role.name for p in players1}
    player2_map = {p.id: p.role.name for p in players2}
    assert player1_map == player2_map

    # 2. Check that shuffling happened and is uncorrelated
    original_ids = [agent["id"] for agent in sample_agents_config]
    shuffled_ids = list(player1_map.keys())

    original_roles = [agent["role"] for agent in sample_agents_config]
    shuffled_roles = list(player1_map.values())

    # Assert that both lists were actually shuffled
    assert original_ids != shuffled_ids
    assert original_roles != shuffled_roles

    # Assert that the original pairings are broken
    # e.g., Player1 is no longer guaranteed to be a Werewolf
    original_map = {agent["id"]: agent["role"] for agent in sample_agents_config}
    assert player1_map != original_map

    # 3. Check a specific known outcome for the given seed to prevent regression
    # Based on seed=44 for roles and seed=44+123=167 for ids
    expected_map_seed42 = {"Player3": "Villager", "Player2": "Werewolf", "Player1": "Doctor", "Player4": "Seer"}
    assert player1_map == expected_map_seed42
