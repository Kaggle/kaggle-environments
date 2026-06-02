"""
Engine-direct regression tests for the vendored Kaggle engine.

These tests exercise reinforcetactics.kaggle.reinforce_tactics_engine without
going through the kaggle-environments interpreter. They cover bugs that are
not reachable from the interpreter's hot path (legal-action enumeration,
direct save/load round-trips, fog-of-war memory clearing) so we don't
re-introduce them when re-syncing the vendored engine in the future.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name
import numpy as np
import pandas as pd
import pytest

from reinforcetactics.kaggle.reinforce_tactics_engine import GameState
from reinforcetactics.kaggle.reinforce_tactics_engine.core.unit import Unit


def _small_map(width=10, height=10):
    """Build a 10x10 grass map with HQ + building for each player."""
    m = np.array([["p"] * width for _ in range(height)], dtype=object)
    m[0, 0] = "h_1"
    m[0, 1] = "b_1"
    m[height - 1, width - 1] = "h_2"
    m[height - 1, width - 2] = "b_2"
    return pd.DataFrame(m)


def _new_game(**kwargs):
    """Build a 2-player GameState with enough gold for several unit creations."""
    game = GameState(_small_map(), num_players=2, **kwargs)
    game.player_gold = {1: 5000, 2: 5000}
    return game


# ---------------------------------------------------------------------------
# Finding #4: 2-player assertion
# ---------------------------------------------------------------------------


class TestPlayerCountAssertion:
    def test_accepts_two_players(self):
        GameState(_small_map(), num_players=2)  # should not raise

    def test_rejects_one_player(self):
        with pytest.raises(ValueError, match="2-player only"):
            GameState(_small_map(), num_players=1)

    def test_rejects_three_players(self):
        with pytest.raises(ValueError, match="2-player only"):
            GameState(_small_map(), num_players=3)


# ---------------------------------------------------------------------------
# Finding #1: get_legal_actions does not AttributeError
# ---------------------------------------------------------------------------


class TestLegalActions:
    def test_get_legal_actions_returns_dict(self):
        """Used to AttributeError on _cache_valid."""
        game = GameState(_small_map(), num_players=2)
        actions = game.get_legal_actions()
        assert isinstance(actions, dict)
        assert "create_unit" in actions
        assert "move" in actions
        assert "attack" in actions
        assert "seize" in actions

    def test_create_unit_listed_for_buildings(self):
        """A fresh game should let player 1 create units at their building."""
        game = GameState(_small_map(), num_players=2)
        actions = game.get_legal_actions(player=1)
        # At least one create_unit option should be available
        assert len(actions["create_unit"]) > 0
        # Building location is at (1, 0)
        building_creates = [a for a in actions["create_unit"] if (a["x"], a["y"]) == (1, 0)]
        assert len(building_creates) > 0

    # --- Finding #2: move-filter correctness ---------------------------------

    def test_move_destinations_exclude_own_unit(self):
        """A unit's reachable set must not include tiles occupied by allies."""
        game = _new_game()
        ally_a = game.create_unit("W", 4, 4, player=1)
        game.create_unit("W", 5, 4, player=1)  # ally blocks (5, 4)
        ally_a.can_move = True

        actions = game.get_legal_actions(player=1)
        moves_for_a = [m for m in actions["move"] if m["unit"] is ally_a]
        destinations = {(m["to_x"], m["to_y"]) for m in moves_for_a}
        assert (5, 4) not in destinations, "Own-unit-occupied tile leaked into legal moves"


# ---------------------------------------------------------------------------
# Finding #5: end_turn() honours max_turns
# ---------------------------------------------------------------------------


class TestMaxTurnsDraw:
    def test_end_turn_triggers_game_over_at_max_turns(self):
        game = GameState(_small_map(), num_players=2, max_turns=3)
        # Two end_turns per full round; loop generously.
        for _ in range(20):
            if game.game_over:
                break
            game.end_turn()
        assert game.game_over is True
        assert game.winner is None  # draw, no winner
        assert game.turn_number >= game.max_turns


# ---------------------------------------------------------------------------
# Findings #6, #7, #8: save/load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def _ready_game(self):
        game = _new_game(max_turns=42)
        game.create_unit("W", 3, 3, player=1)
        game.create_unit("M", 6, 6, player=2)
        # Move once so the unit's original_x / original_y diverges from x / y.
        warrior = next(u for u in game.units if u.type == "W")
        warrior.x, warrior.y = 4, 3
        warrior.has_moved = True
        warrior.distance_moved = 1
        # Record an action.
        game.record_action("custom", note="test")
        return game

    def test_unit_from_dict_preserves_original_position(self):
        """Finding #6 — original_x / original_y survive to_dict / from_dict."""
        u = Unit("W", 4, 3, 1)
        u.original_x = 2
        u.original_y = 5
        restored = Unit.from_dict(u.to_dict())
        assert restored.original_x == 2
        assert restored.original_y == 5

    def test_to_dict_serialises_action_history(self):
        """Finding #7 — action_history must round-trip."""
        game = self._ready_game()
        payload = game.to_dict()
        assert "action_history" in payload
        # We recorded at least the warrior creation, mage creation, and 'custom' action.
        assert len(payload["action_history"]) >= 1
        assert any(a.get("type") == "custom" for a in payload["action_history"])

    def test_from_dict_restores_max_turns_and_history(self):
        """Findings #7 + #8 — from_dict round-trips max_turns and action_history."""
        game = self._ready_game()
        payload = game.to_dict()
        # Reuse the original map_data for restoration.
        restored = GameState.from_dict(payload, _small_map())
        assert restored.max_turns == 42
        assert len(restored.action_history) == len(game.action_history)
        # Unit positions and original positions both preserved.
        warrior = next(u for u in restored.units if u.type == "W")
        assert (warrior.x, warrior.y) == (4, 3)
        # original position diverged before save -> must survive
        assert (warrior.original_x, warrior.original_y) != (warrior.x, warrior.y)


# ---------------------------------------------------------------------------
# Finding #9: visibility memory clear is invoked on update
# ---------------------------------------------------------------------------


class TestVisibilityMemoryClear:
    def test_clear_stale_unit_memory_called(self, monkeypatch):
        game = GameState(_small_map(), num_players=2, fog_of_war=True)
        calls = []
        original = game.visibility_maps[1].clear_stale_unit_memory

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(game.visibility_maps[1], "clear_stale_unit_memory", spy)
        game.update_visibility(player=1)
        assert calls, "update_visibility() should invoke clear_stale_unit_memory()"
        assert calls[0][1].get("max_turns") == 10


# ---------------------------------------------------------------------------
# Finding #10: TileGrid.to_numpy is safe when max_health == 0
# ---------------------------------------------------------------------------


class TestToNumpyHealthGuard:
    def test_zero_max_health_does_not_divide(self):
        game = GameState(_small_map(), num_players=2)
        # Force a structure tile to report max_health = 0.
        target = game.grid.tiles[0][0]  # HQ tile
        target.max_health = 0
        target.health = 0
        # Must not raise ZeroDivisionError.
        arr = game.grid.to_numpy()
        assert arr[0, 0, 2] == 0  # health channel zeroed when max_health is invalid


# ---------------------------------------------------------------------------
# Finding #3: resign() removes the resigning player's units
# ---------------------------------------------------------------------------


class TestResign:
    def test_resign_strips_units_and_sets_winner(self):
        game = _new_game()
        game.create_unit("W", 3, 3, player=1)
        game.create_unit("M", 6, 6, player=2)
        assert len(game.units) == 2

        game.resign(player=1)
        assert game.game_over is True
        assert game.winner == 2
        assert all(u.player != 1 for u in game.units)
        assert len(game.units) == 1
