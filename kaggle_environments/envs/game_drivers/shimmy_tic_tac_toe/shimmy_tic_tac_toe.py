"""
Shimmy Tic-Tac-Toe environment for kaggle-environments.
Uses remote_game_drivers with HTTPClient for protobuf-based agent communication.
"""

import json
import os

from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

# Global environment instance (initialized on first interpreter call)
_env: OpenSpielCompatibilityV0 | None = None


def random_agent(obs: dict, config: dict) -> int:
    """Random agent that uses action masking to only select legal moves."""
    import random

    # Get action mask from the global environment
    if _env is None:
        return 0

    current_agent = _env.agent_selection
    action_mask = _env.infos.get(current_agent, {}).get("action_mask", [])

    # Find legal actions (where mask is 1)
    legal_actions = [i for i, legal in enumerate(action_mask) if legal]

    # Return random legal action
    return random.choice(legal_actions) if legal_actions else 0


agents = {"random": random_agent}


def interpreter(state, env):
    """
    Main game loop interpreter for Shimmy Tic-Tac-Toe.

    Uses Shimmy's OpenSpielCompatibilityV0 directly for turn-by-turn play.
    This matches kaggle-environments' interpreter pattern.
    """
    global _env

    # Initialize Shimmy environment on first call
    if _env is None:
        import pyspiel

        game = pyspiel.load_game("tic_tac_toe")
        _env = OpenSpielCompatibilityV0(env=game)
        _env.reset()

    # Game is done
    if env.done:
        return state

    # Get current player from PettingZoo AEC API
    current_os_agent = _env.agent_selection  # "player_0" or "player_1"

    # Map OpenSpiel agent to kaggle-environments index (0 or 1)
    current_kaggle_id = int(current_os_agent.split("_")[1])

    # Determine active/inactive agents based on current player
    active = state[current_kaggle_id]
    inactive = state[1 - current_kaggle_id]

    # Check if both agents are in valid state
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        return state

    # Get observation for current player
    obs_array = _env.observe(current_os_agent)

    # Update kaggle-environments observation
    # OpenSpiel observation is a numpy array: [0, 0, 1, 2, 0, ...]
    # 0=empty, 1=player 0's mark, 2=player 1's mark
    active.observation.board = obs_array.tolist() if hasattr(obs_array, "tolist") else list(obs_array)
    active.observation.mark = current_kaggle_id

    # Get action from agent
    action = active.action

    # Validate action
    if action is None or not isinstance(action, int) or action < 0 or action > 8:
        active.status = f"Invalid action: {action}"
        inactive.status = "DONE"
        inactive.reward = 1
        active.reward = -1
        return state

    # Step the environment with the action
    try:
        _env.step(action)
    except Exception as e:
        active.status = f"Invalid move: {e}"
        inactive.status = "DONE"
        inactive.reward = 1
        active.reward = -1
        return state

    # Check if game is done
    if all(_env.terminations.values()) or all(_env.truncations.values()):
        # Game complete - assign rewards
        for os_agent, reward in _env.rewards.items():
            idx = int(os_agent.split("_")[1])
            state[idx].reward = reward
            state[idx].status = "DONE"
        return state

    # Swap active/inactive for next turn
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    """Render the current game state."""
    board = state[0].observation.board
    if not board or len(board) != 9:
        return "Game not started"

    def format_cell(c):
        # OpenSpiel format: 0=empty, 1=player 0, 2=player 1
        if c == 0:
            return " "
        elif c == 1:
            return "X"
        elif c == 2:
            return "O"
        return str(c)

    lines = []
    lines.append("  0 | 1 | 2")
    lines.append(" -----------")
    for row in range(3):
        cells = [format_cell(board[row * 3 + col]) for col in range(3)]
        lines.append(f"  {cells[0]} | {cells[1]} | {cells[2]}")
        if row < 2:
            lines.append(" -----------")

    return "\n".join(lines)


def html_renderer():
    """HTML renderer (placeholder for now)."""
    return "<div>Shimmy Tic-Tac-Toe</div>"


# Load specification
dirpath = os.path.dirname(__file__)
jsonpath = os.path.abspath(os.path.join(dirpath, "shimmy_tic_tac_toe.json"))
with open(jsonpath) as f:
    specification = json.load(f)
