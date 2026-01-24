"""Shimmy Tic-Tac-Toe environment for kaggle-environments.

This is a thin wrapper around shimmy_base that provides:
- Game-specific configuration (game_name, env_name, title, description)
- Custom renderer for tic-tac-toe board visualization
"""

from kaggle_environments.envs.game_drivers.shimmy_base import create_shimmy_environment

# Game-specific configuration
GAME_NAME = "tic_tac_toe"  # OpenSpiel game name
ENV_NAME = "shimmy_tic_tac_toe"  # kaggle-environments name
TITLE = "Shimmy Tic-Tac-Toe"
DESCRIPTION = "Tic-Tac-Toe via OpenSpiel through Shimmy remote game driver"


def _tic_tac_toe_renderer(state, env):
    """Render the tic-tac-toe board state."""
    try:
        obs = state[0].observation
        board = getattr(obs, "board", None) or getattr(obs, "observation", None)
        if not board or len(board) != 9:
            return "Game not started"
    except (AttributeError, IndexError):
        return "Game not started"

    def format_cell(c):
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


def _tic_tac_toe_html_renderer():
    """HTML renderer for tic-tac-toe."""
    return "<div>Shimmy Tic-Tac-Toe</div>"


# Create environment using base module
_env = create_shimmy_environment(
    game_name=GAME_NAME,
    env_name=ENV_NAME,
    title=TITLE,
    description=DESCRIPTION,
    renderer=_tic_tac_toe_renderer,
    html_renderer=_tic_tac_toe_html_renderer,
)

# Export required symbols for kaggle-environments
agents = _env["agents"]
interpreter = _env["interpreter"]
renderer = _env["renderer"]
html_renderer = _env["html_renderer"]
specification = _env["specification"]
