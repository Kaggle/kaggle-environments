"""Shimmy Connect Four environment for kaggle-environments.

This is a thin wrapper around shimmy_base that provides:
- Game-specific configuration (game_name, env_name, title, description)
- Custom renderer for Connect Four board visualization
"""

from kaggle_environments.envs.game_drivers.shimmy_base import create_shimmy_environment

# Game-specific configuration
GAME_NAME = "connect_four"  # OpenSpiel game name
ENV_NAME = "shimmy_connect_four"  # kaggle-environments name
TITLE = "Shimmy Connect Four"
DESCRIPTION = "Connect Four via OpenSpiel through Shimmy remote game driver"


def _connect_four_renderer(state, env):
    """Render the Connect Four board state."""
    try:
        obs = state[0].observation
        board = getattr(obs, "board", None) or getattr(obs, "observation", None)
        # Connect Four is 6 rows x 7 columns = 42 cells
        if not board or len(board) != 42:
            return "Game not started"
    except (AttributeError, IndexError):
        return "Game not started"

    def format_cell(c):
        if c == 0:
            return "."
        elif c == 1:
            return "X"
        elif c == 2:
            return "O"
        return str(c)

    lines = []
    lines.append(" 0 1 2 3 4 5 6")
    lines.append("---------------")
    # Board is stored row by row, 6 rows x 7 columns
    for row in range(6):
        cells = [format_cell(board[row * 7 + col]) for col in range(7)]
        lines.append(f"|{'|'.join(cells)}|")
    lines.append("---------------")

    return "\n".join(lines)


def _connect_four_html_renderer():
    """HTML renderer for Connect Four."""
    return "<div>Shimmy Connect Four</div>"


# Create environment using base module
_env = create_shimmy_environment(
    game_name=GAME_NAME,
    env_name=ENV_NAME,
    title=TITLE,
    description=DESCRIPTION,
    renderer=_connect_four_renderer,
    html_renderer=_connect_four_html_renderer,
)

# Export required symbols for kaggle-environments
agents = _env["agents"]
interpreter = _env["interpreter"]
renderer = _env["renderer"]
html_renderer = _env["html_renderer"]
specification = _env["specification"]
