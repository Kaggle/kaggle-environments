"""
OpenSpiel games via Shimmy wrapper for kaggle_environments.

This environment provides access to OpenSpiel games through the Shimmy wrapper,
which converts them to the PettingZoo AEC API. The actual game logic is handled
by the shimmy_remote_driver in remote_game_drivers.

Supported games include:
- Board games: tic_tac_toe, connect_four, chess, go, checkers, etc.
- Card games: kuhn_poker, leduc_poker, gin_rummy, etc.
- Abstract games: nim, pig, matrix_rps, etc.

Usage with ProtoAgent:
    from kaggle_environments import make

    env = make('openspiel_shimmy', configuration={'game_name': 'tic_tac_toe'})
    agents = [
        {'type': 'proto', 'url': 'http://localhost:8080/agent'},
        {'type': 'proto', 'url': 'http://localhost:8081/agent'},
    ]
    env.run(agents)
"""

# List of supported OpenSpiel games
SUPPORTED_GAMES = frozenset(
    [
        # Board games
        "amazons",
        "backgammon",
        "breakthrough",
        "checkers",
        "chess",
        "clobber",
        "connect_four",
        "dots_and_boxes",
        "go",
        "havannah",
        "hex",
        "hive",
        "lines_of_action",
        "mancala",
        "nine_mens_morris",
        "othello",
        "oware",
        "pentago",
        "quoridor",
        "tic_tac_toe",
        "twixt",
        "ultimate_tic_tac_toe",
        "y",
        # Card/dice games
        "gin_rummy",
        "kuhn_poker",
        "leduc_poker",
        "liars_dice",
        # Abstract/mathematical games
        "blotto",
        "coin_game",
        "goofspiel",
        "markov_soccer",
        "matrix_rps",
        "nim",
        "oshi_zumo",
        "pig",
        # Imperfect information games
        "battleship",
        "dark_chess",
        "dark_hex",
        "phantom_ttt",
        # Variant games
        "einstein_wurfelt_nicht",
        "latent_ttt",
    ]
)


def interpreter(state, environment):
    """
    Interpreter for OpenSpiel/Shimmy games.

    The actual game logic is handled by the shimmy_remote_driver.
    This interpreter is a pass-through that maintains compatibility
    with kaggle_environments while the game state is managed externally.
    """
    return state


def renderer(state, environment):
    """
    Text renderer for OpenSpiel/Shimmy games.
    """
    game_name = environment.configuration.get("game_name", "unknown")
    return f"OpenSpiel Game: {game_name}\nUse ProtoAgent with shimmy_remote_driver for full functionality."


def html_renderer(environment):
    """
    HTML renderer for OpenSpiel/Shimmy games.
    """
    game_name = environment.configuration.get("game_name", "unknown")
    return f"""
    <div style="font-family: sans-serif; padding: 20px;">
        <h3>OpenSpiel via Shimmy</h3>
        <p><strong>Game:</strong> {game_name}</p>
        <p>This environment uses proto-based networking via kaggle_evaluation relay.</p>
        <p>Rendering is handled by the game driver.</p>
    </div>
    """


def random_agent(observation, configuration):
    """
    A simple random agent for testing.

    Note: For actual gameplay, use ProtoAgent with a shimmy_remote_driver
    inference server that can properly handle the action space and masks.
    """
    return 0  # Default action - real agents should use the action_mask from info
