import logging
import os
import subprocess
import sys

import yaml

logger = logging.getLogger(__name__)


def run_single_game_cli(game_dir, game_config, use_random_agents, debug):
    """
    Sets up and runs a single game instance by calling run.py. Running a separate process has the distinct advantage
    of an atomic game execution unit, so the logging and dumps including html render and json are cleaner.
    """
    out_config = {"game_config": game_config}
    config_path = os.path.join(game_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(out_config, f, default_flow_style=False)

    run_py_path = os.path.join(os.path.dirname(__file__), "run.py")
    cmd = [
        sys.executable,
        run_py_path,
        "--config_path",
        config_path,
        "--output_dir",
        game_dir,
    ]
    if use_random_agents:
        cmd.append("--random_agents")
    if debug:
        cmd.append("--debug")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Game in {game_dir} completed successfully.")
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(f"Stderr (non-fatal) from game in {game_dir}: {result.stderr}")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Error running game in {game_dir}.\nReturn Code: {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message) from e
