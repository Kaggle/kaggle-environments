import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime

from kaggle_environments import PROJECT_ROOT, make

logger = logging.getLogger(__name__)


def log_launch_command():
    """Logs the command used to launch the script."""
    command = " ".join(sys.argv)
    logger.info(f"Launch command: {command}")


class LogExecutionTime:
    """
    A context manager to log the execution time of a code block.
    The elapsed time is stored in the `elapsed_time` attribute.

    Example:
        logger = logging.getLogger(__name__)
        with LogExecutionTime(logger, "My Task") as timer:
            # Code to be timed
            time.sleep(1)
        print(f"Task took {timer.elapsed_time:.2f} seconds.")
        print(f"Formatted time: {timer.elapsed_time_formatted()}")
    """

    def __init__(self, logger_obj: logging.Logger, task_str: str):
        """
        Initializes the context manager.

        Args:
            logger_obj: The logger instance to use for output.
            task_str: A descriptive string for the task being timed.
        """
        self.logger = logger_obj
        self.task_str = task_str
        self.start_time = None
        self.elapsed_time = 0.0

    def __enter__(self):
        """Records the start time when entering the context."""
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.task_str}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Calculates and logs the elapsed time upon exiting the context."""
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        self.logger.info(f"Finished: {self.task_str} in {self.elapsed_time_formatted()}.")

    def elapsed_time_formatted(self) -> str:
        """Returns the elapsed time as a formatted string (HH:MM:SS)."""
        return time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))


def append_timestamp_to_dir(dir_path, append=True):
    if not append:
        return dir_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dir_path + f"_{timestamp}"
    return out


def shuffle_roles_inplace(config):
    agents = config["agents"]
    roles = [agent["role"] for agent in agents]
    random.shuffle(roles)
    for new_role, agent in zip(roles, agents):
        agent["role"] = new_role


def run_werewolf(output_dir, base_name, config, agents, debug):
    """
    Runs a game of Werewolf, saves the replay, and logs the execution time.

    Args:
        output_dir (str): The directory where the output files will be saved.
        base_name (str): The base name for the output files (HTML, JSON).
        config (dict): The configuration for the Werewolf environment.
        agents (list): A list of agents to participate in the game.
        debug (bool): A flag to enable or disable debug mode.
    """
    start_time = time.time()
    logger.info(f"Results saved to {output_dir}.")
    os.makedirs(output_dir, exist_ok=True)
    html_file = os.path.join(output_dir, f"{base_name}.html")
    json_file = os.path.join(output_dir, f"{base_name}.json")

    with LogExecutionTime(logger_obj=logger, task_str="env run") as timer:
        env = make("werewolf", debug=debug, configuration=config)
        env.run(agents)

    env.info["total_run_time"] = timer.elapsed_time
    env.info["total_run_time_formatted"] = timer.elapsed_time_formatted()

    logger.info("Game finished")
    env_out = env.render(mode="html")
    with open(html_file, "w") as out:
        out.write(env_out)
    logger.info(f"HTML replay written to {html_file}")
    env_out = env.render(mode="json")
    with open(json_file, "w") as out:
        out.write(env_out)
    logger.info(f"JSON replay written to {json_file}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info(f"Script finished in {formatted_time}.")
    return env


def setup_logger(output_dir, base_name):
    """
    Sets up a logger to output to both the console and a log file.

    Args:
        output_dir (str): The directory where the log file will be saved.
        base_name (str): The base name for the log file.
    """
    log_file = os.path.join(output_dir, f"{base_name}.log")
    os.makedirs(output_dir, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode="w")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def log_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit code
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
            logger.info(f"Running from git commit: {git_hash}")
        else:
            logger.info("Not a git repository or git command failed.")
    except FileNotFoundError:
        logger.info("Git command not found.")
