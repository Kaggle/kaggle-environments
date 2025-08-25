import logging
import os
import time
import random
from datetime import datetime

from kaggle_environments import make


logger = logging.getLogger(__name__)


def append_timestamp_to_dir(dir_path, append=True):
    if not append: return dir_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = dir_path + f"_{timestamp}"
    return out


def shuffle_roles_inplace(config):
    agents = config['agents']
    roles = [agent['role'] for agent in agents]
    random.shuffle(roles)
    for new_role, agent in zip(roles, agents):
        agent['role'] = new_role


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

    env = make(
        'werewolf',
        debug=debug,
        configuration=config
    )
    env.run(agents)
    logger.info("Game finished")
    env_out = env.render(mode='html')
    with open(html_file, 'w') as out:
        out.write(env_out)
    logger.info(f"HTML replay written to {html_file}")
    env_out = env.render(mode='json')
    with open(json_file, 'w') as out:
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
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
