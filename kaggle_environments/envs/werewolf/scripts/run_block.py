import argparse
import collections
import logging
import math
import multiprocessing
import os
import random
import subprocess
import sys
from itertools import permutations
from typing import List

import tenacity
import yaml
from tqdm import tqdm

from kaggle_environments.envs.werewolf.runner import setup_logger

# Initialize a placeholder logger
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_rotationally_unique_configs(roles: List[str]) -> List[List[str]]:
    """
    Generates all unique permutations of roles, filtering out any that are
    rotational duplicates. This creates a set of 'necklaces'.
    """
    all_perms = list(set(permutations(roles)))
    seen_configs = set()
    unique_necklaces = []

    for perm in all_perms:
        if perm not in seen_configs:
            unique_necklaces.append(list(perm))
            temp_deque = collections.deque(perm)
            for _ in range(len(perm)):
                temp_deque.rotate(1)
                seen_configs.add(tuple(temp_deque))
    return unique_necklaces


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO)
)
def run_single_game_with_retry(game_dir, game_config, use_random_agents, debug):
    """
    Sets up and runs a single game instance by calling run.py.
    Uses tenacity to retry on failure.
    """
    out_config = {"game_config": game_config}
    config_path = os.path.join(game_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(out_config, f, default_flow_style=False)

    run_py_path = os.path.join(os.path.dirname(__file__), 'run.py')
    cmd = [
        sys.executable,
        run_py_path,
        '--config_path', config_path,
        '--output_dir', game_dir,
    ]
    if use_random_agents:
        cmd.append('--random_agents')
    if debug:
        cmd.append('--debug')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Game in {game_dir} completed successfully.")
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(f"Stderr (non-fatal) from game in {game_dir}: {result.stderr}")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Error running game in {game_dir}.\n"
            f"Return Code: {e.returncode}\n"
            f"Stdout: {e.stdout}\n"
            f"Stderr: {e.stderr}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message) from e


def game_runner_wrapper(args):
    """Wrapper to unpack arguments for the multiprocessing pool."""
    game_dir, game_config, use_random_agents, debug, _, _ = args
    run_single_game_with_retry(game_dir, game_config, use_random_agents, debug)


def generate_game_tasks(output_dir, num_blocks, config, use_random_agents, debug):
    """
    Generates all game configurations for the entire experiment.
    """
    base_game_config = config['game_config']
    players_data = base_game_config['agents']
    base_roles = [agent['role'] for agent in players_data]

    logger.info("Generating unique role configurations (filtering rotations)...")
    all_role_configs = get_rotationally_unique_configs(base_roles)
    logger.info(f"Found {len(all_role_configs)} rotationally unique arrangements.")

    available_role_configs = []

    for block_index in range(num_blocks):
        block_dir = os.path.join(output_dir, f"block_{block_index}")
        os.makedirs(block_dir, exist_ok=True)

        if not available_role_configs:
            if num_blocks > len(all_role_configs):
                logger.warning("Sampling with replacement as num_blocks > unique configurations.")
            available_role_configs = list(all_role_configs)
            random.shuffle(available_role_configs)

        block_role_config = available_role_configs.pop()
        random.shuffle(players_data)
        current_players_deque = collections.deque(players_data)

        for game_in_block in range(len(players_data)):
            game_dir = os.path.join(block_dir, f"game_{game_in_block}")
            os.makedirs(game_dir, exist_ok=True)

            current_players = list(current_players_deque)
            game_agents_config = [
                {**player_config, 'role': block_role_config[i]}
                for i, player_config in enumerate(current_players)
            ]

            game_config = {**base_game_config, 'agents': game_agents_config}
            yield (game_dir, game_config, use_random_agents, debug, block_index, game_in_block)
            current_players_deque.rotate(1)


def run_experiment(output_dir, num_blocks, config, use_random_agents, debug, parallel, num_processes):
    """
    Runs a tournament by generating all game tasks and processing them,
    potentially in parallel.
    """
    if debug:
        logger.warning("Debug mode is enabled. Forcing sequential execution.")

    base_game_config = config['game_config']
    players_data = base_game_config['agents']
    total_games = num_blocks * len(players_data)

    if parallel:
        logger.info(f"Running games in parallel with up to {num_processes} processes.")

    game_tasks = generate_game_tasks(
        output_dir, num_blocks, config, use_random_agents, debug
    )

    with tqdm(total=total_games, desc="Processing Games") as pbar:
        if parallel:
            with multiprocessing.Pool(processes=num_processes) as pool:
                for _ in pool.imap_unordered(game_runner_wrapper, game_tasks):
                    pbar.update(1)
        else:
            for task_args in game_tasks:
                game_runner_wrapper(task_args)
                pbar.update(1)

    logger.info("All game tasks have been processed.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, 'configs', 'run', 'run_config.yaml')

    parser = argparse.ArgumentParser(
        description="Run a block-design experiment for the Werewolf game, "
                    "where each block is a complete role rotation amongst the players."
    )
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for game replays and logs.",
                        default="werewolf_block_experiment")
    parser.add_argument("-c", '--config', type=str, default=default_config_path,
                        help="Path to the base configuration YAML file.")
    parser.add_argument("-b", "--num_blocks", type=int, default=10,
                        help="Number of blocks to run. Each block is a complete role rotation.")
    parser.add_argument("-r", "--use_random_agents", action="store_true",
                        help='Use random agents for all players for fast testing.')
    parser.add_argument("-d", "--debug", action="store_true",
                        help='Enable debug mode for the game environment. ' \
                             'Note that you can use debug mode to enable intra game sequential execution.')
    parser.add_argument("-p", "--parallel", action="store_true",
                        help='Run games in parallel using multiple processes.')
    parser.add_argument("-n", "--num_processes", type=int, default=None,
                        help="Number of processes for parallel execution.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logger(args.output_dir, 'run_block')

    config = load_config(args.config)

    num_players = len(config.get('game_config', {}).get('agents', []))
    if args.num_processes is None:
        num_processes = multiprocessing.cpu_count() * 0.9
        if not args.debug:
            num_processes /= num_players
        num_processes = max(1, math.floor(num_processes))
    else:
        num_processes = args.num_processes

    logger.info("Starting experiment with the following settings:")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Number of Blocks: {args.num_blocks}")
    logger.info(f"Parallel Execution: {args.parallel}")
    if args.parallel:
        logger.info(f"Number of Processes: {num_processes}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info(f"Use Random Agents: {args.use_random_agents}")

    run_experiment(
        output_dir=args.output_dir,
        num_blocks=args.num_blocks,
        config=config,
        use_random_agents=args.use_random_agents,
        debug=args.debug,
        parallel=args.parallel,
        num_processes=num_processes
    )
    logger.info("Experiment finished successfully.")


if __name__ == '__main__':
    main()
