import argparse
import collections
import logging
import math
import multiprocessing
import os
import random
from itertools import permutations
from typing import Any, Dict, List

import tenacity
import yaml
from tqdm import tqdm

from kaggle_environments.envs.werewolf.runner import LogExecutionTime, append_timestamp_to_dir, setup_logger
from kaggle_environments.envs.werewolf.scripts.utils import run_single_game_cli

# Initialize a placeholder logger
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_all_unique_role_configs(role_configs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Generates all unique permutations of role configurations.
    A role configuration is a dict with 'role' and 'role_params'.
    """

    def make_hashable(config):
        role = config["role"]
        params = config.get("role_params", {})
        if params:
            return role, frozenset(params.items())
        return role, frozenset()

    def make_unhashable(hashable_config):
        role, params_frozenset = hashable_config
        return {"role": role, "role_params": dict(params_frozenset)}

    hashable_configs = [make_hashable(c) for c in role_configs]
    all_perms_hashable = list(set(permutations(hashable_configs)))
    all_perms = [[make_unhashable(c) for c in p] for p in all_perms_hashable]
    return all_perms


run_single_game_with_retry = tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
)(run_single_game_cli)


def game_runner_wrapper(args):
    """Wrapper to unpack arguments for the multiprocessing pool."""
    game_dir, game_config, use_random_agents, debug, _, _ = args
    run_single_game_with_retry(game_dir, game_config, use_random_agents, debug)


def generate_game_tasks(output_dir, num_blocks, config, use_random_agents, debug, shuffle_player_ids):
    """
    Generates all game configurations for the entire experiment.
    """
    base_game_config = config["game_config"]
    players_data = base_game_config["agents"]
    base_role_configs = [{"role": agent["role"], "role_params": agent.get("role_params", {})} for agent in players_data]

    logger.info("Generating all unique role configurations...")
    all_role_configs = get_all_unique_role_configs(base_role_configs)
    logger.info(f"Found {len(all_role_configs)} unique arrangements.")

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
                {**player_config, **block_role_config[i]} for i, player_config in enumerate(current_players)
            ]

            if shuffle_player_ids:
                player_ids = [agent["id"] for agent in game_agents_config]
                random.shuffle(player_ids)
                for i, agent in enumerate(game_agents_config):
                    agent["id"] = player_ids[i]

            game_config = {**base_game_config, "agents": game_agents_config}
            yield (game_dir, game_config, use_random_agents, debug, block_index, game_in_block)
            current_players_deque.rotate(1)


def run_experiment(
    output_dir, num_blocks, config, use_random_agents, debug, parallel, num_processes, shuffle_player_ids
):
    """
    Runs a tournament by generating all game tasks and processing them,
    potentially in parallel.
    """
    if debug:
        logger.warning("Debug mode is enabled. Forcing sequential execution.")

    base_game_config = config["game_config"]
    players_data = base_game_config["agents"]
    total_games = num_blocks * len(players_data)

    if parallel:
        logger.info(f"Running games in parallel with up to {num_processes} processes.")

    game_tasks = generate_game_tasks(output_dir, num_blocks, config, use_random_agents, debug, shuffle_player_ids)

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
    default_config_path = os.path.join(script_dir, "configs", "run", "run_config.yaml")

    parser = argparse.ArgumentParser(
        description="Run a block-design experiment for the Werewolf game, "
        "where each block is a complete role rotation amongst the players."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for game replays and logs.",
        default="werewolf_block_experiment",
    )
    parser.add_argument(
        "-c", "--config", type=str, default=default_config_path, help="Path to the base configuration YAML file."
    )
    parser.add_argument(
        "-b",
        "--num_blocks",
        type=int,
        default=10,
        help="Number of blocks to run. Each block is a complete role rotation.",
    )
    parser.add_argument(
        "-r", "--use_random_agents", action="store_true", help="Use random agents for all players for fast testing."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode for the game environment. "
        "Note that you can use debug mode to enable intra game sequential execution.",
    )
    parser.add_argument("-p", "--parallel", action="store_true", help="Run games in parallel using multiple processes.")
    parser.add_argument(
        "-n", "--num_processes", type=int, default=None, help="Number of processes for parallel execution."
    )
    parser.add_argument(
        "-a", "--append_timestamp_to_dir", action="store_true", help="Append a timestamp to the output directory."
    )
    parser.add_argument(
        "-s",
        "--shuffle_player_ids",
        action="store_true",
        help="Shuffle player ids for each game to account for name bias.",
    )

    args = parser.parse_args()

    output_dir = append_timestamp_to_dir(args.output_dir, append=args.append_timestamp_to_dir)

    os.makedirs(output_dir, exist_ok=True)

    setup_logger(output_dir, "run_block")

    config = load_config(args.config)

    num_players = len(config.get("game_config", {}).get("agents", []))
    if args.num_processes is None:
        num_processes = multiprocessing.cpu_count() * 0.9
        if not args.debug:
            num_processes /= num_players
        num_processes = max(1, math.floor(num_processes))
    else:
        num_processes = args.num_processes

    logger.info("Starting experiment with the following settings:")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Number of Blocks: {args.num_blocks}")
    logger.info(f"Parallel Execution: {args.parallel}")
    if args.parallel:
        logger.info(f"Number of Processes: {num_processes}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info(f"Use Random Agents: {args.use_random_agents}")
    logger.info(f"Shuffle Player IDs: {args.shuffle_player_ids}")

    with LogExecutionTime(logger_obj=logger, task_str="block experiment"):
        run_experiment(
            output_dir=output_dir,
            num_blocks=args.num_blocks,
            config=config,
            use_random_agents=args.use_random_agents,
            debug=args.debug,
            parallel=args.parallel,
            num_processes=num_processes,
            shuffle_player_ids=args.shuffle_player_ids,
        )
    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
