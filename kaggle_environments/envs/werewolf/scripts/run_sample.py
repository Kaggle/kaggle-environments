import argparse
import logging
import math
import multiprocessing
import os
import random
import shutil

import tenacity
import yaml
from tqdm import tqdm

from kaggle_environments.envs.werewolf.runner import (
    append_timestamp_to_dir,
    log_git_hash,
    log_launch_command,
    setup_logger,
)
from kaggle_environments.envs.werewolf.scripts.utils import run_single_game_cli

logger = logging.getLogger(__name__)


def load_agent_pool(config_paths: list[str]) -> list[dict]:
    """Loads agent configurations from multiple YAML files and combines them."""
    agent_pool = []
    for path in config_paths:
        with open(path, "r") as f:
            agents = yaml.safe_load(f)
            if isinstance(agents, list):
                agent_pool.extend(agents)
    return agent_pool


run_single_game_with_retry = tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=100),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
)(run_single_game_cli)


def game_runner_wrapper(args):
    """Wrapper to unpack arguments for the multiprocessing pool."""
    game_dir, game_config, use_random_agents, debug = args
    # This function will be responsible for running a single game.
    # We can use a simplified version of the logic in run_block.py's wrapper.
    # For now, we'll just print the intention.
    # In the next step, we'll implement the actual game running logic.
    # print(f"Running game in: {game_dir}")
    try:
        run_single_game_with_retry(game_dir, game_config, use_random_agents, debug)
    except Exception as e:
        logger.error(f"Game in {game_dir} failed after retries with error: {e}")
        # Optionally, log the full traceback
        logger.debug("Traceback:", exc_info=True)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Werewolf games with sampled agents.")
    parser.add_argument(
        "-p",
        "--agent_pool_configs",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to YAML files containing agent pools.",
    )
    parser.add_argument(
        "-c",
        "--base_game_config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file for game rules.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="experiment/sample_game", help="Output directory for logs and replays."
    )
    parser.add_argument("-n", "--num_games", type=int, default=1, help="Number of games to run.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-a", "--append_timestamp_to_dir", action="store_true", help="Append a timestamp to the output directory."
    )
    parser.add_argument(
        "-s",
        "--shuffle_player_ids",
        action="store_true",
        help="Shuffle player ids for each game to account for name bias.",
    )
    parser.add_argument(
        "-r", "--use_random_agents", action="store_true", help="Use random agents for all players for fast testing."
    )
    parser.add_argument("--parallel", action="store_true", help="Run games in parallel using multiple processes.")
    parser.add_argument(
        "--without_replacement",
        action="store_true",
        help="If provided, sample agents without replacement. Defaults to sampling with replacement.",
    )
    parser.add_argument(
        "--shuffle_roles",
        action="store_true",
        help="If provided, shuffle the roles from the base config for each game.",
    )
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes for parallel execution.")
    return parser.parse_args()


def setup_environment(args):
    """Sets up the output directory and logging."""
    run_output_dir = append_timestamp_to_dir(args.output_dir, append=args.append_timestamp_to_dir)
    os.makedirs(run_output_dir, exist_ok=True)
    setup_logger(run_output_dir, "run_sample")
    log_launch_command()
    log_git_hash()

    # Save configs
    config_dump_dir = os.path.join(run_output_dir, "configs")
    base_config_path = os.path.join(config_dump_dir, "base_config.yaml")
    agents_pool_dir = os.path.join(config_dump_dir, "pool_of_agents")
    os.makedirs(agents_pool_dir, exist_ok=True)
    for config_path in args.agent_pool_configs:
        shutil.copy(config_path, agents_pool_dir)
    shutil.copy(args.base_game_config, base_config_path)
    logger.info(f"Copied experiment configs to {config_dump_dir}")
    return run_output_dir


def generate_game_tasks(args, run_output_dir, agent_pool, game_config_template):
    """Generates configurations for each game to be run."""
    original_agents_config = game_config_template.get("agents", [])
    num_agents_per_game = len(original_agents_config)

    if num_agents_per_game == 0:
        logger.error("The base game config must specify a list of agents with roles.")
        return []

    if args.without_replacement:
        if num_agents_per_game > len(agent_pool):
            logger.error(
                f"Cannot sample {num_agents_per_game} agents without replacement "
                f"from a pool of only {len(agent_pool)} agents."
            )
            return []
        logger.info("Sampling agents without replacement.")
    else:
        logger.info("Sampling agents with replacement.")

    role_configs = [
        {"role": agent["role"], "role_params": agent.get("role_params")} for agent in original_agents_config
    ]

    game_tasks = []
    for i in range(args.num_games):
        game_dir = os.path.join(run_output_dir, f"game_{i}")
        os.makedirs(game_dir, exist_ok=True)

        if args.shuffle_roles:
            random.shuffle(role_configs)

        if args.without_replacement:
            sampled_agent_specs = random.sample(agent_pool, k=num_agents_per_game)
        else:
            sampled_agent_specs = random.choices(agent_pool, k=num_agents_per_game)

        new_agents_config = []
        for j, original_agent in enumerate(original_agents_config):
            sampled_spec = sampled_agent_specs[j]
            role_config = role_configs[j]
            role_params = role_config.get("role_params") or {}
            new_agent = {
                **sampled_spec,
                "role": role_config["role"],
                "id": original_agent["id"],
                "role_params": role_params,
            }
            new_agents_config.append(new_agent)

        if args.shuffle_player_ids:
            player_ids = [agent["id"] for agent in new_agents_config]
            random.shuffle(player_ids)
            for agent, player_id in zip(new_agents_config, player_ids):
                agent["id"] = player_id

        final_game_config = {**game_config_template, "agents": new_agents_config}
        game_tasks.append((game_dir, final_game_config, args.use_random_agents, args.debug))

    return game_tasks


def run_games(args, game_tasks):
    """Executes the generated game tasks, either sequentially or in parallel."""
    logger.info(f"Generated {len(game_tasks)} game tasks.")
    if args.parallel:
        num_processes = args.num_processes or max(1, math.floor(multiprocessing.cpu_count() * 0.8))
        logger.info(f"Running games in parallel with up to {num_processes} processes.")
        with tqdm(total=len(game_tasks), desc="Processing Games") as pbar:
            with multiprocessing.Pool(processes=num_processes) as pool:
                for _ in pool.imap_unordered(game_runner_wrapper, game_tasks):
                    pbar.update(1)
    else:
        logger.info("Running games sequentially.")
        for task in tqdm(game_tasks, desc="Processing Games"):
            game_runner_wrapper(task)


def main():
    """Main function to orchestrate the sampling and running of games."""
    args = parse_arguments()
    run_output_dir = setup_environment(args)

    agent_pool = load_agent_pool(args.agent_pool_configs)
    if not agent_pool:
        logger.error("Agent pool is empty. Please check your agent pool config files.")
        return

    with open(args.base_game_config, "r") as f:
        base_config = yaml.safe_load(f)
        game_config_template = base_config.get("game_config", {})

    game_tasks = generate_game_tasks(args, run_output_dir, agent_pool, game_config_template)

    if game_tasks:
        run_games(args, game_tasks)
        logger.info(f"All games finished. Results saved in: {run_output_dir}")


if __name__ == "__main__":
    main()
