"""Run the settings in a given config with all agents llm agents by substituting all with a single model.
This is useful for example to evaluate the game rule balance.
"""

import argparse
import copy
import logging
import multiprocessing
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import tenacity
import yaml
from tqdm import tqdm

from kaggle_environments.envs.werewolf.runner import LogExecutionTime, append_timestamp_to_dir, setup_logger
from kaggle_environments.envs.werewolf.scripts.utils import run_single_game_cli

logger = logging.getLogger(__name__)


run_single_game_with_retry = tenacity.retry(
    wait=tenacity.wait_random_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
)(run_single_game_cli)


def game_runner_wrapper(args):
    """Wrapper to unpack arguments for the multiprocessing pool."""
    game_dir, game_config, use_random_agents, debug = args
    run_single_game_with_retry(game_dir, game_config, use_random_agents, debug)


def shuffle_field(agents, field_name):
    values = [agent[field_name] for agent in agents]
    random.shuffle(values)
    for agent, value in zip(agents, values):
        agent[field_name] = value


def run_self_play_games(
    model_name,
    thumbnail,
    output_dir,
    num_games,
    config,
    use_random_agents,
    debug,
    parallel,
    num_processes,
    shuffle_roles,
):
    """
    Generates and runs game tasks for the self-play experiment.
    """
    if debug:
        logger.warning("Debug mode is enabled. Forcing sequential execution.")

    game_tasks = []
    base_game_config = config["game_config"]

    # modify the config to use a single model
    agents = base_game_config["agents"]
    for agent in agents:
        agent["thumbnail"] = thumbnail
        agent["agent_id"] = f"llm/{model_name}"
        agent["display_name"] = os.path.basename(model_name)
        agent["llms"][0]["model_name"] = model_name

    for i in range(num_games):
        game_output_dir = os.path.join(output_dir, f"game_{i}")
        os.makedirs(game_output_dir, exist_ok=True)

        game_config = copy.deepcopy(base_game_config)

        if shuffle_roles:
            logger.info(f"Shuffling roles for game {i}")
            role_configs = [
                {"role": agent["role"], "role_params": agent.get("role_params", {})} for agent in game_config["agents"]
            ]
            random.shuffle(role_configs)
            for agent, role_config in zip(game_config["agents"], role_configs):
                agent["role"] = role_config["role"]
                agent["role_params"] = role_config["role_params"]

        # shuffle player ids
        logger.info(f"Shuffling player ids for game {i}")
        shuffle_field(game_config["agents"], "id")

        task = (game_output_dir, game_config, use_random_agents, debug)
        game_tasks.append(task)

    with tqdm(total=num_games, desc="Running Self-Play Games") as pbar:
        if parallel:
            with ThreadPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(game_runner_wrapper, task) for task in game_tasks]
                for future in as_completed(futures):
                    # You could also add error handling here by checking future.exception()
                    pbar.update(1)
        else:
            for task in game_tasks:
                game_runner_wrapper(task)
                pbar.update(1)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "configs", "run", "roundrobin_discussion_small.yaml")

    parser = argparse.ArgumentParser(description="Run N self-play Werewolf games based on a configuration file.")
    parser.add_argument(
        "-c", "--config_path", type=str, default=default_config_path, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="werewolf_self_play",
        help="Output directory for the log and replay files.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="gemini/gemini-2.5-flash",
        help="The model name by litellm for self play.",
    )
    parser.add_argument(
        "-t",
        "--thumbnail",
        type=str,
        default="https://storage.googleapis.com/kaggle-static/game-arena/werewolf/thumbnails/gemini.png",
        help="The thumbnail image url.",
    )
    parser.add_argument("-n", "--num_games", type=int, default=1, help="Number of self-play games to run.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-r", "--random_agents", action="store_true", help="Use random agents for all players for fast testing."
    )
    parser.add_argument(
        "-a", "--append_timestamp_to_dir", action="store_true", help="Append a timestamp to the output directory."
    )
    parser.add_argument(
        "-s", "--shuffle_roles", action="store_true", help="If provided, shuffle the roles for each game."
    )
    parser.add_argument("-p", "--parallel", action="store_true", help="Run games in parallel using multiple processes.")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes for parallel execution.")

    args = parser.parse_args()

    run_output_dir = append_timestamp_to_dir(args.output_dir, append=args.append_timestamp_to_dir)
    os.makedirs(run_output_dir, exist_ok=True)
    setup_logger(output_dir=run_output_dir, base_name="self_play")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    num_processes = args.num_processes
    if args.parallel and num_processes is None:
        # Default to 4x the number of CPUs for I/O bound tasks
        num_processes = multiprocessing.cpu_count() * 4

    logger.info("Starting self-play with the following settings:")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Thumbnail: {args.thumbnail}")
    logger.info(f"Output Directory: {run_output_dir}")
    logger.info(f"Number of Games: {args.num_games}")
    logger.info(f"Config Path: {args.config_path}")
    logger.info(f"Parallel Execution: {args.parallel}")
    if args.parallel:
        logger.info(f"Number of Processes: {num_processes}")
    logger.info(f"Debug Mode: {args.debug}")
    logger.info(f"Use Random Agents: {args.random_agents}")
    logger.info(f"Shuffle Roles: {args.shuffle_roles}")

    with LogExecutionTime(logger_obj=logger, task_str=f"{args.num_games} self-play games"):
        run_self_play_games(
            model_name=args.model_name,
            thumbnail=args.thumbnail,
            output_dir=run_output_dir,
            num_games=args.num_games,
            config=config,
            use_random_agents=args.random_agents,
            debug=args.debug,
            parallel=args.parallel,
            num_processes=num_processes,
            shuffle_roles=args.shuffle_roles,
        )

    logger.info("Self-play run finished successfully.")


if __name__ == "__main__":
    main()
