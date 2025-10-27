import argparse
import json
import logging
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml

from kaggle_environments.envs.werewolf.runner import run_werewolf, setup_logger
from kaggle_environments.envs.werewolf.werewolf import LLM_MODEL_NAMES, CostSummary

logger = logging.getLogger(__name__)

AGENT_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Jamie", "Morgan", "Skyler"]
DEFAULT_MODEL = "gemini/gemini-2.5-flash"


def setup_game_config(max_turns: int, base_config: dict, model_name: str):
    """
    Sets up the game configuration for a single run.
    """
    config = base_config.copy()

    # Define roles and shuffle them
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
    random.shuffle(roles)
    random.shuffle(AGENT_NAMES)

    # Create agent configurations
    agents_config = []
    for i, role in enumerate(roles):
        player_name = AGENT_NAMES[i]
        agents_config.append(
            {
                "role": role,
                "id": player_name,
                "agent_id": f"llm/{model_name}",
                "display_name": f"{model_name}/{player_name}",
                "agent_harness_name": "llm_harness",
                "chat_mode": "text",
                "llms": [{"model_name": model_name}],
            }
        )

    config["agents"] = agents_config

    # Update discussion protocol with the specified max_turns
    if "discussion_protocol" in config and config["discussion_protocol"]["name"] == "TurnByTurnBiddingDiscussion":
        config["discussion_protocol"]["params"]["max_turns"] = max_turns
    else:
        logger.warning("Could not find 'TurnByTurnBiddingDiscussion' protocol to set max_turns.")

    # Set a new random seed for each game to ensure role/name shuffling is different
    config["seed"] = random.randint(0, 2**32 - 1)

    agent_harnesses = [f"llm/{model_name}"] * len(roles)

    return config, agent_harnesses


def plot_results(summary_data, output_dir):
    """
    Plots the results and saves them to files.
    """
    max_turns = sorted([int(k) for k in summary_data.keys()])
    metrics = ["total_cost", "total_tokens", "total_prompt_tokens", "total_completion_tokens"]

    for metric in metrics:
        means = [summary_data[str(t)][metric]["mean"] for t in max_turns]
        stds = [summary_data[str(t)][metric]["std"] for t in max_turns]

        plt.figure(figsize=(10, 6))
        plt.errorbar(max_turns, means, yerr=stds, fmt="-o", capsize=5, ecolor="red", markeredgecolor="black")
        plt.xlabel("Maximum Turns in Discussion")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs. Maximum Turns")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xticks(max_turns)

        plot_filename = os.path.join(output_dir, f"{metric}_vs_max_turns.png")
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Saved plot: {plot_filename}")


def plot_token_trajectories(trajectories_data, output_dir):
    """
    Plots token usage trajectories, grouped by max_turns, and saves them to files.
    """
    for metric, trajectories_by_turns in trajectories_data.items():
        if not trajectories_by_turns:
            continue

        plt.figure(figsize=(12, 8))

        # Create a color map for the different turn settings
        turn_keys = sorted(trajectories_by_turns.keys(), key=int)
        colors = plt.cm.viridis(np.linspace(0, 1, len(turn_keys)))
        color_map = {turns: color for turns, color in zip(turn_keys, colors)}

        for turns, trajectories in sorted(trajectories_by_turns.items(), key=lambda item: int(item[0])):
            for i, traj in enumerate(trajectories):
                # Only add a label to the first trajectory of each group for a clean legend
                label = f"Max Turns: {turns}" if i == 0 else None
                plt.plot(np.arange(len(traj)), traj, linestyle="-", alpha=0.4, color=color_map[turns], label=label)

        plt.title(f"{metric.replace('_', ' ').title()} per Query Step Trajectories")
        plt.xlabel("Query Step")
        plt.ylabel(f"{metric.replace('_', ' ').title()} per Query Step")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()

        plot_filename = os.path.join(output_dir, f"{metric}_trajectories.png")
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Saved trajectory plot: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(description="Measure LLM cost for the Werewolf game.")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs/run/comprehensive.yaml"),
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="cost_measurement",
        help="Output directory for logs, replays, and results.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        choices=LLM_MODEL_NAMES,
        help="LiteLLM model name to use for all agents.",
    )
    parser.add_argument("-d", "--disable_debug_mode", action="store_true", help="Disable debug mode.")

    args = parser.parse_args()

    # Create a unique subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    log_filename = f"measure_cost_{timestamp}"
    setup_logger(output_dir=run_output_dir, base_name=log_filename)
    logger.info(f"Starting cost measurement script. Results will be saved in: {run_output_dir}")

    # Load base game configuration
    with open(args.config_path, "r") as f:
        base_config = yaml.safe_load(f).get("game_config", {})

    max_turns_to_test = [8, 12, 16, 20, 24]
    runs_per_setting = 3
    results = {
        str(t): {"total_cost": [], "total_tokens": [], "total_prompt_tokens": [], "total_completion_tokens": []}
        for t in max_turns_to_test
    }
    all_trajectories = {
        "total_tokens": {str(t): [] for t in max_turns_to_test},
        "reasoning_tokens": {str(t): [] for t in max_turns_to_test},
        "text_tokens": {str(t): [] for t in max_turns_to_test},
    }

    for turns in max_turns_to_test:
        logger.info(f"--- Starting runs for max_turns = {turns} ---")
        for run in range(runs_per_setting):
            base_name = f"game_turns_{turns}_run_{run + 1}"
            logger.info(f"Starting {base_name}...")

            game_config, agent_harnesses = setup_game_config(turns, base_config, args.model_name)

            try:
                final_env = run_werewolf(
                    output_dir=run_output_dir,
                    base_name=base_name,
                    config=game_config,
                    agents=agent_harnesses,
                    debug=not args.disable_debug_mode,
                )

                # Extract cost summary
                cost_summary_dict = final_env.info.get("GAME_END", {}).get("cost_summary", {})
                if cost_summary_dict:
                    cost_summary = CostSummary(**cost_summary_dict)
                    results[str(turns)]["total_cost"].append(cost_summary.total_cost)
                    results[str(turns)]["total_tokens"].append(cost_summary.total_tokens)
                    results[str(turns)]["total_prompt_tokens"].append(cost_summary.total_prompt_tokens)
                    results[str(turns)]["total_completion_tokens"].append(cost_summary.total_completion_tokens)
                    logger.info(f"Finished {base_name}. Total Cost: ${cost_summary.total_cost:.4f}")

                    for agent_summary in cost_summary.cost_per_agent:
                        if agent_summary.data and agent_summary.data.usage_history:
                            usage_history_dicts = [usage.model_dump() for usage in agent_summary.data.usage_history]

                            total_tokens_traj = [usage.get("total_tokens", 0) or 0 for usage in usage_history_dicts]
                            all_trajectories["total_tokens"][str(turns)].append(total_tokens_traj)

                            reasoning_tokens_traj = [
                                usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
                                for usage in usage_history_dicts
                            ]
                            all_trajectories["reasoning_tokens"][str(turns)].append(reasoning_tokens_traj)

                            text_tokens_traj = [
                                (u.get("completion_tokens", 0) or 0)
                                - (u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0)
                                for u in usage_history_dicts
                            ]
                            all_trajectories["text_tokens"][str(turns)].append(text_tokens_traj)
                else:
                    logger.error(f"Could not find cost summary for {base_name}.")

            except Exception as e:
                logger.error(f"An error occurred during {base_name}: {e}", exc_info=True)

    # Calculate mean and standard deviation
    summary_data = {}
    for turns, metrics in results.items():
        summary_data[turns] = {}
        for metric, values in metrics.items():
            if values:
                summary_data[turns][metric] = {"mean": np.mean(values), "std": np.std(values), "raw_values": values}
            else:
                summary_data[turns][metric] = {"mean": 0, "std": 0, "raw_values": []}

    # Save summary to JSON
    summary_filename = os.path.join(run_output_dir, "cost_analysis_summary.json")
    with open(summary_filename, "w") as f:
        json.dump(summary_data, f, indent=4)
    logger.info(f"Saved summary results to {summary_filename}")

    # Plot results
    plot_results(summary_data, run_output_dir)
    plot_token_trajectories(all_trajectories, run_output_dir)

    logger.info("--- Cost measurement script finished ---")


if __name__ == "__main__":
    main()
