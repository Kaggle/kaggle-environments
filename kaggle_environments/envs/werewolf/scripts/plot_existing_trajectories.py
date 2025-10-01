import argparse
import glob
import json
import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from kaggle_environments.envs.werewolf.werewolf import CostSummary

# Add the project root to the Python path to allow importing from kaggle_environments
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_token_trajectories(trajectories_data, output_dir):
    """
    Plots token usage trajectories, grouped by max_turns, and saves them to files.
    """
    for metric, trajectories_by_turns in trajectories_data.items():
        if not trajectories_by_turns:
            logger.warning(f"No data found for metric '{metric}'. Skipping plot.")
            continue

        plt.figure(figsize=(12, 8))

        # Create a color map for the different turn settings
        turn_keys = sorted(trajectories_by_turns.keys(), key=int)
        colors = plt.cm.viridis(np.linspace(0, 1, len(turn_keys)))
        color_map = {turns: color for turns, color in zip(turn_keys, colors)}

        for turns, trajectories in sorted(trajectories_by_turns.items(), key=lambda item: int(item[0])):
            for i, traj in enumerate(trajectories):
                if not all(isinstance(x, (int, float)) for x in traj):
                    logger.error(
                        f"Trajectory for metric '{metric}' (turns={turns}) contains non-numeric data. Skipping."
                    )
                    continue
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
    parser = argparse.ArgumentParser(
        description="Load data from a measure_cost.py output directory and generate token trajectory plots."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the output directory of a previous measure_cost.py run.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    logger.info(f"Loading data from: {args.input_dir}")

    all_trajectories = {"total_tokens": {}, "reasoning_tokens": {}, "text_tokens": {}}

    # Find all game replay JSON files
    game_files = glob.glob(os.path.join(args.input_dir, "game_*_run_*.json"))
    if not game_files:
        logger.error(f"No game replay files (game_*_run_*.json) found in {args.input_dir}.")
        return

    logger.info(f"Found {len(game_files)} game replay files to process.")

    for game_file in game_files:
        # Extract max_turns from filename
        match = re.search(r"game_turns_(\d+)_run_", os.path.basename(game_file))
        if not match:
            logger.warning(f"Could not parse max_turns from filename: {game_file}. Skipping.")
            continue
        turns = match.group(1)

        with open(game_file, "r") as f:
            game_data = json.load(f)

        cost_summary_dict = game_data.get("info", {}).get("GAME_END", {}).get("cost_summary")
        if not cost_summary_dict:
            logger.warning(f"No cost_summary found in {game_file}. Skipping.")
            continue

        cost_summary = CostSummary(**cost_summary_dict)

        for agent_summary in cost_summary.cost_per_agent:
            if agent_summary.data and agent_summary.data.usage_history:
                usage_history_dicts = [usage.model_dump() for usage in agent_summary.data.usage_history]

                total_tokens_traj = [usage.get("total_tokens", 0) or 0 for usage in usage_history_dicts]
                all_trajectories["total_tokens"].setdefault(turns, []).append(total_tokens_traj)

                reasoning_tokens_traj = [
                    usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
                    for usage in usage_history_dicts
                ]
                all_trajectories["reasoning_tokens"].setdefault(turns, []).append(reasoning_tokens_traj)

                text_tokens_traj = [
                    (u.get("completion_tokens", 0) or 0)
                    - (u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0)
                    for u in usage_history_dicts
                ]
                all_trajectories["text_tokens"].setdefault(turns, []).append(text_tokens_traj)

    logger.info("Finished processing all files. Generating plots...")
    plot_token_trajectories(all_trajectories, args.input_dir)
    logger.info(f"--- Script finished. Plots saved in {args.input_dir} ---")


if __name__ == "__main__":
    main()
