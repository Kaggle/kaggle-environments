import argparse
import json
import logging
import os

from kaggle_environments import make

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """
    Rerenders a Werewolf game replay HTML file from an existing game record JSON.
    This is useful for updating the replay viewer to the latest version without
    rerunning the entire game simulation.
    """
    parser = argparse.ArgumentParser(
        description="Rerender a Werewolf game HTML replay from a JSON game record.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_json",
        type=str,
        required=True,
        help="Path to the input game record JSON file (e.g., werewolf_game.json).",
    )
    parser.add_argument(
        "-o", "--output_html", type=str, required=True, help="Path to write the newly rendered HTML output file."
    )
    args = parser.parse_args()

    logging.info(f"Loading game record from: {args.input_json}")
    if not os.path.exists(args.input_json):
        logging.error(f"Error: Input file not found at {args.input_json}")
        return

    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            replay_data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Failed to decode JSON from {args.input_json}. The file might be corrupted.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the file: {e}")
        return

    logging.info("Successfully loaded game data. Initializing Kaggle environment...")

    # The environment name should be stored in the replay, but we default to 'werewolf'
    env_name = replay_data.get("name", "werewolf")
    if env_name != "werewolf":
        logging.warning(f"Game record is for '{env_name}', but we are rendering with the 'werewolf' environment.")

    try:
        # Recreate the environment state from the replay file
        env = make(
            "werewolf",
            configuration=replay_data.get("configuration"),
            steps=replay_data.get("steps", []),
            info=replay_data.get("info", {}),
        )
        logging.info("Environment initialized. Rendering new HTML...")

        # Render the HTML. This will use the werewolf.js file included in the
        # installed kaggle_environments package.
        html_content = env.render(mode="html")

        output_dir = os.path.dirname(args.output_html)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        logging.info(f"Successfully rerendered HTML to: {args.output_html}")

    except Exception as e:
        logging.error(f"An error occurred during environment creation or rendering: {e}")
        logging.error(
            "Please ensure the 'kaggle_environments' package is correctly installed and the JSON file is valid."
        )


if __name__ == "__main__":
    main()
