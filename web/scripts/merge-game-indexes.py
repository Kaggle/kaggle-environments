#!/usr/bin/env python3
"""
Merge per-game index.json files with existing versions in GCS.

Each game's index.json is a JSON object describing the available visualizer
versions and which one should be used by default:

  {"versions": ["default", "v2"], "default": "v2"}

The "default" field is optional and is sourced from
web/config/default-visualizers.json (a game -> default-version map). When a
game has no entry there, the field is omitted and consumers fall back to
versions[0].

This script performs a union merge of the "versions" list so that visualizers
deployed by different branches coexist in the index. The legacy plain-array
format (e.g., ["default", "v2"]) is still accepted when reading an existing
index for backwards compatibility.

Usage:
  # Merge all games from a build directory (main branch builds):
  python3 merge-game-indexes.py --bucket kaggle-static --build-dir ./build

  # Merge a single game/visualizer (branch builds):
  python3 merge-game-indexes.py --bucket kaggle-static --game open_spiel_go --visualizer v2
"""
import argparse
import json
import os
import subprocess
import sys

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "default-visualizers.json"
)


def gcs_cat(path):
    """Read a file from GCS. Returns None if not found."""
    try:
        result = subprocess.run(
            ["gcloud", "storage", "cat", path],
            capture_output=True, text=True, check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def gcs_cp_stdin(content, dest):
    """Write string content to a GCS path via stdin pipe."""
    subprocess.run(
        ["gcloud", "storage", "cp", "-", dest],
        input=content, text=True, check=True,
    )


def load_default_visualizers(config_path):
    """Load the game -> default-version map. Returns {} if the file is absent."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def merge_visualizers(existing_json, new_names):
    """Union existing visualizer versions with new names, sorted for stability.

    Accepts both the legacy plain-array format (["default", "v2"]) and the
    object format ({"versions": [...], "default": ...}) for the existing index.
    Returns the merged, sorted list of version names.
    """
    existing = []
    if existing_json:
        parsed = json.loads(existing_json)
        if isinstance(parsed, dict):
            existing = parsed.get("versions", [])
        else:
            existing = parsed
    return sorted(set(existing) | set(new_names))


def build_index(versions, default=None):
    """Build the index.json object from a version list and optional default.

    The "default" field is only included when it names a version that is
    actually present, so the index never points at a missing visualizer.
    """
    index = {"versions": versions}
    if default and default in versions:
        index["default"] = default
    return index


def update_game_index(bucket, game, visualizers, defaults=None):
    """Download existing index for a game, merge in new visualizers, upload."""
    gcs_path = f"gs://{bucket}/episode-visualizers/{game}/index.json"
    existing = gcs_cat(gcs_path)
    versions = merge_visualizers(existing, visualizers)
    default = (defaults or {}).get(game)
    index = build_index(versions, default)
    print(f"  {game}: {index}")
    gcs_cp_stdin(json.dumps(index), gcs_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-game index.json files with existing versions in GCS."
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--build-dir", help="Build directory with game subdirs (batch mode)")
    parser.add_argument("--game", help="Single game name (single mode)")
    parser.add_argument("--visualizer", help="Single visualizer name (single mode)")
    parser.add_argument(
        "--default-config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the game -> default-visualizer JSON config",
    )
    args = parser.parse_args()

    defaults = load_default_visualizers(args.default_config)

    if args.game and args.visualizer:
        # Single game mode (branch builds)
        print(f"Updating index for {args.game}...")
        update_game_index(args.bucket, args.game, [args.visualizer], defaults)
    elif args.build_dir:
        # Batch mode (main branch builds)
        print("Updating per-game indexes...")
        for game in sorted(os.listdir(args.build_dir)):
            game_path = os.path.join(args.build_dir, game)
            if not os.path.isdir(game_path):
                continue
            # Each subdirectory of a game dir is a visualizer
            visualizers = [
                d for d in os.listdir(game_path)
                if os.path.isdir(os.path.join(game_path, d))
            ]
            if visualizers:
                update_game_index(args.bucket, game, visualizers, defaults)
        print("Per-game index update complete.")
    else:
        parser.error("Provide either --build-dir (batch mode) or --game and --visualizer (single mode)")


if __name__ == "__main__":
    main()
