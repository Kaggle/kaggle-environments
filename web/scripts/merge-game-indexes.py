#!/usr/bin/env python3
"""
Merge per-game index.json files with existing versions in GCS.

Each game's index.json is a JSON array of available visualizer names
(e.g., ["default", "v2"]). This script performs a union merge so that
visualizers deployed by different branches coexist in the index.

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


def merge_visualizers(existing_json, new_names):
    """Union existing visualizer list with new names, sorted for stability."""
    existing = json.loads(existing_json) if existing_json else []
    return sorted(set(existing) | set(new_names))


def update_game_index(bucket, game, visualizers):
    """Download existing index for a game, merge in new visualizers, upload."""
    gcs_path = f"gs://{bucket}/episode-visualizers/{game}/index.json"
    existing = gcs_cat(gcs_path)
    merged = merge_visualizers(existing, visualizers)
    print(f"  {game}: {merged}")
    gcs_cp_stdin(json.dumps(merged), gcs_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-game index.json files with existing versions in GCS."
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--build-dir", help="Build directory with game subdirs (batch mode)")
    parser.add_argument("--game", help="Single game name (single mode)")
    parser.add_argument("--visualizer", help="Single visualizer name (single mode)")
    args = parser.parse_args()

    if args.game and args.visualizer:
        # Single game mode (branch builds)
        print(f"Updating index for {args.game}...")
        update_game_index(args.bucket, args.game, [args.visualizer])
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
                update_game_index(args.bucket, game, visualizers)
        print("Per-game index update complete.")
    else:
        parser.error("Provide either --build-dir (batch mode) or --game and --visualizer (single mode)")


if __name__ == "__main__":
    main()
