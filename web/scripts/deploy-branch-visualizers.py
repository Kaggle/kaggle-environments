#!/usr/bin/env python3
"""
Deploy branch-owned visualizers staged by plan-branch-deploy.js.

Reads the deploy plan from /workspace/branch-deploy/plan.json, rsyncs
each visualizer's assets to GCS, and updates per-game index.json files.
"""
import argparse
import json
import os
import subprocess
import sys

STAGE_DIR = "/workspace/branch-deploy"
PLAN_FILE = os.path.join(STAGE_DIR, "plan.json")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy branch-owned visualizers and update per-game indexes."
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    args = parser.parse_args()

    if not os.path.exists(PLAN_FILE):
        print("No deploy plan found, nothing to do.")
        return

    with open(PLAN_FILE) as f:
        plan = json.load(f)

    if not plan:
        print("Empty deploy plan, nothing to do.")
        return

    print(f"Deploying {len(plan)} visualizer(s)...")

    for entry in plan:
        game, viz = entry["game"], entry["viz"]
        source = os.path.join(STAGE_DIR, game, viz)
        dest = f"gs://{args.bucket}/episode-visualizers/{game}/{viz}/"

        print(f"\nDeploying {game}/{viz}...")
        subprocess.run([
            "gcloud", "storage", "rsync",
            "--delete-unmatched-destination-objects",
            "--recursive", source, dest,
        ], check=True)

    # Update per-game indexes
    print("\nUpdating per-game indexes...")
    for entry in plan:
        game, viz = entry["game"], entry["viz"]
        subprocess.run([
            "python3", "web/scripts/merge-game-indexes.py",
            "--bucket", args.bucket,
            "--game", game,
            "--visualizer", viz,
        ], check=True)

    print(f"\nDone! Deployed {len(plan)} visualizer(s).")


if __name__ == "__main__":
    main()
