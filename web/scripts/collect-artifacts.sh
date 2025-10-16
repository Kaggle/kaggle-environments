#!/bin/bash
# This script collects the build artifacts from all game visualizer packages
# and places them into a root-level /build directory, structured for deployment.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Get the root directory of the project
ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR="$ROOT_DIR/build"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Main Logic ---
echo "--- Starting Artifact Collection ---"

# 1. Clean up previous build directory
echo "Cleaning up old build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 2. Find all visualizer packages and process them
echo "Finding visualizer packages..."
pnpm m ls --json --depth -1 | jq -c '.[] | select(if .name then .name | test("@kaggle-environments/.*-visualizer") else false end)' |
while IFS= read -r line; do
  # Parse package details from the JSON line
  pkg_name=$(echo "$line" | jq -r '.name')
  pkg_path=$(echo "$line" | jq -r '.path')
  source_dist="$pkg_path/dist"

  if [ -d "$source_dist" ]; then
    # Extract game and visualizer names from the path
    game_name=$(echo "$pkg_path" | sed 's|.*/envs/\([^/]*\)/.*|\1|')
    visualizer_name=$(basename "$pkg_path")
    dest_dir="$BUILD_DIR/$game_name/$visualizer_name"

    echo "Collecting '$pkg_name' from '$source_dist'"
    echo "  -> Destination: $dest_dir"

    # Create destination directory and copy artifacts
    mkdir -p "$dest_dir"
    cp -r "$source_dist"/* "$dest_dir/"
  else
    echo "Skipping '$pkg_name': No 'dist' directory found at '$source_dist'"
  fi
done

echo "--- Artifact Collection Complete ---"
echo "All artifacts have been collected in: $BUILD_DIR"
