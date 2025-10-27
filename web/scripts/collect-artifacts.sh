#!/bin/bash
# This script collects the build artifacts from all game visualizer packages
# and places them into a root-level /build directory, structured for deployment.
# It also generates a manifest.json file mapping games to their visualizers.

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

# 2. Initialize manifest
manifest_json="{}"

# 3. Find all visualizer packages and process them
echo "Finding visualizer packages..."
# The only change is on the line below!
while IFS= read -r line; do
  # Parse package details from the JSON line
  pkg_name=$(echo "$line" | jq -r '.name')
  pkg_path=$(echo "$line" | jq -r '.path')
  source_dist="$pkg_path/dist"

  if [ -d "$source_dist" ]; then
    # Extract game and visualizer names from the path
    game_dir=$(dirname "$(dirname "$pkg_path")")
    game_name=$(basename "$game_dir")
    visualizer_name=$(basename "$pkg_path")
    # Open Spiel game names are configured dynamically to be prefixed with "open_spiel_"
    if [[ "$pkg_path" == *"open_spiel_env"* ]]; then
      game_name="open_spiel_$game_name"
    fi
    dest_dir="$BUILD_DIR/$game_name/$visualizer_name"

    echo "Collecting '$pkg_name' from '$source_dist'"
    echo "  -> Destination: $dest_dir"

    # Create destination directory and copy artifacts
    mkdir -p "$dest_dir"
    cp -r "$source_dist"/* "$dest_dir/"

    # Add entry to the manifest
    echo "  -> Updating manifest for '$game_name'"
    manifest_json=$(echo "$manifest_json" | jq --arg game "$game_name" --arg viz "$visualizer_name" '
      if .[$game] == null then
        .[$game] = [$viz]
      else
        .[$game] += [$viz]
      end
    ')
  else
    echo "Skipping '$pkg_name': No 'dist' directory found at '$source_dist'"
  fi
done < <(pnpm m ls --json --depth -1 | jq -c '.[] | select(if .name then .name | test("@kaggle-environments/.*-visualizer") else false end)')

# 4. Write the manifest file
echo "Writing manifest.json..."
echo "$manifest_json" | jq . > "$BUILD_DIR/manifest.json" # Pretty-print the final JSON

echo "--- Artifact Collection Complete ---"
echo "All artifacts and manifest.json have been collected in: $BUILD_DIR"
