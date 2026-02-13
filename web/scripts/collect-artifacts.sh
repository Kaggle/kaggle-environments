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

# SKIP_GAMES: comma-separated list of patterns to skip deployment
# Patterns can be:
#   - Game name only: "open_spiel_go" - skips all visualizers for that game
#   - Game/visualizer: "open_spiel_chess/v2" - skips only that specific visualizer
# Example: SKIP_GAMES="open_spiel_go,open_spiel_chess/v2" skips go entirely and only chess v2
SKIP_GAMES="${SKIP_GAMES:-}"

# Function to check if a game/visualizer should be skipped
# Args: $1 = game_name, $2 = visualizer_name
should_skip_game() {
  local game_name="$1"
  local visualizer_name="$2"
  if [ -z "$SKIP_GAMES" ]; then
    return 1  # Don't skip if SKIP_GAMES is not set
  fi

  # Split SKIP_GAMES by comma and check each pattern
  IFS=',' read -ra patterns <<< "$SKIP_GAMES"
  for pattern in "${patterns[@]}"; do
    # Trim whitespace
    pattern=$(echo "$pattern" | xargs)
    if [ -z "$pattern" ]; then
      continue
    fi

    # Check if pattern includes a visualizer (contains /)
    if [[ "$pattern" == *"/"* ]]; then
      # Pattern is game/visualizer - must match both
      local pattern_game="${pattern%/*}"
      local pattern_viz="${pattern#*/}"
      if [[ "$game_name" == *"$pattern_game"* ]] && [[ "$visualizer_name" == "$pattern_viz" ]]; then
        return 0  # Skip this specific visualizer
      fi
    else
      # Pattern is game only - skip all visualizers for matching games
      if [[ "$game_name" == *"$pattern"* ]]; then
        return 0  # Skip this game
      fi
    fi
  done
  return 1  # Don't skip
}

# --- Main Logic ---
echo "--- Starting Artifact Collection ---"

if [ -n "$SKIP_GAMES" ]; then
  echo "SKIP_GAMES is set: '$SKIP_GAMES'"
  echo "Games matching these patterns will NOT be deployed."
fi

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

    # Check if this game/visualizer should be skipped
    if should_skip_game "$game_name" "$visualizer_name"; then
      echo "Skipping '$pkg_name': '$game_name/$visualizer_name' matches SKIP_GAMES pattern"
      continue
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
