#!/bin/bash
set -e

# Change to the directory where the script is located
cd "$(dirname "$0")"

uv sync

uv run pytest tests/ kaggle_environments/
