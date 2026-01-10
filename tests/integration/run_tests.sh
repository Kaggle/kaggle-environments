#!/bin/bash
# Integration test runner for kaggle-environments
#
# This script runs integration tests inside the Docker container.
# The container must be built first using: docker/build_cpu.sh
#
# Usage:
#   ./run_tests.sh              # Run all integration tests
#   ./run_tests.sh -k "rps"     # Run only RPS tests
#   ./run_tests.sh --verbose    # Run with verbose output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default pytest arguments
PYTEST_ARGS="-v -s --tb=short"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            PYTEST_ARGS="$PYTEST_ARGS -vv"
            shift
            ;;
        -k)
            PYTEST_ARGS="$PYTEST_ARGS -k $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Verbose output"
            echo "  -k PATTERN       Only run tests matching PATTERN"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            # Pass through any other arguments to pytest
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

echo "=============================================="
echo "Kaggle Environments Integration Tests"
echo "=============================================="
echo ""

# Check if Docker image exists
if ! docker image inspect python-simulations-cpu >/dev/null 2>&1; then
    echo "Docker image 'python-simulations-cpu' not found."
    echo "Building image with docker/build_cpu.sh..."
    "$REPO_ROOT/docker/build_cpu.sh"
fi

echo "Running tests in Docker container..."
echo "Pytest args: $PYTEST_ARGS"
echo ""

# Run tests in Docker container
docker run --rm \
    -v "$REPO_ROOT/tests:/usr/src/app/kaggle_environments/tests:ro" \
    -v "$REPO_ROOT/kaggle_environments:/usr/src/app/kaggle_environments/kaggle_environments:ro" \
    -e PYTHONUNBUFFERED=1 \
    --workdir /usr/src/app/kaggle_environments \
    python-simulations-cpu \
    python -m pytest tests/integration/test_envs.py $PYTEST_ARGS

echo ""
echo "=============================================="
echo "Tests completed successfully!"
echo "=============================================="
