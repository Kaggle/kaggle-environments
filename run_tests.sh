#!/bin/bash
# Test runner for kaggle-environments
#
# Usage:
#   ./run_tests.sh                    # Run tests locally with uv
#   ./run_tests.sh --docker           # Run tests in Docker container
#   ./run_tests.sh -k "rps"           # Run only RPS tests (local)
#   ./run_tests.sh --docker -k "rps"  # Run only RPS tests (Docker)
#   ./run_tests.sh --verbose          # Run with verbose output

set -e

# Change to the directory where the script is located
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

USE_DOCKER=false
PYTEST_ARGS="--tb=short"

while [ $# -gt 0 ]; do
    case $1 in
        --docker)
            USE_DOCKER=true
            shift
            ;;
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
            echo "  --docker         Run tests in Docker container"
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

if [ "$USE_DOCKER" = true ]; then
    echo "=============================================="
    echo "Kaggle Environments Tests (Docker)"
    echo "=============================================="
    echo ""

    # Check if Docker image exists
    if ! docker image inspect python-simulations-cpu >/dev/null 2>&1; then
        echo "Docker image 'python-simulations-cpu' not found."
        echo "Building image with docker/build_cpu.sh..."
        "$REPO_ROOT/docker/build_cpu.sh"
    fi

    echo "Running tests in Docker container..."
    echo ""

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
else
    echo "=============================================="
    echo "Kaggle Environments Tests (Local)"
    echo "=============================================="
    echo ""

    uv sync

    echo "Running tests with uv..."
    echo ""

    uv run pytest tests/ kaggle_environments/ $PYTEST_ARGS
fi
