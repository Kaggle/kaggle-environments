# Integration Tests for kaggle-environments

This directory contains integration tests that run full episodes of each environment
using random agents. Tests are designed to run inside the Docker container built with
`docker/build_cpu.sh`.

## Quick Start

### 1. Build the Docker Image

```bash
cd /path/to/kaggle-environments
./docker/build_cpu.sh
```

### 2. Run All Integration Tests

```bash
./tests/integration/run_tests.sh
```

Or run specific tests:

```bash
# Run only RPS tests
./tests/integration/run_tests.sh -k "rps"

# Run with verbose output
./tests/integration/run_tests.sh --verbose

# Run a specific test class
./tests/integration/run_tests.sh -k "TestEpisodeExecution"
```

### 3. Run Tests Manually in Docker

```bash
docker run --rm \
    -v $(pwd)/tests:/usr/src/app/kaggle_environments/tests:ro \
    -e PYTHONUNBUFFERED=1 \
    --workdir /usr/src/app/kaggle_environments \
    python-simulations-cpu \
    python -m pytest tests/integration/test_envs.py -v -s
```

## Test Structure

### `test_envs.py` - Single Container Tests

Main integration tests that run in a single container:

- **TestEnvironmentDiscovery**: Verify environments are registered
- **TestEnvironmentCreation**: Test `make()` function for each environment
- **TestEpisodeExecution**: Run full episodes for each environment
- **TestEvaluateFunction**: Test the `evaluate()` convenience function
- **TestOpenSpielEnvironments**: Tests for OpenSpiel-wrapped games
- **TestEnvironmentRendering**: Test ANSI and JSON rendering
- **TestAgentTypes**: Test callable, lambda, and training agents

### `test_multicontainer.py` - Multi-Container Tests

Tests for running orchestrator and agents in separate containers (advanced):

- Requires `MULTICONTAINER_TEST=1` environment variable
- Uses Docker Compose for container orchestration
- Demonstrates production-like agent isolation

## Environments Tested

| Environment | Agents | Random Agent | Notes |
|-------------|--------|--------------|-------|
| rps | 2 | rock, paper, scissors | Rock-Paper-Scissors |
| connectx | 2 | random, negamax | Connect Four variant |
| halite | 2,4 | random | Space resource collection |
| hungry_geese | 2-8 | random, greedy | Multi-player snake |
| kore_fleets | 2,4 | random, miner | Fleet management |
| mab | 2+ | random | Multi-Armed Bandit |
| cabt | 2 | random | Card Battle |
| open_spiel_* | varies | random | OpenSpiel games |

### Environments with Special Requirements

- **werewolf**: Requires 6+ agents with specific configuration
- **lux_ai_s3**: Requires `luxai_s3` package
- **lux_ai_2021**: Requires external dependencies
- **football**: Requires `gfootball` package

## Multi-Container Setup (Advanced)

For production-like testing with isolated agent containers:

### Using Docker Compose

```bash
cd tests/integration

# Start orchestrator and agents
docker-compose up orchestrator agent-1 agent-2

# Run multi-container tests
MULTICONTAINER_TEST=1 docker-compose run test-runner
```

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Orchestrator  │────▶│    Agent 1      │
│   (Port 8080)   │     │   (Port 8081)   │
└────────┬────────┘     └─────────────────┘
         │
         │              ┌─────────────────┐
         └─────────────▶│    Agent 2      │
                        │   (Port 8082)   │
                        └─────────────────┘
```

## Adding New Tests

1. Add test cases to `test_envs.py` for new environments
2. Update `ENV_CONFIGS` dictionary with environment parameters
3. Add any special handling in the appropriate test class

Example:

```python
def test_new_env_episode(self):
    """Run a NewEnv episode."""
    env = make("new_env", configuration={"episodeSteps": 20})
    agents = ["random", "random"]
    steps = env.run(agents)
    
    assert env.done, "Episode should be done"
    final_state = steps[-1]
    print(f"\nNewEnv: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")
```

## Troubleshooting

### Docker Image Not Found

```bash
# Rebuild the image
./docker/build_cpu.sh
```

### Test Timeouts

Some environments (halite, kore_fleets) can take longer. Adjust `episodeSteps` in
the test configuration to reduce runtime.

### Missing Dependencies

Some environments require additional packages not in the base image:
- OpenSpiel games require `pyspiel`
- Lux AI requires `luxai_s3` or `luxai2021`
- Football requires `gfootball`

These tests are automatically skipped if dependencies are missing.
