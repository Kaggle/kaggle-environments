# Proto-Based Remote Game Drivers Integration

This directory contains the integration of `remote_game_drivers` with `kaggle_environments` using proto-based networking via `kaggle_evaluation.core.relay`.

## Overview

This integration bypasses the traditional `kaggle_environments` schema-based networking system (which uses the `requests` library) and instead uses protobuf serialization via `kaggle_evaluation.core.relay`. This allows for:

1. **Schema-less communication**: No need to define custom JSON schemas for each game
2. **Rich type support**: Native support for DataFrames, numpy arrays, Pydantic models, dataclasses, and enums
3. **Unified networking**: Single proto schema handles all game types
4. **Better performance**: gRPC-based communication with efficient serialization

## Architecture

### Key Components

1. **`ProtoAgent`** (`kaggle_environments/proto_agent.py`):
   - Agent class that communicates via protobuf instead of HTTP
   - Uses `relay.Client` for serialization/deserialization
   - Compatible with `kaggle_environments` agent system

2. **`proto_relay.py`** (`kaggle_environments/proto_relay.py`):
   - Utility functions for proto-based communication
   - Wrappers around `kaggle_evaluation.core.relay` functions
   - Server and client creation helpers

3. **`proto_driver.py`** (this directory):
   - Environment definition for remote game drivers
   - Minimal interpreter/renderer (game logic handled by driver)
   - `ProtoGameDriverWrapper` for integration

### Communication Flow

```
┌─────────────────────┐         ┌──────────────────────┐
│ kaggle_environments │         │  Inference Server    │
│                     │         │  (User's Agent)      │
│  ProtoAgent         │◄───────►│                      │
│  (relay.Client)     │  gRPC   │  process_turn()      │
│                     │  Proto  │  (relay.Server)      │
└─────────────────────┘         └──────────────────────┘
         │
         │ Manages
         ▼
┌─────────────────────┐
│  GameDriver         │
│  (remote_game_      │
│   drivers)          │
└─────────────────────┘
```

## Usage

### 1. Define a Proto Agent

```python
from kaggle_environments import make

# Proto agent specification
proto_agent = {
    'type': 'proto',
    'proto_config': {
        'channel_address': 'localhost',
        'port': 50051,
    }
}
```

### 2. Create an Inference Server

```python
import kaggle_evaluation.core.relay as relay
from remote_game_drivers.gymnasium_remote_driver.remote_agent import RemoteAgent

class MyAgent(RemoteAgent):
    def choose_action(self, action_space, observation, info=None):
        # Your agent logic here
        return action

def process_turn(game_data):
    """Endpoint called by ProtoAgent"""
    agent = MyAgent()
    action_space = game_data['action_space']
    observation = game_data['observation']
    return agent.deliver_agent_info(action_space, observation)

# Start server
server, port = relay.define_server(process_turn, port=50051)
server.start()
```

### 3. Run the Environment

```python
from kaggle_environments import make

# Create environment
env = make('proto_driver_gymnasium')

# Define agents
agents = [
    {'type': 'proto', 'proto_config': {'port': 50051}},
    {'type': 'proto', 'proto_config': {'port': 50052}},
]

# Run
env.run(agents)
```

## Comparison with Traditional Approach

### Traditional (Schema-based with requests)

**Pros:**
- Simple HTTP-based communication
- Easy to debug with standard HTTP tools

**Cons:**
- Requires custom JSON schema for each game
- Limited type support (JSON primitives only)
- Manual serialization for complex types
- HTTP overhead

### Proto-based (This Integration)

**Pros:**
- No custom schemas needed per game
- Rich type support (DataFrames, numpy, Pydantic, etc.)
- Efficient gRPC communication
- Automatic serialization/deserialization
- Consistent with Hearth gateway patterns

**Cons:**
- Requires `kaggle_evaluation` dependency
- More complex setup initially

## Integration with Existing Code

This integration follows the patterns established in:

- **`hearth/game_drivers/gymnasium/gym_gateway.py`**: Gateway that manages game execution
- **`hearth/game_drivers/gymnasium/gym_inference_server.py`**: Inference server pattern
- **`hearth/kaggle_evaluation/core/relay.py`**: Proto serialization utilities

The key difference is that this brings those patterns into `kaggle_environments` for broader use.

## Supported Data Types

Via `kaggle_evaluation.core.relay`, the following types are natively supported:

- **Primitives**: str, int, float, bool, None
- **Collections**: list, tuple, dict (with str keys)
- **Scientific**: pandas.DataFrame, polars.DataFrame, numpy.ndarray
- **Data Models**: Pydantic BaseModel, dataclasses, Enums
- **Binary**: io.BytesIO

## Example: Gymnasium Integration

```python
from remote_game_drivers.gymnasium_remote_driver.game_driver import GameDriver
from kaggle_environments.envs.remote_game_drivers.proto_driver import create_proto_driver_environment

# Create environment for a specific game
env_spec = create_proto_driver_environment(
    game_driver_class=GameDriver,
    game_name='Blackjack-v1',
    driver_config={
        'agent_ids': ['agent_0', 'agent_1'],
        'game_config': {'random_seed': 42}
    }
)

# Register with kaggle_environments
from kaggle_environments import register
register('proto_blackjack', env_spec)
```

## Dependencies

- `kaggle_evaluation` (from hearth)
- `remote_game_drivers`
- `grpc` (via kaggle_evaluation)
- `protobuf` (via kaggle_evaluation)

## Files in This Directory

- **`proto_driver.py`**: Main environment implementation
- **`proto_driver.json`**: Minimal JSON specification
- **`example_usage.py`**: Usage examples and patterns
- **`README.md`**: This file
- **`__init__.py`**: Package exports

## Future Enhancements

1. **Auto-registration**: Automatically register all remote_game_drivers games
2. **Configuration helpers**: Simplify driver_config setup
3. **Visualization**: Better integration with game-specific renderers
4. **Testing utilities**: Helper functions for testing proto agents
5. **Port management**: Automatic port allocation for multiple agents

## Troubleshooting

### Import Errors

If you see import errors for `kaggle_evaluation`:
```python
import sys
sys.path.insert(0, "/path/to/hearth")
```

### Connection Issues

If agents can't connect:
- Verify server is running on expected port
- Check firewall settings
- Ensure `relay.define_server()` completed successfully

### Serialization Errors

If you see serialization errors:
- Check that data types are supported (see list above)
- For custom types, use Pydantic models or dataclasses
- Set allowed modules with `relay.set_allowed_modules()`

## See Also

- `kaggle_environments/proto_agent.py`: ProtoAgent implementation
- `kaggle_environments/proto_relay.py`: Relay utilities
- `hearth/kaggle_evaluation/core/relay.py`: Core relay implementation
- `remote_game_drivers/core/base_classes.py`: Game driver base classes
