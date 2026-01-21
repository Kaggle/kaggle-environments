"""
HTTP server wrapper for kaggle-environments agents using relay's protobuf serialization.

This runs in agent containers and wraps user agent functions to expose them via HTTP
with protobuf serialization (compatible with relay's HTTPClient).

Usage:
    python agent_server.py --agent-path /path/to/agent.py --port 8080
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from flask import Flask, Response, request


def load_agent_function(agent_path: str):
    """Load agent function from a Python file."""
    path = Path(agent_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent from {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    # Find the agent function (last callable in the module)
    callables = [v for v in vars(module).values() if callable(v) and not v.__name__.startswith("_")]
    if not callables:
        raise ValueError(f"No callable agent function found in {agent_path}")

    return callables[-1]


def create_agent_server(agent_function, port: int = 8080):
    """Create Flask server that wraps agent function with protobuf serialization."""
    # Import kaggle_evaluation dependencies lazily
    import kaggle_evaluation.core.generated.kaggle_evaluation_pb2 as kaggle_evaluation_proto
    from kaggle_evaluation.core.relay import _deserialize, _serialize

    app = Flask(__name__)

    @app.route("/<endpoint>", methods=["POST"])
    def handle_request(endpoint):
        """Handle protobuf-encoded requests."""
        try:
            # Parse protobuf request
            request_proto = kaggle_evaluation_proto.KaggleEvaluationRequest()
            request_proto.ParseFromString(request.data)

            # Deserialize arguments
            args = [_deserialize(arg) for arg in request_proto.args]
            kwargs = {key: _deserialize(value) for key, value in request_proto.kwargs.items()}

            # Call agent function
            # Agent functions typically take (observation, configuration)
            if len(args) >= 2:
                observation, configuration = args[0], args[1]
            else:
                observation = args[0] if args else kwargs.get("observation", {})
                configuration = kwargs.get("configuration", {})

            action = agent_function(observation, configuration)

            # Serialize response
            response_proto = kaggle_evaluation_proto.KaggleEvaluationResponse(payload=_serialize(action))

            return Response(response_proto.SerializeToString(), mimetype="application/x-protobuf")

        except Exception as e:
            # Return error as protobuf
            error_proto = kaggle_evaluation_proto.KaggleEvaluationResponse(payload=_serialize({"error": str(e)}))
            return Response(error_proto.SerializeToString(), status=500, mimetype="application/x-protobuf")

    return app


def main():
    parser = argparse.ArgumentParser(description="Run agent as HTTP server with protobuf serialization")
    parser.add_argument("--agent-path", required=True, help="Path to agent Python file")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")

    args = parser.parse_args()

    # Load agent function
    agent_function = load_agent_function(args.agent_path)
    print(f"Loaded agent function: {agent_function.__name__}")

    # Create and run server
    app = create_agent_server(agent_function, args.port)
    print(f"Starting agent server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
