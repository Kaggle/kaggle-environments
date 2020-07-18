# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import traceback
from . import errors, utils
from .agent import Agent
from .core import environments, evaluate, make
from logging.config import dictConfig

parser = argparse.ArgumentParser(description="Kaggle Simulations")
parser.add_argument(
    "action",
    choices=["list", "evaluate", "run", "step", "load", "act", "dispose", "http-server"],
    help="List environments. Evaluate many episodes. Run a single episode. Step the environment. Load the environment. Start http server.",
)
parser.add_argument("--environment", type=str,
                    help="Environment to run against.")
parser.add_argument("--debug", type=bool, help="Print debug statements.")
parser.add_argument(
    "--agents", type=str, nargs="*", help="Agent(s) to run with the environment."
)
parser.add_argument(
    "--configuration",
    type=json.loads,
    help="Environment configuration to setup the environment.",
)
parser.add_argument(
    "--steps",
    type=json.loads,
    help="Environment starting states (default=[resetState]).",
)
parser.add_argument(
    "--state",
    type=json.loads,
    help="Single agent state used for evaluation (default={}).",
)
parser.add_argument(
    "--episodes", type=int, help="Number of episodes to evaluate (default=1)"
)
parser.add_argument(
    "--render",
    type=json.loads,
    help="Response from run, step, or load. Calls environment render. (default={mode='json'})",
)
parser.add_argument(
    "--port", type=int, help="http-server Port (default=8000)."
)
parser.add_argument(
    "--host", type=str, help="http-server Host (default=127.0.0.1)."
)
parser.add_argument(
    "--in", type=str, help="Episode replay file to load. Only works when the action is load."
)
parser.add_argument(
    "--out", type=str, help="Output file to write the results of the episode. This does nothing when the action is http-server."
)
parser.add_argument(
    "--log", type=str, help="Agent log file to write the std out, resource, and step timing for each agent. Also used to load logs from a file with the load action."
)


def render(args, env):
    mode = utils.get(args.render, str, "json", path=["mode"])
    if mode == "human" or mode == "ansi":
        args.render["mode"] = "ansi"
    elif mode == "ipython" or mode == "html":
        args.render["mode"] = "html"
    else:
        args.render["mode"] = "json"
    result = env.render(**args.render)
    return result


def action_list(args):
    return json.dumps([*environments])


def action_evaluate(args):
    return json.dumps(
        evaluate(
            args.environment, args.agents, args.configuration, args.steps, args.episodes
        )
    )


cached_agent = None


def action_act(args):
    global cached_agent
    if len(args.agents) != 1:
        return {"error": "One agent must be provided."}
    raw = args.agents[0]

    # Pass empty steps in here because we just need the configuration from the environment
    env = make(args.environment, args.configuration, [], args.debug)
    config = env.configuration
    timeout = config.actTimeout

    if cached_agent is None or cached_agent.raw != raw:
        cached_agent = Agent(raw, env)
        timeout = config.agentTimeout
    observation = utils.get(args.state, dict, {}, ["observation"])
    action, log = cached_agent.act(observation, timeout)
    if isinstance(action, errors.DeadlineExceeded):
        action = "DeadlineExceeded"
    elif isinstance(action, BaseException):
        action = "BaseException::" + str(action)

    if args.log is not None:
        with open(args.log, mode="w+") as log_file:
            log_file.write(log)

    return {"action": action}


def action_step(args):
    env = make(args.environment, args.configuration, args.steps, args.logs, args.debug)
    runner = env.__agent_runner(args.agents)
    env.step(runner.act())
    if args.log is not None:
        with open(args.log, mode="w+") as log_file:
            json.dump(env.logs[-1], log_file)
            log_file.write(",")
    return render(args, env)


def action_run(args):
    env = make(args.environment, args.configuration, args.steps, args.logs, args.debug)
    env.run(args.agents)
    if args.log is not None:
        with open(args.log, mode="w") as log_file:
            json.dump(env.logs, log_file)
    return render(args, env)


def action_load(args):
    if args.input is not None:
        with open(args.input, mode="r") as replay_file:
            args = {**json.load(replay_file), **args}

    if args.log is not None:
        with open(args.log, mode="r") as log_file:
            args.logs = json.load(log_file)

    env = make(args.environment, args.configuration, args.steps, args.logs, args.debug)
    return render(args, env)


disposed = True


# This method is only called at the end of an episode to write the final array brace in the logs file
def action_dispose(args):
    global cached_agent, disposed
    if not disposed:
        cached_agent = None
        if args.log is not None:
            with open(args.log, mode="w+") as log_file:
                log_file.write("]")


def parse_args(args):
    return utils.structify(
        {
            "action": utils.get(args, str, "list", ["action"]),
            "agents": utils.get(args, list, [], ["agents"]),
            "configuration": utils.get(args, dict, {}, ["configuration"]),
            "environment": args.get("environment", None),
            "episodes": utils.get(args, int, 1, ["episodes"]),
            "state": utils.get(args, dict, {}, ["state"]),
            "steps": utils.get(args, list, [], ["steps"]),
            "logs": utils.get(args, list, [], ["logs"]),
            "render": utils.get(args, dict, {"mode": "json"}, ["render"]),
            "debug": utils.get(args, bool, False, ["debug"]),
            "host": utils.get(args, str, "127.0.0.1", ["host"]),
            "port": utils.get(args, int, 8000, ["port"]),
            "input": utils.get(args, str, None, ["in"]),
            "out": utils.get(args, str, None, ["out"]),
            "log": utils.get(args, str, None, ["log"]),
        }
    )

  
def action_handler(args):
    try:
        if args.action == "list":
            return action_list(args)
        elif args.action == "http-server":
            return {"error": "Already running a http server."}
        elif args.action == "act":
            return action_act(args)

        if args.environment is None:
            return {"error": "Environment required."}

        if args.action == "evaluate":
            return action_evaluate(args)
        elif args.action == "step":
            return action_step(args)
        elif args.action == "run":
            return action_run(args)
        elif args.action == "load":
            return action_load(args)
        elif args.action == "dispose":
            return action_dispose(args)
        else:
            return {"error": "Unknown Action"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def action_http(args):
    global disposed
    disposed = False
    # Write the opening array brace for the logs file if there is a logs file.
    if args.log is not None:
        with open(args.log, mode="w") as log_file:
            log_file.write("[")

    from flask import Flask, request

    # Setup logging to console for Flask
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '%(levelname)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        }
    })

    app = Flask(__name__, static_url_path="", static_folder="")
    app.route("/", methods=["GET", "POST"])(http_request)
    app.run(args.host, args.port, debug=True)


def http_request(request):
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return "", 204, headers

    headers = {"Access-Control-Allow-Origin": "*"}
    params = request.args.to_dict()
    for key in list(params.keys()):
        if key.endswith("[]"):
            params[key.replace("[]", "")] = request.args.getlist(key)
            del params[key]
        elif key.endswith("{}"):
            params[key.replace("{}", "")] = json.loads(params[key])
            del params[key]

    body = request.get_json(silent=True, force=True) or {}
    req = parse_args({**params, **body})
    resp = action_handler(req)
    return resp, 200, headers


def main():
    args = parser.parse_args()
    if args.action == "http-server":
        action_http(args)
    else:
        result = action_handler(parse_args(vars(args)))
        if args.out is None:
            print(result)
        else:
            with open(args.out, mode="w") as out_file:
                out_file.write(str(result))

        return 0


if __name__ == "__main__":
    main()
