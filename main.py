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
from kaggle_environments import Agent, environments, errors, evaluate, make, utils

parser = argparse.ArgumentParser(description="Kaggle Simulations")
parser.add_argument(
    "action",
    choices=["list", "evaluate", "run", "step", "load", "act", "http-server"],
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
    "--middleware", type=str, help="Path to request middleware for use with the http-server."
)
parser.add_argument(
    "--port", type=int, help="http-server Port (default=8000)."
)
parser.add_argument(
    "--host", type=str, help="http-server Host (default=127.0.0.1)."
)


def render(args, env):
    mode = utils.get(args.render, str, "json", path=["mode"])
    if mode == "human" or mode == "ansi":
        args.render["mode"] = "ansi"
    elif mode == "ipython" or mode == "html":
        args.render["mode"] = "html"
    else:
        args.render["mode"] = "json"
    return env.render(**args.render)


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

    # Process the configuration.
    # (additional environment specificproperties come along without being checked).
    err, config = utils.structify(utils.process_schema(
        utils.schemas["configuration"], args.configuration))
    if err:
        return {"error": err}
    timeout = config.actTimeout

    if cached_agent == None or cached_agent.id != raw:
        if cached_agent != None:
            cached_agent.destroy()
        cached_agent = Agent(raw, config, raw)
        timeout = config.agentTimeout
    state = {
        "observation": utils.get(args.state, dict, {}, ["observation"]),
        "reward": args.get("reward", None),
        "info": utils.get(args.state, dict, {}, ["info"])
    }
    action = cached_agent.act(state, timeout)
    if isinstance(action, errors.DeadlineExceeded):
        action = "DeadlineExceeded"
    elif isinstance(action, BaseException):
        action = "BaseException"

    return {"action": action}


def action_step(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    runner = env.__agent_runner(args.agents)
    env.step(runner.act())
    runner.destroy()
    return render(args, env)


def action_run(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    env.run(args.agents)
    return render(args, env)


def action_load(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    return render(args, env)


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
            "render": utils.get(args, dict, {"mode": "json"}, ["render"]),
            "debug": utils.get(args, bool, False, ["debug"]),
            "host": utils.get(args, str, "127.0.0.1", ["host"]),
            "port": utils.get(args, int, 8000, ["port"])
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

        if args.environment == None:
            return {"error": "Environment required."}

        if args.action == "evaluate":
            return action_evaluate(args)
        elif args.action == "step":
            return action_step(args)
        elif args.action == "run":
            return action_run(args)
        elif args.action == "load":
            return action_load(args)
        else:
            return {"error": "Unknown Action"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def action_http(args):
    from flask import Flask, request

    middleware = {"request": None, "response": None}
    if args.middleware != None:
        try:
            raw = utils.read_file(args.middleware)
            local = utils.get_exec(raw)
            middleware["request"] = utils.get(
                local, path=["request"], is_callable=True)
            middleware["response"] = utils.get(
                local, path=["response"], is_callable=True)
        except Exception as e:
            return {"error": str(e), "trace": traceback.format_exc()}

    app = Flask(__name__, static_url_path="", static_folder="")
    app.route("/", methods=["GET", "POST"]
              )(lambda: http_request(request, middleware))
    app.run(args.host, args.port, debug=True)


def http_request(request, middleware):
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

        return ("", 204, headers)

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
    if middleware["request"] != None:
        req = middleware["request"](req)

    resp = action_handler(req)
    if middleware["response"] != None:
        resp = middleware["response"](req, resp)

    return (resp, 200, headers)


def main():
    args = parser.parse_args()
    if args.action == "http-server":
        action_http(args)
    print(action_handler(parse_args(vars(args))))


if __name__ == "__main__":
    main()
