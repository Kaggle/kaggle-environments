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
from kaggle_environments import environments, evaluate, make, utils


parser = argparse.ArgumentParser(description="Kaggle Simulations")
parser.add_argument(
    "action",
    choices=["list", "evaluate", "run", "step", "load", "http-server"],
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
    "--episodes", type=int, help="Number of episodes to evaluate (default=1)"
)
parser.add_argument(
    "--render",
    type=json.loads,
    help="Response from run, step, or load. Calls environment render. (default={mode='json'})",
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


def action_step(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    env.step(env.__get_actions(args.agents))
    return render(args, env)


def action_run(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    env.run(args.agents)
    return render(args, env)


def action_load(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    return render(args, env)


def action_handler(args):
    args = utils.structify(
        {
            "action": utils.get(args, str, "list", ["action"]),
            "agents": utils.get(args, list, [], ["agents"]),
            "configuration": utils.get(args, dict, {}, ["configuration"]),
            "environment": args.get("environment", None),
            "episodes": utils.get(args, int, 1, ["episodes"]),
            "steps": utils.get(args, list, [], ["steps"]),
            "render": utils.get(args, dict, {"mode": "json"}, ["render"]),
            "debug": utils.get(args, bool, False, ["debug"])
        }
    )

    for index, agent in enumerate(args.agents):
        agent = utils.read_file(agent, agent)
        args.agents[index] = utils.get_last_callable(agent, agent)

    if args.action == "list":
        return action_list(args)

    if args.environment == None:
        return {"error": "Environment required."}

    try:
        if args.action == "http-server":
            return {"error": "Already running a http server."}
        elif args.action == "evaluate":
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

    app = Flask(__name__, static_url_path="", static_folder="")
    app.route("/", methods=["GET", "POST"])(lambda: http_request(request))
    app.run("127.0.0.1", 8000, debug=True)


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
    args = {**params, **body}
    return (action_handler(args), 200, headers)

def main():
    args = parser.parse_args()
    if args.action == "http-server":
        action_http(args)
    print(action_handler(vars(args)))

if __name__ == "__main__":
    main()
