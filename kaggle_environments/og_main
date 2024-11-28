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
from typing import *
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
    "--info",
    type=json.loads,
    help="Information about agents playing.",
)
parser.add_argument(
    "--steps",
    type=json.loads,
    help="Environment starting states (default=[resetState]).",
)
parser.add_argument(
    "--logs",
    type=json.loads,
    help="Environment starting logs (default=[]).",
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
    help="Response from run, step, or load. Calls environment render (default={mode='json'}).",
)
parser.add_argument(
    "--display",
    type=str,
    help="Shortcut to the --render {mode=''} argument (default json).",
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
    "--out", type=str, help="Output file to write the results of the episode. Does nothing in http-server mode."
)
parser.add_argument(
    "--log", type=str, help="Agent log file to write the std out, resource, and step timing for each agent. Also used to load logs from a file with the load action."
)


def render(args, env):
    mode = \
        args.display \
        if args.display is not None \
        else utils.get(args.render, str, "json", path=["mode"])

    if mode == "human" or mode == "ansi" or mode == "txt":
        args.render["mode"] = "ansi"
    elif mode == "ipython" or mode == "html":
        args.render["mode"] = "html"
    elif mode == "webm" or mode == "video":
        args.render["mode"] = "webm"
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

    env = make(args.environment, args.configuration, args.info, state=args.state, debug=args.debug)

    is_first_run = cached_agent is None or cached_agent.raw != raw
    if is_first_run:
        cached_agent = Agent(raw, env)
    observation = utils.get(args.state, dict, {}, ["observation"])
    action, log = cached_agent.act(observation)
    if isinstance(action, errors.DeadlineExceeded):
        action = "DeadlineExceeded"
    elif isinstance(action, BaseException):
        action = "BaseException::" + str(action)

    if args.log_path is not None:
        with open(args.log_path, mode="a") as log_file:
            if not is_first_run:
                log_file.write(",\n ")
            json.dump([log], log_file)

    return {"action": action}


def action_step(args):
    env = {"logs": args.logs}
    try:
        env = make(args.environment, args.configuration, args.info, args.steps, args.logs, args.debug)
        runner = env.__agent_runner(args.agents)
        env.step(runner.act())
    finally:
        if args.log_path is not None:
            with open(args.log_path, mode="a") as log_file:
                json.dump(env.logs[-1], log_file)
                log_file.write(",")
    return render(args, env)


def action_run(args):
    # Create a fake env so we can make the real env in our try body
    env = utils.structify({"logs": args.logs})
    try:
        env = make(args.environment, args.configuration, args.info, args.steps, args.logs, args.debug)
        env.run(args.agents)
    finally:
        if args.log_path is not None:
            with open(args.log_path, mode="w") as log_file:
                json.dump(env.logs, log_file, indent=2)
    return render(args, env)


def action_load(args):
    if args.log_path is not None:
        with open(args.log_path, mode="r") as log_file:
            args.logs = json.load(log_file)

    if args.in_path is not None:
        with open(args.in_path, mode="r") as replay_file:
            json_args = json.load(replay_file)
        env = make(json_args["name"], json_args["configuration"], json_args["info"], json_args["steps"], args.logs, args.debug)
    else:
        env = make(args.environment, args.configuration, args.info, args.steps, args.logs, args.debug)
    return render(args, env)


disposed = True


# This method is only called at the end of an episode to write the final array brace in the logs file and dispose the cached agent to force reinitialization
def action_dispose(args):
    global cached_agent, disposed
    if disposed:
        return "Already disposed"

    cached_agent = None
    if args.log_path is not None:
        with open(args.log_path, mode="a") as log_file:
            log_file.write("]")
    disposed = True
    return "Successfully disposed"


def parse_args(args):
    return utils.structify(
        {
            "action": utils.get(args, str, "list", ["action"]),
            "agents": utils.get(args, list, [], ["agents"]),
            "configuration": utils.get(args, dict, {}, ["configuration"]),
            "environment": args.get("environment", args.get("name", None)),
            "episodes": utils.get(args, int, 1, ["episodes"]),
            "state": utils.get(args, dict, {}, ["state"]),
            "steps": utils.get(args, list, [], ["steps"]),
            "logs": utils.get(args, list, [], ["logs"]),
            "render": utils.get(args, dict, {"mode": "json"}, ["render"]),
            "display": utils.get(args, str, None, ["display"]),
            "debug": utils.get(args, bool, False, ["debug"]),
            "host": utils.get(args, str, "127.0.0.1", ["host"]),
            "port": utils.get(args, int, 8000, ["port"]),
            "in_path": utils.get(args, str, None, ["in"]),
            "out_path": utils.get(args, str, None, ["out"]),
            "log_path": utils.get(args, str, None, ["log"]),
            "info": utils.get(args, dict, {}, ["info"]),
        }
    )

  
def action_handler(args):
    try:
        if args.action == "list":
            return action_list(args)
        if args.action == "http-server":
            return {"error": "Already running a http server."}
        if args.action == "act":
            return action_act(args)
        if args.action == "dispose":
            return action_dispose(args)
        if args.action == "load":
            return action_load(args)

        if args.environment is None:
            return {"error": "Environment required."}

        if args.action == "evaluate":
            return action_evaluate(args)
        if args.action == "step":
            return action_step(args)
        if args.action == "run":
            return action_run(args)

        return {"error": "Unknown Action"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


log_path: Optional[str] = None


def action_http(args):
    from flask import Flask, request

    if args.log_path is not None:
        global log_path
        log_path = args.log_path

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
    app.route("/", methods=["GET", "POST"])(lambda: http_request(request))
    app.run(args.host, args.port, debug=args.debug, use_reloader=args.debug)


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
    args = {**params, **body}
    if "render" in args and isinstance(args["render"], str):
        # Manually deserialize render argument
        # We should eventually refactor this to use the same deserializer as the cmd line arg parser
        args["render"] = json.loads(args["render"])
    args = parse_args(args)
    if args.log_path is None:
        args.log_path = log_path

    global disposed
    # Write the opening array brace for the logs file if there is a logs file.
    if disposed and args["action"] != "dispose" and args.log_path is not None:
        with open(args.log_path, mode="w") as log_file:
            log_file.write("[")
        disposed = False

    resp = action_handler(args)
    return resp, 200, headers


def main():
    args = parse_args(vars(parser.parse_args()))
    if args.action == "http-server":
        action_http(args)
    else:
        result = action_handler(args)
        if args.out_path is None:
            print(result)
        else:
            with open(args.out_path, encoding="utf-8", mode="w") as out_file:
                out_file.write(str(result))

        return 0


if __name__ == "__main__":
    main()
