import pytest
import argparse
import json
from kaggle_environments import make, errors, utils
"""
{
  "roles": ["WEREWOLF", "VILLAGER", "VILLAGER", "SEER", "DOCTOR"],
  "names": ["gpt-4o", "claude-3", "gemini-pro", "player-4", "player-5"],
  "player_thumbnails": {
    "gpt-4o": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude-3": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "gemini-pro": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png"
  }
}
"""


URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png"
}

def test_llm_players(html_file, json_file, log_file, log_path):
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager"]
    names = ["gemini-2.0-flash-1", "gemini-2.0-flash-2", 
             "gemini-2.5-pro-3", "gemini-2.5-pro-4", "gemini-2.5-pro-5", "gemini-2.5-pro-6", "gemini-2.5-pro-7"]
    thumbnails = [URLS['gemini'], URLS['gemini'], URLS['gemini'], URLS['gemini'], URLS['gemini'], URLS['gemini'],
                  URLS['gemini']]
    agents = ['llm/vertex_ai/gemini-2.0-flash', 'llm/vertex_ai/gemini-2.0-flash', 
              "llm/vertex_ai/gemini-2.5-pro","llm/vertex_ai/gemini-2.5-pro",
              "llm/vertex_ai/gemini-2.5-pro","llm/vertex_ai/gemini-2.5-pro","llm/vertex_ai/gemini-2.5-pro"]

    agents_config = [{"role": role, "id": name, "agent_id": agent, "thumbnail": url} for role, name, agent, url in
                     zip(roles, names, agents, thumbnails)]
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "actTimeout": 30,
            "agents": agents_config
        }
    )
    env_logs = utils.structify({"logs": args.logs})
    env.run(agents)
    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        out = env.renderer(state, env)
        print(out)

    if args.log_path is not None:
            with open(args.log_path, mode="w") as log_file:
                json.dump(env_logs.logs, log_file, indent=2)

    env_out = env.render(mode='html')
    with open(html_file,'w') as out:
        out.write(env_out)
    env_out = env.render(mode='json') 
    with open(json_file,'w') as out :
        out.write(env_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test_run")
    parser.add_argument("--html",type=str,help="html output file", required=True)
    parser.add_argument("--json",type=str,help="json output file", required=True)
    parser.add_argument("--logs",type=str,help="log file")
    parser.add_argument("--log_path",type=str,help="log file")
    args = parser.parse_args()
    test_llm_players(args.html, args.json, args.logs, args.log_path)