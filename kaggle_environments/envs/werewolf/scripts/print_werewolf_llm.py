import json
import os
import re
import sys

# ANSI Colors
GREEN = "\033[92m"
RESET = "\033[0m"


def format_json_string(s: str) -> str:
    """Attempts to pretty-print a JSON string or a markdown-fenced JSON string."""
    if not s:
        return s

    # Try to extract from markdown fences
    if "```json" in s:
        start = s.find("```json") + 7
        end = s.find("```", start)
        s = s[start:end].strip()
    elif "```" in s:
        start = s.find("```") + 3
        end = s.find("```", start)
        s = s[start:end].strip()

    try:
        data = json.loads(s)
        return json.dumps(data, indent=2)
    except:
        return s


def pretty_print_game(json_path: str):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    with open(json_path, "r") as f:
        try:
            game_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    steps = game_data.get("steps", [])
    info = game_data.get("info", {})
    game_end = info.get("GAME_END", {})
    config_agents = game_data.get("configuration", {}).get("agents", [])

    print("=" * 100)
    print(f"GAME RECORD: {json_path}")
    print("=" * 100)
    print("NOTE: Green parenthesized display names are added for visual aid and are not present in the raw event logs.")
    print()

    # State tracking
    roster = []
    alive_players = []
    revealed_roles = {}  # pid -> role
    id_to_dname = {}

    # --- Populate ID Map first for Enrichment ---
    # Try using GAME_END first (most reliable for final state)
    source_agents = game_end.get("all_players", []) if (game_end and "all_players" in game_end) else []

    # Fallback to configuration if GAME_END missing/empty
    if not source_agents and config_agents:
        source_agents = [{"agent": a, "id": a.get("id")} for a in config_agents]

    # First pass: count for disambiguation
    name_counts = {}
    for p in source_agents:
        agent = p.get("agent", {}) if "agent" in p else p
        dname = agent.get("display_name", agent.get("name", "Unknown"))
        name_counts[dname] = name_counts.get(dname, 0) + 1

    current_counts = {}
    for p in source_agents:
        agent = p.get("agent", {}) if "agent" in p else p
        pid = agent.get("id", p.get("id", "Unknown"))
        dname = agent.get("display_name", agent.get("name", "Unknown"))

        final_dname = dname
        if name_counts.get(dname, 0) > 1:
            cur = current_counts.get(dname, 0) + 1
            current_counts[dname] = cur
            final_dname = f"{dname} ({cur})"

        id_to_dname[pid] = final_dname
        # Store back in p for the table loop if it matches
        p["_disambiguated_name"] = final_dname

    # Regex for enrichment
    # Sort by length descending to avoid partial matches if IDs share prefixes (though \b handles words)
    sorted_ids = sorted(id_to_dname.keys(), key=len, reverse=True)
    if sorted_ids:
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted_ids)) + r")\b")
    else:
        pattern = None

    def enrich_text(text: str) -> str:
        if not pattern or not text:
            return text

        def replace(match):
            pid = match.group(1)
            dname = id_to_dname.get(pid, "")
            return f"{pid} {GREEN}({dname}){RESET}"

        return pattern.sub(replace, text)

    # --- 0. Detailed Player Config ---
    if game_end and "all_players" in game_end:
        print("--- DETAILED PLAYER CONFIG (AFTER SHUFFLE) ---")
        print(f"{'PLAYER ID':<10} | {'DISPLAY NAME':<20} | {'ROLE':<15} | {'TEAM':<15} | {'STATUS'}")
        print("-" * 90)

        def infer_team(r):
            if r == "Werewolf":
                return "Werewolf"
            if r in ["Villager", "Seer", "Doctor"]:
                return "Villager"
            return "Unknown"

        for p in game_end.get("all_players", []):
            agent = p.get("agent", {})
            pid = agent.get("id", p.get("id", "Unknown"))
            role = agent.get("role", "Unknown")
            team = agent.get("team", infer_team(role))
            status = "Alive" if p.get("alive") else f"Eliminated (Day {p.get('eliminated_during_day')})"
            dname = p.get("_disambiguated_name", "Unknown")

            dname_fmt = (dname[:17] + "...") if len(dname) > 20 else dname
            print(f"{pid:<10} | {dname_fmt:<20} | {role:<15} | {team:<15} | {status}")

            roster.append(pid)
            if p.get("alive"):
                alive_players.append(pid)

        print("-" * 90)
        print()

    action_count = 0
    seen_event_descriptions = set()

    # If roster still empty (no GAME_END), we'll fill it from logs

    try:
        for step_idx, step in enumerate(steps):
            # 1. Collect and Print any unique Global Moderator Events from all player views
            step_events = []
            for agent_state in step:
                obs = agent_state.get("observation", {})
                raw_obs = obs.get("raw_observation", {})
                event_views = raw_obs.get("new_player_event_views", [])

                for event in event_views:
                    desc = event.get("description", "").strip()
                    if desc and desc not in seen_event_descriptions:
                        if event.get("source") == "MODERATOR":
                            step_events.append(event)
                            seen_event_descriptions.add(desc)

            step_events.sort(key=lambda x: x.get("created_at", ""))

            for event in step_events:
                event_name = event.get("event_name")
                data = event.get("data") or {}
                desc = event.get("description", "").strip()

                if event_name == "moderator_announcement":
                    if "player_ids" in data and not roster:
                        roster = data["player_ids"]
                        alive_players = list(roster)
                        print("--- INITIAL ROSTER (From Log) ---")
                        # We can direct print this list with enrichment
                        print(f"Players: {enrich_text(', '.join(roster))}")
                        print("-" * 50)
                        print()

                elif event_name == "elimination":
                    pid = data.get("eliminated_player_id")
                    if pid and pid in alive_players:
                        alive_players.remove(pid)
                    role = data.get("eliminated_player_role_name")
                    if pid and role:
                        revealed_roles[pid] = role

                print(f"--- GLOBAL EVENT (Step {step_idx}) ---")
                print(enrich_text(desc))

                if event_name == "discussion_order":
                    order = data.get("chat_order_of_player_ids", [])
                    dead = [p for p in roster if p not in order and p not in alive_players]
                    if dead:
                        dead_str = ", ".join([f"{enrich_text(p)} ({revealed_roles.get(p, 'Unknown')})" for p in dead])
                        print(f"NOTE: Missing from order (Eliminated): {dead_str}")

                elif event_name == "vote_order":
                    order = data.get("vote_order_of_player_ids", [])
                    dead = [p for p in roster if p not in order and p not in alive_players]
                    if dead:
                        dead_str = ", ".join([f"{enrich_text(p)} ({revealed_roles.get(p, 'Unknown')})" for p in dead])
                        print(f"NOTE: Missing from voting (Eliminated): {dead_str}")

                print("-" * 50)
                print()

            # 2. Print Player Actions
            for agent_idx, agent_state in enumerate(step):
                action = agent_state.get("action")
                if not action:
                    continue

                kwargs = action.get("kwargs", {}) if isinstance(action, dict) else {}
                raw_prompt = kwargs.get("raw_prompt")
                raw_completion = kwargs.get("raw_completion")

                if raw_prompt or raw_completion:
                    action_count += 1
                    obs = agent_state.get("observation", {})
                    raw_obs = obs.get("raw_observation", {})

                    player_id = raw_obs.get("player_id", obs.get("player_id", f"Agent_{agent_idx}"))
                    role = raw_obs.get("role", obs.get("role", "Unknown"))
                    day = raw_obs.get("day", obs.get("day", "?"))
                    phase = raw_obs.get("detailed_phase", obs.get("detailed_phase", "Unknown"))

                    cost = kwargs.get("cost")
                    p_tokens = kwargs.get("prompt_tokens")
                    c_tokens = kwargs.get("completion_tokens")

                    print(f"ACTION #{action_count} | STEP: {step_idx} | DAY: {day} | PHASE: {phase}")
                    # Enrich player ID in header
                    print(f"PLAYER: {enrich_text(player_id)} ({role})")
                    if p_tokens is not None or c_tokens is not None:
                        usage = []
                        if p_tokens:
                            usage.append(f"Prompt: {p_tokens}")
                        if c_tokens:
                            usage.append(f"Completion: {c_tokens}")
                        if cost:
                            usage.append(f"Cost: ${cost:.5f}")
                        print(f"USAGE: {' | '.join(usage)}")

                    print("-" * 50)

                    if raw_prompt:
                        print("PROMPT:")
                        print(enrich_text(raw_prompt.strip()))
                        print()

                    if raw_completion:
                        print("LLM OUTPUT (Formatted):")
                        print(enrich_text(format_json_string(raw_completion)))
                        print()

                    print("=" * 100)
                    print()
                    sys.stdout.flush()

        # --- Final Results ---
        if game_end:
            print("=" * 100)
            print("FINAL RESULTS")
            print("=" * 100)

            winner_team = game_end.get("winner_team", "Unknown")
            winner_ids = game_end.get("winner_ids", [])
            loser_ids = game_end.get("loser_ids", [])
            reason = game_end.get("reason", "No specific reason provided.")

            print(f"WINNING TEAM: {winner_team}")
            print(f"WINNERS: {enrich_text(', '.join(winner_ids))}")
            print(f"LOSERS:  {enrich_text(', '.join(loser_ids))}")
            print(f"REASON:  {enrich_text(reason)}")
            print("-" * 50)

            print("FINAL PLAYER STATUS:")
            for p in game_end.get("all_players", []):
                agent = p.get("agent", {})
                pid = agent.get("id", p.get("id", "Unknown"))
                dname = p.get("_disambiguated_name", agent.get("display_name", agent.get("name", "Unknown")))
                role = agent.get("role", "Unknown")
                status = "Alive" if p.get("alive") else f"Eliminated (Day {p.get('eliminated_during_day')})"
                outcome = "WON" if pid in winner_ids else "LOST"
                # Enrich PID here too
                print(f"- {enrich_text(pid):<10} [{role}]: {status} [{outcome}]")

        if action_count == 0:
            print("No LLM interactions found in this game record.")

    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 print_werewolf_llm.py <path_to_game_json> [--force-color]")
        sys.exit(1)

    path = sys.argv[1]
    # Check for force color flag
    if "--force-color" in sys.argv:
        # Force enable
        pass
    elif not sys.stdout.isatty():
        # Disable colors if not TTY (e.g. piped to less)
        GREEN = ""
        RESET = ""

    pretty_print_game(path)
