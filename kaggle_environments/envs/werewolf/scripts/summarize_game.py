import json
import os
import sys
import re
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models for Structured Output ---

class PlayerStats(BaseModel):
    player_id: str
    display_name: str
    role: str
    persuasion: int = Field(..., description="1-10: How much did others follow this player?")
    deception: int = Field(..., description="1-10: How well did they hide their true nature (for wolves) or avoid mis-elimination (for villagers)?")
    aggression: int = Field(..., description="1-10: How often did they initiate attacks?")
    analysis: int = Field(..., description="1-10: Quality of their deductions.")

class PlayerHighlight(BaseModel):
    player_name: str
    role: str
    summary: str
    key_move: str

class EntertainmentMetrics(BaseModel):
    excitement_score: int = Field(..., description="1-10 rating of how entertaining the game was to watch.")
    dramatic_moments: List[str] = Field(..., description="List of specific explosive or turning point moments.")
    outcome_type: str = Field(..., description="e.g., 'Nail-biter', 'Stomp', 'Chaos', 'Masterclass', 'Throw'")
    mvp_id: str = Field(..., description="The ID of the player who contributed most to the entertainment or outcome.")

class GameAnalysis(BaseModel):
    title: str
    narrative_summary: str
    winner_team: str
    mvp_player: str
    mvp_reasoning: str
    best_play: str
    biggest_mistake: str
    player_highlights: List[PlayerHighlight]
    player_stats: List[PlayerStats]
    entertainment_metrics: EntertainmentMetrics

# --- Log Extraction Logic (Adapted from print_werewolf_llm.py) ---

def format_json_string(s: str) -> str:
    """Attempts to pretty-print a JSON string or a markdown-fenced JSON string."""
    if not s:
        return s
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

from copy import deepcopy
from kaggle_environments.envs.werewolf.game.roles import shuffle_ids

# --- NameManager (Adapted from add_audio.py) ---

class NameManager:
    """Manages player display names and disambiguation."""

    def __init__(self, replay_data: Dict):
        self.id_to_display = {}
        self._initialize_names(replay_data)

    def _initialize_names(self, replay_data: Dict):
        """Disambiguates display names matching werewolf.py logic including randomization."""
        config = replay_data.get("configuration", {})
        agents = deepcopy(config.get("agents", []))  # Deepcopy to avoid modifying original
        info = replay_data.get("info", {})

        # 1. Inject Kaggle Display Names (pre-shuffle)
        # Match logic from werewolf.py: inject_kaggle_scheduler_info
        kaggle_agents_info = info.get("Agents")
        if kaggle_agents_info and isinstance(kaggle_agents_info, list):
            for agent, kaggle_agent_info in zip(agents, kaggle_agents_info):
                display_name = kaggle_agent_info.get("Name", "")
                if display_name:
                    agent["display_name"] = display_name

        # 2. Handle Randomization (Match roles.py logic)
        seed = config.get("seed")
        randomize_ids = config.get("randomize_ids", False)

        if randomize_ids and seed is not None:
            # roles.py: shuffle_ids matches agents to NEW ids
            agents = shuffle_ids(agents, seed + 123)

        # 3. Count occurrences
        name_counts = {}
        for agent in agents:
            name = agent.get("display_name") or agent.get("name") or agent.get("id")
            name_counts[name] = name_counts.get(name, 0) + 1

        # 4. Assign unique names and build map
        current_counts = {}
        for agent in agents:
            agent_id = agent.get("id")  # This is now the randomized ID if applicable
            name = agent.get("display_name") or agent.get("name") or agent.get("id")

            if name_counts[name] > 1:
                current_counts[name] = current_counts.get(name, 0) + 1
                unique_name = f"{name} ({current_counts[name]})"
            else:
                unique_name = name

            self.id_to_display[agent_id] = unique_name

    def get_name(self, agent_id: str) -> str:
        """Returns the disambiguated display name for an agent ID."""
        return self.id_to_display.get(agent_id, agent_id)

    def replace_names(self, text: str) -> str:
        """Replaces all occurrences of known agent IDs in text with display names."""
        if not text:
            return text

        # Sort by length descending to prevent partial matches
        sorted_ids = sorted(self.id_to_display.keys(), key=len, reverse=True)

        for agent_id in sorted_ids:
            display_name = self.id_to_display[agent_id]
            if agent_id != display_name:
                # Use regex for whole word matching, optionally consuming "Player" prefix
                pattern = r'\b(?:Player\s*)?' + re.escape(agent_id) + r'\b'
                text = re.sub(pattern, display_name, text)

        return text

def extract_game_transcript(json_path: str) -> str:
    if not os.path.exists(json_path):
        return f"Error: File {json_path} not found."

    with open(json_path, "r") as f:
        try:
            game_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error decoding JSON: {e}"

    name_manager = NameManager(game_data)

    steps = game_data.get("steps", [])
    info = game_data.get("info", {})
    game_end = info.get("GAME_END", {})
    config_agents = game_data.get("configuration", {}).get("agents", [])

    transcript = []
    transcript.append(f"GAME RECORD: {json_path}")
    transcript.append("=" * 50)

    # Roster reconstruction
    roster = []
    alive_players = []
    revealed_roles = {}
    winner_ids = game_end.get("winner_ids", [])
    
    # 1. Player List
    source_agents = game_end.get("all_players", []) if (game_end and "all_players" in game_end) else []
    if not source_agents and config_agents:
        source_agents = [{"agent": a, "id": a.get("id")} for a in config_agents]

    transcript.append("ROSTER:")
    for p in source_agents:
        agent_data = p.get("agent", {}) if "agent" in p else p
        pid = agent_data.get("id", p.get("id", "Unknown"))
        role = agent_data.get("role", "Unknown")
        display_name = name_manager.get_name(pid)
        
        roster_line = f"- {display_name}"
        if display_name != pid:
            roster_line += f" (ID: {pid})"
        roster_line += f" -> Role: {role}"
        
        transcript.append(roster_line)
        roster.append(pid)
        alive_players.append(pid)
    
    transcript.append("=" * 50)
    transcript.append("EVENTS:")

    seen_event_descriptions = set()

    for step_idx, step in enumerate(steps):
        # 1. Global Events
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
            
            # Skip some spammy events
            if event_name in ["vote_request", "chat_request", "phase_change", "phase_divider"]:
                continue
            
            # Apply name replacement to description
            desc = name_manager.replace_names(desc)
            transcript.append(f"[Step {step_idx}] {desc}")

            if event_name == "elimination":
                pid = data.get("eliminated_player_id")
                role = data.get("eliminated_player_role_name")
                if pid:
                    display_pid = name_manager.get_name(pid)
                    transcript.append(f"   -> {display_pid} (Role: {role}) eliminated.")

        # 2. Player Actions (Chat, Vote, Ability)
        for agent_idx, agent_state in enumerate(step):
            action = agent_state.get("action")
            if not action:
                continue

            kwargs = action.get("kwargs", {}) if isinstance(action, dict) else {}
            
            raw_completion = kwargs.get("raw_completion")
            
            obs = agent_state.get("observation", {})
            raw_obs = obs.get("raw_observation", {})
            player_id = raw_obs.get("player_id", obs.get("player_id", f"Agent_{agent_idx}"))
            display_player = name_manager.get_name(player_id)
            
            if raw_completion:
                # Try to parse the JSON output from the agent
                try:
                    parsed = json.loads(format_json_string(raw_completion))
                    message = parsed.get("message")
                    reasoning = parsed.get("reasoning")
                    target = parsed.get("target_id")
                    
                    if message:
                        # Replace names in message
                        msg_display = name_manager.replace_names(message)
                        transcript.append(f"[Step {step_idx}] {display_player} says: \"{msg_display}\"")
                        if reasoning:
                            res_display = name_manager.replace_names(reasoning)
                            transcript.append(f"   (Internal Thought: {res_display})")
                    elif target:
                         # It's an action (Vote, Heal, Seer)
                         display_target = name_manager.get_name(target)
                         transcript.append(f"[Step {step_idx}] {display_player} performs action on {display_target}.")
                         if reasoning:
                             res_display = name_manager.replace_names(reasoning)
                             transcript.append(f"   (Internal Thought: {res_display})")
                         
                except:
                    pass

    return "\n".join(transcript)


# --- Gemini Summarization ---

def get_gcloud_project() -> Optional[str]:
    """Attempts to get the Google Cloud project ID from environment or gcloud config."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        return project
    
    try:
        # Try gcloud
        import subprocess
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, check=True
        )
        project = result.stdout.strip()
        if project and "unset" not in project:
            return project
    except Exception:
        pass
    return None

def summarize_with_gemini(transcript: str, model_id: str = "gemini-3-pro-preview") -> Optional[GameAnalysis]:
    api_key = os.environ.get("GEMINI_API_KEY")
    client = None

    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        project = get_gcloud_project()
        if project:
            print(f"Using Vertex AI with project: {project}")
            client = genai.Client(vertexai=True, project=project, location="us-central1")
        else:
            print("Error: No GEMINI_API_KEY found and could not determine GOOGLE_CLOUD_PROJECT.")
            return None

    prompt = f"""
You are an expert commentator and analyst for the game of Werewolf (also known as Mafia).
I will provide you with a structured log of a game session.
Your task is to analyze the game and provide a detailed summary, highlighting the key moments, player performances, and strategy.

Here is the game transcript:
{transcript}

Please provide the analysis in the requested structured JSON format.
"""

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GameAnalysis,
                temperature=0.5,
            )
        )
        
        # Check if response is valid
        if not response.parsed:
             print("Error: Gemini response could not be parsed.")
             # print(response.text) # Debugging
             return None
             
        return response.parsed

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

# --- Main Execution ---

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize Werewolf Game")
    parser.add_argument("-i", "--input_path", required=True, help="Path to the game replay JSON")
    parser.add_argument("-o", "--output_dir", help="Directory to save outputs (defaults to input file's directory)")
    parser.add_argument("--model", default="gemini-3-pro-preview", help="Gemini Model ID")
    parser.add_argument("--dry-run", action="store_true", help="Generate transcript only, do not call LLM")
    args = parser.parse_args()

    json_path = args.input_path
    model_id = args.model
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(os.path.abspath(json_path))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")

    print(f"Reading game log from: {json_path}")
    transcript = extract_game_transcript(json_path)
    
    if len(transcript) < 100:
        print("Transcript is too short or empty. Something went wrong with extraction.")
        print(transcript)
        sys.exit(1)
        
    print(f"Transcript length: {len(transcript)} characters.")
    
    # Always save transcript
    with open(transcript_path, "w") as f:
        f.write(transcript)
    print(f"Transcript saved to: {transcript_path}")

    if args.dry_run:
        return

    print(f"Sending to Gemini ({model_id})...")
    
    analysis = summarize_with_gemini(transcript, model_id)
    
    if analysis:
        # Save structured JSON
        with open(summary_path, "w") as f:
             f.write(analysis.model_dump_json(indent=2))
        print(f"Summary saved to: {summary_path}")

        print("\n" + "="*50)
        print(f"GAME SUMMARY: {analysis.title}")
        print("="*50)
        print(f"\n{analysis.narrative_summary}\n")
        
        print(f"Winner: {analysis.winner_team}")
        print(f"MVP: {analysis.mvp_player} - {analysis.mvp_reasoning}")
        print(f"Best Play: {analysis.best_play}")
        print(f"Biggest Mistake: {analysis.biggest_mistake}")
        
        print("\n" + "-"*30)
        print("ENTERTAINMENT METRICS")
        print("-"*30)
        print(f"Score: {analysis.entertainment_metrics.excitement_score}/10 ({analysis.entertainment_metrics.outcome_type})")
        print("Dramatic Moments:")
        for moment in analysis.entertainment_metrics.dramatic_moments:
            print(f"- {moment}")

        print("\n" + "-"*30)
        print("PLAYER STATS")
        print("-"*30)
        
        for stat in analysis.player_stats:
            print(f"\n{stat.display_name} ({stat.role})")
            print(f"  Persuasion: {stat.persuasion}/10 | Deception: {stat.deception}/10")
            print(f"  Aggression: {stat.aggression}/10 | Analysis:  {stat.analysis}/10")

        print("\n" + "-"*30)
        print("PLAYER HIGHLIGHTS")
        print("-"*30)
        
        for player in analysis.player_highlights:
            print(f"\nPlayer: {player.player_name} ({player.role})")
            print(f"Summary: {player.summary}")
            print(f"Key Move: {player.key_move}")
            
    else:
        print("Failed to generate analysis.")

if __name__ == "__main__":
    main()
