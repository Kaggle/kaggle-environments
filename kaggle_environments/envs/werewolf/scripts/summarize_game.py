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

class PlayerHighlight(BaseModel):
    player_name: str = Field(..., description="The name of the player.")
    role: str = Field(..., description="The role of the player (e.g., Werewolf, Seer, Villager).")
    rating: int = Field(..., description="A rating from 1 to 10 of the player's performance.")
    summary: str = Field(..., description="A brief summary of the player's performance and playstyle.")
    key_move: str = Field(..., description="The most impactful action or message by this player.")

class GameAnalysis(BaseModel):
    title: str = Field(..., description="A creative and catchy title for the game summary.")
    narrative_summary: str = Field(..., description="A detailed narrative summary of the game, highlighting the flow of events and key turning points.")
    winner_team: str = Field(..., description="The team that won the game.")
    mvp_player: str = Field(..., description="The name of the MVP player.")
    mvp_reasoning: str = Field(..., description="Why this player was chosen as MVP.")
    best_play: str = Field(..., description="The single best strategic move in the game.")
    biggest_mistake: str = Field(..., description="The biggest strategic error made in the game.")
    player_highlights: List[PlayerHighlight] = Field(..., description="Highlights for each player in the game.")

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

def extract_game_transcript(json_path: str) -> str:
    if not os.path.exists(json_path):
        return f"Error: File {json_path} not found."

    with open(json_path, "r") as f:
        try:
            game_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error decoding JSON: {e}"

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
        agent = p.get("agent", {}) if "agent" in p else p
        pid = agent.get("id", p.get("id", "Unknown"))
        role = agent.get("role", "Unknown")
        transcript.append(f"- {pid} ({role})")
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
            
            # Skip some spammy events if needed, but keeping most for context
            if event_name in ["vote_request", "chat_request", "phase_change", "phase_divider"]:
                continue

            transcript.append(f"[Step {step_idx}] {desc}")

            if event_name == "elimination":
                pid = data.get("eliminated_player_id")
                role = data.get("eliminated_player_role_name")
                if pid:
                    transcript.append(f"   -> {pid} (Role: {role}) eliminated.")

        # 2. Player Actions (Chat, Vote, Ability)
        for agent_idx, agent_state in enumerate(step):
            action = agent_state.get("action")
            if not action:
                continue

            kwargs = action.get("kwargs", {}) if isinstance(action, dict) else {}
            # We specifically look for 'message' in raw_completion for chat
            # Or just use the high-level description if available, but raw content is better for determining 'play style'
            
            raw_prompt = kwargs.get("raw_prompt")
            raw_completion = kwargs.get("raw_completion")
            
            obs = agent_state.get("observation", {})
            raw_obs = obs.get("raw_observation", {})
            player_id = raw_obs.get("player_id", obs.get("player_id", f"Agent_{agent_idx}"))
            
            if raw_completion:
                # Try to parse the JSON output from the agent to get the 'message' or 'reasoning'
                try:
                    parsed = json.loads(format_json_string(raw_completion))
                    message = parsed.get("message")
                    reasoning = parsed.get("reasoning")
                    target = parsed.get("target_id")
                    
                    if message:
                        transcript.append(f"[Step {step_idx}] {player_id} says: \"{message}\"")
                        if reasoning:
                            transcript.append(f"   (Internal Thought: {reasoning})")
                    elif target:
                         # It's an action (Vote, Heal, Seer)
                         action_type = "Actions"
                         if "Note" in str(parsed): pass 
                         # We rely on the moderator event for the PUBLIC result, but the INTERNAL THOUGHT is key for the summary.
                         transcript.append(f"[Step {step_idx}] {player_id} performs action on {target}.")
                         if reasoning:
                             transcript.append(f"   (Internal Thought: {reasoning})")
                         
                except:
                    # If parsing fails, just ignore or print raw?
                    # Start with ignoring to keep it clean, or print sanitized
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
    parser.add_argument("json_path", help="Path to the game replay JSON")
    # Set default model to gemini-3-pro-preview
    parser.add_argument("model_id", nargs="?", default="gemini-3-pro-preview", help="Gemini Model ID")
    parser.add_argument("--dry-run", action="store_true", help="Generate transcript only, do not call LLM")
    args = parser.parse_args()

    json_path = args.json_path
    model_id = args.model_id

    print(f"Reading game log from: {json_path}")
    transcript = extract_game_transcript(json_path)
    
    if len(transcript) < 100:
        print("Transcript is too short or empty. Something went wrong with extraction.")
        print(transcript)
        sys.exit(1)
        
    print(f"Transcript length: {len(transcript)} characters.")

    if args.dry_run:
        with open("transcript.txt", "w") as f:
            f.write(transcript)
        print("Transcript saved to transcript.txt")
        return

    print(f"Sending to Gemini ({model_id})...")
    
    analysis = summarize_with_gemini(transcript, model_id)
    
    if analysis:
        print("\n" + "="*50)
        print(f"GAME SUMMARY: {analysis.title}")
        print("="*50)
        print(f"\n{analysis.narrative_summary}\n")
        
        print(f"Winner: {analysis.winner_team}")
        print(f"MVP: {analysis.mvp_player} - {analysis.mvp_reasoning}")
        print(f"Best Play: {analysis.best_play}")
        print(f"Biggest Mistake: {analysis.biggest_mistake}")
        print("\n" + "-"*30)
        print("PLAYER HIGHLIGHTS")
        print("-"*30)
        
        for player in analysis.player_highlights:
            print(f"\nPlayer: {player.player_name} ({player.role}) - Rating: {player.rating}/10")
            print(f"Summary: {player.summary}")
            print(f"Key Move: {player.key_move}")
            
    else:
        print("Failed to generate analysis.")

if __name__ == "__main__":
    main()
