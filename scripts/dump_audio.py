import argparse
import base64
import hashlib
import json
import os
import random
import http.server
import shutil
import socketserver
import wave
from kaggle_environments import make
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Configuration ---
PORT = 8000
OUTPUT_DIR = "werewolf_replay"
AUDIO_DIR_NAME = "audio"
OUTPUT_HTML_FILENAME = "replay.html"
MODERATOR_VOICE = "enceladus"

# --- Global Paths ---
AUDIO_DIR = os.path.join(OUTPUT_DIR, AUDIO_DIR_NAME)
OUTPUT_HTML_FILE = os.path.join(OUTPUT_DIR, OUTPUT_HTML_FILENAME)


# --- Helper Functions ---
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Saves PCM audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def get_tts_audio(client, text: str, voice_name: str) -> bytes | None:
    """Fetches TTS audio from Gemini API."""
    if not text or not client:
        return None
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
            )
        )
        return response.candidates[0].content.parts[0].inline_data.data
    except Exception as e:
        print(f"  - Error generating audio for '{text[:30]}...': {e}")
        return None


def copy_assets(output_dir):
    """Copies the assets directory for 3D model rendering."""
    source_assets_dir = "assets"
    dest_assets_dir = os.path.join(output_dir, "assets")
    if os.path.exists(source_assets_dir):
        if os.path.exists(dest_assets_dir):
            shutil.rmtree(dest_assets_dir)
        shutil.copytree(source_assets_dir, dest_assets_dir)
        print(f"Copied '{source_assets_dir}' to '{dest_assets_dir}' for 3D rendering.")


def setup_environment():
    """Sets up the Werewolf game environment and agent configurations."""
    print("1. Setting up Werewolf environment with random agents...")
    URLS = {
        "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
        "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
        "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
        "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png",
        "deepseek": "https://images.seeklogo.com/logo-png/61/1/deepseek-ai-icon-logo-png_seeklogo-611473.png",
        "kimi": "https://images.seeklogo.com/logo-png/61/1/kimi-logo-png_seeklogo-611650.png",
        "qwen": "https://images.seeklogo.com/logo-png/61/1/qwen-icon-logo-png_seeklogo-611724.png"
    }
    parameter_dict = {
        "together_ai/deepseek-ai/DeepSeek-R1": {"max_tokens": 163839},
        "claude-4-sonnet-20250514": {"max_tokens": 64000}
    }
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
    random.shuffle(roles)
    names = ["gemini-2.5-flash", "deepseek-r1", "kimi-k2", "qwen3", "gpt-4.1", "o4-mini", "gemini-2.5-pro", "grok-4"]
    models = [
        "gemini/gemini-2.5-flash", "together_ai/deepseek-ai/DeepSeek-R1",
        "together_ai/moonshotai/Kimi-K2-Instruct", "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "gpt-4.1", "o4-mini", "gemini/gemini-2.5-pro", "xai/grok-4-latest",
    ]
    brands = ['gemini', 'deepseek', 'kimi', 'qwen', 'openai', 'openai', 'gemini', 'grok']
    voices = ['Kore', 'Charon', 'Leda', 'Despina', 'Erinome', 'Gacrux', 'Achird', 'Puck']

    agents_config = [
        {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
         "thumbnail": URLS[brand], "display_name": model,
         "agent_harness_name": "llm_harness",
         "llms": [{"model_name": model, "parameters": parameter_dict.get(model, {})}]}
        for role, name, brand, model in zip(roles, names, brands, models)
    ]
    for agent, voice in zip(agents_config, voices):
        agent['voice'] = voice

    player_voice_map = {agent["id"]: agent["voice"] for agent in agents_config}
    env = make(
        'werewolf', debug=True,
        configuration={
            "actTimeout": 300, "runTimeout": 3600, "agents": agents_config,
            "discussion_protocol": {"name": "RoundRobinDiscussion", "params": {"max_rounds": 1}}
        }
    )
    return env, player_voice_map


def run_game(env):
    """Runs a full game episode."""
    print("2. Running a full game episode...")
    agents = ['random'] * 8
    env.run(agents)


def extract_game_data(env):
    """Extracts dialogue and events from the game log."""
    unique_speaker_messages = set()
    dynamic_moderator_messages = set()
    moderator_log_steps = env.info.get("MODERATOR_OBSERVATION", [])

    for step_log in moderator_log_steps:
        for data_entry in step_log:
            json_str = data_entry.get('json_str')
            if not json_str: continue
            try:
                history_event = json.loads(json_str)
                data = history_event.get('data', {})
                data_type = data_entry.get("data_type")

                if data_type == "ChatDataEntry":
                    if data.get("actor_id") and data.get("message"):
                        unique_speaker_messages.add((data["actor_id"], data["message"]))
                elif data_type == "DayExileElectedDataEntry":
                    dynamic_moderator_messages.add(
                        f"Player {data['elected_player_id']} was exiled by vote. Their role was a {data['elected_player_role_name']}.")
                elif data_type == "WerewolfNightEliminationDataEntry":
                    dynamic_moderator_messages.add(
                        f"Player {data['eliminated_player_id']} was eliminated. Their role was a {data['eliminated_player_role_name']}.")
                elif data_type == "DoctorSaveDataEntry":
                    dynamic_moderator_messages.add(
                        f"Player {data['saved_player_id']} was attacked but saved by a Doctor!")
                elif data_type == "GameEndResultsDataEntry":
                    dynamic_moderator_messages.add(f"The game is over. The {data['winner_team']} team has won!")
                elif data_type == "WerewolfNightEliminationElectedDataEntry":
                    dynamic_moderator_messages.add(
                        f"The werewolves have chosen to eliminate player {data['elected_target_player_id']}.")
            except json.JSONDecodeError:
                print(f"  - Warning: Could not decode JSON: {json_str}")

    return unique_speaker_messages, dynamic_moderator_messages


def generate_audio_files(client, unique_speaker_messages, dynamic_moderator_messages, player_voice_map):
    """Generates and saves all required audio files, returning a map for the HTML."""
    print("3. Extracting dialogue and generating audio files...")
    audio_map = {}

    static_moderator_messages = {
        "night_begins": "(rate=\"fast\", volume=\"soft\", voice=\"mysterious\")[As darkness descends, the village falls silent.](rate=\"medium\", pitch=\"-2st\")[Everyone, close your eyes.]",
        "day_begins": "(rate=\"fast\", volume=\"loud\")[Wake up, villagers!] (rate=\"medium\", voice=\"neutral\")[The sun rises on a new day.] (break=\"50ms\") (rate=\"medium\", voice=\"somber\")[Let's see who survived the night.]",
        "discussion_begins": "(voice=\"authoritative\")[The town meeting now begins.] (voice=\"neutral\")[You have a few minutes to discuss and find the werewolves among you.] (voice=\"authoritative\")[Begin.]",
        "voting_begins": "(rate=\"slow\", voice=\"serious\")[The time for talk is over.] (break=\"50ms\") (rate=\"medium\", volume=\"loud\", voice=\"dramatic\")[Now, you must cast your votes!]",
    }

    messages_to_generate = []
    # Queue static moderator messages
    for key, message in static_moderator_messages.items():
        messages_to_generate.append(("moderator", key, message, MODERATOR_VOICE))
    # Queue dynamic moderator messages
    for message in dynamic_moderator_messages:
        messages_to_generate.append(("moderator", message, message, MODERATOR_VOICE))
    # Queue player messages
    for speaker_id, message in unique_speaker_messages:
        voice = player_voice_map.get(speaker_id)
        if voice:
            messages_to_generate.append((speaker_id, message, message, voice))
        else:
            print(f"  - Warning: No voice found for speaker: {speaker_id}")

    for speaker, key, message, voice in messages_to_generate:
        map_key = f"{speaker}:{key}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path_on_disk = os.path.join(AUDIO_DIR, filename)
        audio_path_for_html = os.path.join(AUDIO_DIR_NAME, filename)

        if not os.path.exists(audio_path_on_disk):
            print(f"  - Generating audio for {speaker} ({voice}): \"{message[:40]}...\" ")
            audio_content = get_tts_audio(client, message, voice_name=voice)
            if audio_content:
                wave_file(audio_path_on_disk, audio_content)
                audio_map[map_key] = audio_path_for_html
        else:
            audio_map[map_key] = audio_path_for_html

    return audio_map


def render_html(env, audio_map, output_file):
    """Renders the game to HTML and injects the audio map."""
    print("4. Rendering the game to an HTML file...")
    html_content = env.render(mode="html")

    print("5. Injecting the local audio map into the HTML...")
    audio_map_json = json.dumps(audio_map)
    injection_script = f"<script>window.AUDIO_MAP = {audio_map_json};</script>"
    html_content = html_content.replace('</head>', f'{injection_script}</head>')

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)


def start_server(directory, port, filename):
    """Starts a local HTTP server to serve the replay."""
    print(f"\n6. Starting local server to serve from the '{directory}' directory...")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

    with socketserver.TCPServer(('', port), Handler) as httpd:
        print(f"\nServing replay at: http://localhost:{port}/{filename}")
        print("Open this URL in your web browser.")
        print(f"Or you can zip the '{directory}' directory and share it.")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main(generate_audio=True):
    """Main function to generate and serve the Werewolf replay."""
    # --- Initial Setup ---
    os.makedirs(AUDIO_DIR, exist_ok=True)
    copy_assets(OUTPUT_DIR)

    client = None
    if generate_audio:
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            print("Error: GEMINI_API_KEY not found. Audio generation requires it.")
            print("Run with --no-audio to generate a replay without sound.")
            return
        client = genai.Client()

    # --- Core Workflow ---
    env, player_voice_map = setup_environment()
    run_game(env)

    audio_map = {}
    if generate_audio:
        unique_speaker_messages, dynamic_moderator_messages = extract_game_data(env)
        audio_map = generate_audio_files(
            client, unique_speaker_messages, dynamic_moderator_messages, player_voice_map
        )
    else:
        print("3. Skipping audio generation.")

    render_html(env, audio_map, OUTPUT_HTML_FILE)
    start_server(OUTPUT_DIR, PORT, OUTPUT_HTML_FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a Werewolf game replay.")
    parser.add_argument('--no-audio', action='store_true', help="Disable audio generation.")
    args = parser.parse_args()
    main(generate_audio=not args.no_audio)
