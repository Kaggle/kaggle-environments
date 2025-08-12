import argparse
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import wave

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

from kaggle_environments.envs.werewolf.runner import run_werewolf, setup_logger, shuffle_roles_inplace


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def setup_environment(game_config, script_config):
    """Sets up the Werewolf game environment and agent configurations."""
    print("1. Setting up Werewolf environment...")

    player_voices = script_config['voices']['players']

    for agent_config in game_config['agents']:
        agent_id = agent_config['id']
        agent_config['voice'] = player_voices.get(agent_id)
    player_voice_map = {agent["id"]: agent.get("voice") for agent in game_config['agents']}
    return player_voice_map


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


def generate_audio_files(client, unique_speaker_messages, dynamic_moderator_messages, player_voice_map, script_config):
    """Generates and saves all required audio files, returning a map for the HTML."""
    print("3. Extracting dialogue and generating audio files...")
    audio_map = {}
    paths = script_config['paths']
    audio_dir = os.path.join(paths['output_dir'], paths['audio_dir_name'])
    moderator_voice = script_config['voices']['moderator']
    static_moderator_messages = script_config['audio']['static_moderator_messages']

    messages_to_generate = []
    for key, message in static_moderator_messages.items():
        messages_to_generate.append(("moderator", key, message, moderator_voice))
    for message in dynamic_moderator_messages:
        messages_to_generate.append(("moderator", message, message, moderator_voice))
    for speaker_id, message in unique_speaker_messages:
        voice = player_voice_map.get(speaker_id)
        if voice:
            messages_to_generate.append((speaker_id, message, message, voice))
        else:
            print(f"  - Warning: No voice found for speaker: {speaker_id}")

    for speaker, key, message, voice in messages_to_generate:
        map_key = f"{speaker}:{key}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path_on_disk = os.path.join(audio_dir, filename)
        audio_path_for_html = os.path.join(paths['audio_dir_name'], filename)

        if not os.path.exists(audio_path_on_disk):
            print(f"  - Generating audio for {speaker} ({voice}): \"{message[:40]}...\" ")
            audio_content = get_tts_audio(client, message, voice_name=voice)
            if audio_content:
                wave_file(audio_path_on_disk, audio_content)
                audio_map[map_key] = audio_path_for_html
        else:
            audio_map[map_key] = audio_path_for_html

    return audio_map


def generate_debug_audio_files(output_dir, client, unique_speaker_messages, dynamic_moderator_messages, script_config):
    """Generates a single debug audio file and maps all events to it."""
    print("3. Generating single debug audio for UI testing...")
    paths = script_config['paths']
    debug_audio_dir = os.path.join(output_dir, paths['debug_audio_dir_name'])
    os.makedirs(debug_audio_dir, exist_ok=True)
    audio_map = {}

    debug_message = "Testing start, testing end."
    debug_voice = "achird"

    filename = "debug_audio.wav"
    audio_path_on_disk = os.path.join(debug_audio_dir, filename)
    audio_path_for_html = os.path.join(paths['debug_audio_dir_name'], filename)

    if not os.path.exists(audio_path_on_disk):
        print(f"  - Generating debug audio: \"{debug_message}\"")
        audio_content = get_tts_audio(client, debug_message, voice_name=debug_voice)
        if audio_content:
            wave_file(audio_path_on_disk, audio_content)
        else:
            print("  - Failed to generate debug audio. The map will be empty.")
            return {}
    else:
        print(f"  - Using existing debug audio file: {audio_path_on_disk}")

    static_moderator_messages = script_config['audio']['static_moderator_messages']

    messages_to_map = []
    for key in static_moderator_messages:
        messages_to_map.append(("moderator", key))
    for message in dynamic_moderator_messages:
        messages_to_map.append(("moderator", message))
    for speaker_id, message in unique_speaker_messages:
        messages_to_map.append((speaker_id, message))

    for speaker, key in messages_to_map:
        map_key = f"{speaker}:{key}"
        audio_map[map_key] = audio_path_for_html

    print(f"  - Mapped all {len(audio_map)} audio events to '{audio_path_for_html}'")
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


def main(output_dir, config, debug_audio, use_random_agents, shuffle_roles):
    """Main function to generate and serve the Werewolf replay."""
    script_config = config['script_settings']
    game_config = config['game_config']
    if shuffle_roles:
        shuffle_roles_inplace(game_config)

    paths = script_config['paths']
    audio_dir = os.path.join(output_dir, paths['audio_dir_name'])
    output_html_file = os.path.join(output_dir, paths['output_html_filename'])

    os.makedirs(audio_dir, exist_ok=True)
    copy_assets(output_dir)

    player_voice_map = setup_environment(game_config, script_config)

    agents = [agent['agent_id'] for agent in game_config['agents']]
    if use_random_agents:
        agents = ['random'] * len(agents)

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found. Audio generation requires it.")
        print("Run with --no-audio to generate a replay without sound.")
        return
    client = genai.Client()

    env = run_werewolf(output_dir=output_dir, base_name="replay", config=game_config, agents=agents, debug=True)

    unique_speaker_messages, dynamic_moderator_messages = extract_game_data(env)
    if debug_audio:
        audio_map = generate_debug_audio_files(output_dir, client, unique_speaker_messages, dynamic_moderator_messages,
                                               script_config)
    else:
        audio_map = generate_audio_files(
            client, unique_speaker_messages, dynamic_moderator_messages, player_voice_map, script_config
        )

    render_html(env, audio_map, output_html_file)
    start_server(output_dir, script_config['server']['port'], paths['output_html_filename'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a Werewolf game replay.")
    parser.add_argument("-o", "--output_dir", type=str, help="output directory", default="werewolf_replay")
    parser.add_argument("-n", "--base_name", type=str, help="the base file name of .html, .json and .log",
                        default="out")
    parser.add_argument("-c", '--config', type=str,
                        default='kaggle_environments/envs/werewolf/scripts/configs/standard.yaml',
                        help="Path to the configuration YAML file.")
    parser.add_argument('--debug-audio', action='store_true',
                        help="Generate a single debug audio file for UI testing.")
    parser.add_argument("-r", "--use_random_agents", action="store_true",
                        help='Use random agent for fast testing.')
    parser.add_argument('-s', "--shuffle_roles", action='store_true',
                        help="shuffle the roles of the agents defined in config.")

    args = parser.parse_args()

    setup_logger(output_dir=args.output_dir, base_name=args.base_name)

    config = load_config(args.config)
    main(output_dir=args.output_dir, config=config, debug_audio=args.debug_audio,
         use_random_agents=args.use_random_agents, shuffle_roles=args.shuffle_roles)
