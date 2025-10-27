import argparse
import hashlib
import http.server
import json
import logging
import os
import socketserver
import wave

import yaml
from dotenv import load_dotenv
from google import genai
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech
from google.genai import types

from kaggle_environments.envs.werewolf.game.consts import EventName
from kaggle_environments.envs.werewolf.runner import setup_logger

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Loads the configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Saves PCM audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def get_tts_audio_genai(client, text: str, voice_name: str) -> bytes | None:
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
            ),
        )
        return response.candidates[0].content.parts[0].inline_data.data
    except (GoogleAPICallError, ValueError) as e:
        logger.error(f"  - Error generating audio for '{text[:30]}...': {e}")
        return None


def get_tts_audio_vertex(
    client, text: str, voice_name: str, model_name: str = "gemini-2.5-flash-preview-tts"
) -> bytes | None:
    """Fetches TTS audio from Vertex AI API."""
    if not text or not client:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name, model_name=model_name)

        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, sample_rate_hertz=24000)

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return response.audio_content
    except (GoogleAPICallError, ValueError) as e:
        logger.error(f"  - Error generating audio using Vertex AI for '{text[:30]}...': {e}")
        return None


def extract_game_data_from_json(replay_json):
    """Extracts dialogue and events from a replay JSON object."""
    logger.info("Extracting game data from replay...")
    unique_speaker_messages = set()
    dynamic_moderator_messages = set()
    moderator_log_steps = replay_json.get("info", {}).get("MODERATOR_OBSERVATION", [])

    for step_log in moderator_log_steps:
        for data_entry in step_log:
            # We must read from 'json_str' to match the werewolf.js renderer
            json_str = data_entry.get("json_str")
            data_type = data_entry.get("data_type")  # We still need this for filtering

            try:
                # Parse the event data from the json_str, just like the JS does
                event = json.loads(json_str)
                data = event.get("data", {})  # Get the data payload from inside the parsed event
                event_name = event.get("event_name")
                description = event.get("description", "")
                day_count = event.get("day")

            except json.JSONDecodeError as e:
                logger.warning(f"  - Skipping log entry, failed to parse json_str: {e}")
                continue

            # This logic below remains the same, but it now correctly uses
            # the 'data' payload from the parsed 'json_str'.
            if data_type == "ChatDataEntry":
                if data.get("actor_id") and data.get("message"):
                    unique_speaker_messages.add((data["actor_id"], data["message"]))
            elif data_type == "DayExileVoteDataEntry":
                if data.get("actor_id") and data.get("target_id"):
                    dynamic_moderator_messages.add(f"{data['actor_id']} votes to exile {data['target_id']}.")
            elif data_type == "WerewolfNightVoteDataEntry":
                if data.get("actor_id") and data.get("target_id"):
                    dynamic_moderator_messages.add(f"{data['actor_id']} votes to eliminate {data['target_id']}.")
            elif data_type == "SeerInspectActionDataEntry":
                if data.get("actor_id") and data.get("target_id"):
                    dynamic_moderator_messages.add(f"{data['actor_id']} inspects {data['target_id']}.")
            elif data_type == "DoctorHealActionDataEntry":
                if data.get("actor_id") and data.get("target_id"):
                    dynamic_moderator_messages.add(f"{data['actor_id']} heals {data['target_id']}.")
            elif data_type == "DayExileElectedDataEntry":
                if all(k in data for k in ["elected_player_id", "elected_player_role_name"]):
                    dynamic_moderator_messages.add(
                        f"{data['elected_player_id']} was exiled by vote. Their role was a {data['elected_player_role_name']}."
                    )
            elif data_type == "WerewolfNightEliminationDataEntry":
                if all(k in data for k in ["eliminated_player_id", "eliminated_player_role_name"]):
                    dynamic_moderator_messages.add(
                        f"{data['eliminated_player_id']} was eliminated. Their role was a {data['eliminated_player_role_name']}."
                    )
            elif data_type == "DoctorSaveDataEntry":
                if "saved_player_id" in data:
                    dynamic_moderator_messages.add(f"{data['saved_player_id']} was attacked but saved by a Doctor!")
            elif data_type == "SeerInspectResultDataEntry":
                if data.get("role"):
                    dynamic_moderator_messages.add(
                        f"{data['actor_id']} saw {data['target_id']}'s role is {data['role']}."
                    )
                elif data.get("team"):
                    dynamic_moderator_messages.add(
                        f"{data['actor_id']} saw {data['target_id']}'s team is {data['team']}."
                    )
            elif data_type == "GameEndResultsDataEntry":
                if "winner_team" in data:
                    dynamic_moderator_messages.add(f"The game is over. The {data['winner_team']} team has won!")
            elif data_type == "WerewolfNightEliminationElectedDataEntry":
                if "elected_target_player_id" in data:
                    dynamic_moderator_messages.add(
                        f"The werewolves have chosen to eliminate {data['elected_target_player_id']}."
                    )
            elif event_name == EventName.DAY_START:
                dynamic_moderator_messages.add(f"Day {day_count} begins!")
            elif event_name == EventName.NIGHT_START:
                dynamic_moderator_messages.add(f"Night {day_count} begins!")
            elif event_name == EventName.MODERATOR_ANNOUNCEMENT:
                if "discussion rule is" in description:
                    dynamic_moderator_messages.add("Discussion begins!")
                elif "Voting phase begins" in description:
                    dynamic_moderator_messages.add("Exile voting begins!")

    logger.info(f"Found {len(unique_speaker_messages)} unique player messages.")
    logger.info(f"Found {len(dynamic_moderator_messages)} dynamic moderator messages.")
    return unique_speaker_messages, dynamic_moderator_messages


def generate_audio_files(
    client,
    tts_provider,
    unique_speaker_messages,
    dynamic_moderator_messages,
    player_voice_map,
    audio_config,
    output_dir,
):
    """Generates and saves all required audio files, returning a map for the HTML."""
    logger.info("Extracting dialogue and generating audio files...")
    audio_map = {}
    paths = audio_config["paths"]
    audio_dir = os.path.join(output_dir, paths["audio_dir_name"])
    moderator_voice = audio_config["voices"]["moderator"]
    static_moderator_messages = audio_config["audio"]["static_moderator_messages"]

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
            logger.warning(f"  - Warning: No voice found for speaker: {speaker_id}")

    for speaker, key, message, voice in messages_to_generate:
        map_key = f"{speaker}:{key}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path_on_disk = os.path.join(audio_dir, filename)
        audio_path_for_html = os.path.join(paths["audio_dir_name"], filename)

        if not os.path.exists(audio_path_on_disk):
            logger.info(f'  - Generating audio for {speaker} ({voice}): "{message[:40]}..." ')
            audio_content = None
            if tts_provider == "vertex_ai":
                model_name = audio_config.get("vertex_ai_model", "gemini-2.5-flash-preview-tts")
                audio_content = get_tts_audio_vertex(client, message, voice_name=voice, model_name=model_name)
            else:  # google_genai
                audio_content = get_tts_audio_genai(client, message, voice_name=voice)

            if audio_content:
                wave_file(audio_path_on_disk, audio_content)
                audio_map[map_key] = audio_path_for_html
        else:
            audio_map[map_key] = audio_path_for_html

    return audio_map


def generate_debug_audio_files(
    output_dir, client, tts_provider, unique_speaker_messages, dynamic_moderator_messages, audio_config
):
    """Generates a single debug audio file and maps all events to it."""
    logger.info("Generating single debug audio for UI testing...")
    paths = audio_config["paths"]
    debug_audio_dir = os.path.join(output_dir, paths["debug_audio_dir_name"])
    os.makedirs(debug_audio_dir, exist_ok=True)
    audio_map = {}

    debug_message = "Testing start, testing end."
    filename = "debug_audio.wav"
    audio_path_on_disk = os.path.join(debug_audio_dir, filename)
    audio_path_for_html = os.path.join(paths["debug_audio_dir_name"], filename)

    if not os.path.exists(audio_path_on_disk):
        logger.info(f'  - Generating debug audio: "{debug_message}"')
        audio_content = None
        if tts_provider == "vertex_ai":
            model_name = audio_config.get("vertex_ai_model", "gemini-2.5-flash-preview-tts")
            debug_voice = "Charon"
            audio_content = get_tts_audio_vertex(client, debug_message, voice_name=debug_voice, model_name=model_name)
        else:
            debug_voice = "achird"
            audio_content = get_tts_audio_genai(client, debug_message, voice_name=debug_voice)

        if audio_content:
            wave_file(audio_path_on_disk, audio_content)
        else:
            logger.error("  - Failed to generate debug audio. The map will be empty.")
            return {}
    else:
        logger.info(f"  - Using existing debug audio file: {audio_path_on_disk}")

    static_moderator_messages = audio_config["audio"]["static_moderator_messages"]

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

    logger.info(f"  - Mapped all {len(audio_map)} audio events to '{audio_path_for_html}'")
    return audio_map


def render_html(existing_html_path, audio_map, output_file):
    """Reads an existing HTML replay, injects the audio map, and saves it."""
    logger.info(f"Reading existing HTML from: {existing_html_path}")
    with open(existing_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    logger.info("Injecting the local audio map into the HTML...")
    audio_map_json = json.dumps(audio_map)
    injection_script = f"<script>window.AUDIO_MAP = {audio_map_json};</script>"
    html_content = html_content.replace("</head>", f"{injection_script}</head>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Successfully generated audio-enabled HTML at: {output_file}")


def start_server(directory, port, filename):
    """Starts a local HTTP server to serve the replay."""
    logger.info(f"\nStarting local server to serve from the '{directory}' directory...")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"\nServing replay at: http://localhost:{port}/{filename}")
        print("Open this URL in your web browser.")
        print(f"Or you can zip the '{directory}' directory and share it.")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main():
    """Main function to add audio to a Werewolf replay."""
    parser = argparse.ArgumentParser(description="Add audio to a Werewolf game replay.")
    parser.add_argument(
        "-i", "--run_dir", type=str, required=True, help="Path to the directory of a game run generated by run.py."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for the audio-enabled replay. Defaults to 'werewolf_replay_audio' inside the run directory.",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs/audio/standard.yaml"),
        help="Path to the audio configuration YAML file.",
    )
    parser.add_argument(
        "--debug-audio", action="store_true", help="Generate a single debug audio file for all events for UI testing."
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start a local HTTP server to view the replay after generation."
    )
    parser.add_argument(
        "--tts-provider",
        type=str,
        default="vertex_ai",
        choices=["vertex_ai", "google_genai"],
        help="The TTS provider to use for audio synthesis.",
    )
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.run_dir, "werewolf_replay_audio")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(output_dir=args.output_dir, base_name="add_audio")

    logger.info(f"Loading audio config from: {args.config_path}")
    audio_config = load_config(args.config_path)

    replay_json_path = os.path.join(args.run_dir, "werewolf_game.json")
    logger.info(f"Loading game replay from: {replay_json_path}")
    if not os.path.exists(replay_json_path):
        logger.error(f"Replay file not found: {replay_json_path}")
        logger.error("Please ensure you provide a valid run directory created by run.py.")
        return
    with open(replay_json_path, "r") as f:
        replay_data = json.load(f)

    game_config = replay_data["configuration"]
    player_voices = audio_config["voices"]["players"]
    player_voice_map = {
        agent_config["id"]: player_voices.get(agent_config["id"]) for agent_config in game_config["agents"]
    }

    load_dotenv()
    client = None
    if args.tts_provider == "vertex_ai":
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            logger.error("Error: GOOGLE_CLOUD_PROJECT environment variable not found. It is required for Vertex AI.")
            return
        try:
            client = texttospeech.TextToSpeechClient()
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client: {e}")
            logger.error("Please ensure you have authenticated with 'gcloud auth application-default login'")
            return
    else:  # google_genai
        if not os.getenv("GEMINI_API_KEY"):
            logger.error(
                "Error: GEMINI_API_KEY environment variable not found. Audio generation with google.genai requires it."
            )
            return
        client = genai.Client()

    unique_speaker_messages, dynamic_moderator_messages = extract_game_data_from_json(replay_data)

    paths = audio_config["paths"]
    audio_dir = os.path.join(args.output_dir, paths["audio_dir_name"])
    os.makedirs(audio_dir, exist_ok=True)

    if args.debug_audio:
        audio_map = generate_debug_audio_files(
            args.output_dir,
            client,
            args.tts_provider,
            unique_speaker_messages,
            dynamic_moderator_messages,
            audio_config,
        )
    else:
        audio_map = generate_audio_files(
            client,
            args.tts_provider,
            unique_speaker_messages,
            dynamic_moderator_messages,
            player_voice_map,
            audio_config,
            args.output_dir,
        )

    original_html_path = os.path.join(args.run_dir, "werewolf_game.html")
    output_html_file = os.path.join(args.output_dir, paths["output_html_filename"])
    render_html(original_html_path, audio_map, output_html_file)

    if args.serve:
        start_server(args.output_dir, audio_config["server"]["port"], paths["output_html_filename"])


if __name__ == "__main__":
    main()
