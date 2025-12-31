import argparse
import hashlib
import json
import logging
import os
import subprocess
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


def load_prompt(prompt_path):
    """Loads the system instruction prompt from a file."""
    if not os.path.exists(prompt_path):
        logger.warning(f"Prompt file not found: {prompt_path}. Using default.")
        return "Rewrite the following text to be theatrical and expressive for TTS: '{text}'"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_cache(cache_path):
    """Loads the LLM rewrite cache."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode cache file: {cache_path}. Starting fresh.")
    return {}

def save_cache(cache, cache_path):
    """Saves the LLM rewrite cache."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")

def enhance_audio_text(client, text, speaker, prompt_template, cache, cache_key):
    """Enhances text using LLM, checking cache first."""
    if cache_key in cache:
        return cache[cache_key]

    # Construct the prompt
    prompt = prompt_template.format(text=text, speaker=speaker)
    
    enhanced_text = text # Fallback
    try:
        # We need a unified way to call the LLM regardless of provider if possible, 
        # but the script uses different clients for TTS. 
        # For LLM text generation, we likely want to use google-genai or vertex AI generative models.
        # The existing code initializes `client` as either `texttospeech.TextToSpeechClient` (Vertex TTS) 
        # OR `genai.Client` (Gemini).
        # Vertex TTS client CANNOT do text generation. We need a separate GenAI client for Vertex if we are in Vertex mode.
        
        # NOTE: For now, I will assume we can allow a separate optional GenAI client or use the existing one if it is GenAI.
        # If the user selected 'vertex_ai' for TTS, we might not have a GenAI client ready.
        # I'll handle this in the main loop or here.
        
        pass # Placeholder for logic below
    except Exception as e:
        logger.warning(f"LLM enhancement failed for '{text[:20]}...': {e}")
        return text

    return text # Placeholder

# Real implementation helper
def call_llm_generation(client, provider, model_name, prompt):
    try:
        if provider == "google_genai":
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        elif provider == "vertex_ai":
            # We need to construct a Vertex AI GenerativeModel. 
            # This requires 'vertexai' library or using the genai client with vertex backend? 
            # The current script uses `google.genai` for the 'google_genai' provider.
            # For Vertex, typically one uses `vertexai.init` and `GenerativeModel`.
            # Let's try to use `google.genai` if available, or skip if we only have TTS client.
            logger.warning("LLM enhancement with Vertex AI client not fully implemented in this snippet without a separate GenAI client.")
            return None
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return None
    return None

def generate_audio_files(
    client,
    tts_provider,
    unique_speaker_messages,
    dynamic_moderator_messages,
    player_voice_map,
    audio_config,
    output_dir,
    llm_client=None, # New arg
    prompt_template=None,
    cache=None,
    cache_path=None
):
    """Generates and saves all required audio files, returning a map for the HTML."""
    logger.info("Extracting dialogue, enhancing text, and generating audio files...")
    audio_map = {}
    paths = audio_config["paths"]
    audio_dir = os.path.join(output_dir, paths["audio_dir_name"])
    moderator_voice = audio_config["voices"]["moderator"]
    static_moderator_messages = audio_config["audio"]["static_moderator_messages"]

    messages_to_generate = []
    
    # helper to prepare batch
    def add_msg(speaker, key, text, voice):
        messages_to_generate.append({"speaker": speaker, "key": key, "text": text, "voice": voice})

    # Map internal config keys to the exact text strings the JS AudioController expects
    key_aliases = {
        "discussion_begins": "Discussion begins!",
        "voting_begins": "Exile voting begins!"
    }

    for key, message in static_moderator_messages.items():
        add_msg("moderator", key, message, moderator_voice)
        # Also add alias if exists, pointing to SAME text/voice (so same audio hash)
        if key in key_aliases:
            add_msg("moderator", key_aliases[key], message, moderator_voice)
            
    for message in dynamic_moderator_messages:
        add_msg("moderator", message, message, moderator_voice)
    for speaker_id, message in unique_speaker_messages:
        voice = player_voice_map.get(speaker_id)
        if voice:
            add_msg(speaker_id, message, message, voice)
        else:
            logger.warning(f"  - Warning: No voice found for speaker: {speaker_id}")

    # Process messages (Enhance -> TTS)
    cache_updated = False
    
    for msg in messages_to_generate:
        speaker = msg["speaker"]
        original_text = msg["text"]
        voice = msg["voice"]
        key = msg.get("key", original_text)
        
        # 1. Enhance Text
        final_text = original_text
        if llm_client and prompt_template and cache is not None:
             # Cache key based on original text and speaker to avoid re-generating same line
             cache_key = f"{speaker}:{original_text}"
             
             if cache_key in cache:
                 final_text = cache[cache_key]
             else:
                 logger.info(f"  - Enhancing text for {speaker}...")
                 prompt = prompt_template.format(text=original_text, speaker=speaker)
                 enhanced = call_llm_generation(llm_client, "google_genai", "gemini-2.0-flash-exp", prompt) # Hardcoded model for now or config
                 if enhanced:
                     final_text = enhanced
                     cache[cache_key] = final_text
                     cache_updated = True
                     # Save periodically or at end? At end is safer for perf, but risk data loss.
                     if cache_path: save_cache(cache, cache_path) 
                 else:
                     logger.warning("  - LLM enhancement returned None, using original.")

        # 2. Generate Audio
        # Map key uses original text/key to ensure stable IDs even if text changes
        map_key = f"{speaker}:{key}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path_on_disk = os.path.join(audio_dir, filename)
        audio_path_for_html = os.path.join(paths["audio_dir_name"], filename)

        if not os.path.exists(audio_path_on_disk):
            # logger.info(f'  - Generating audio for {speaker} ({voice}): "{final_text[:40]}..." ')
            # Only log if we are actually doing work
            print(f'Generating audio: "{final_text[:60]}..."')
            
            audio_content = None
            if tts_provider == "vertex_ai":
                model_name = audio_config.get("vertex_ai_model", "gemini-2.5-flash-preview-tts")
                audio_content = get_tts_audio_vertex(client, final_text, voice_name=voice, model_name=model_name)
            else:  # google_genai
                audio_content = get_tts_audio_genai(client, final_text, voice_name=voice)

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


def save_audio_map(audio_map, output_dir):
    """Saves the audio map to a JSON file."""
    output_file = os.path.join(output_dir, "audio_map.json")
    logger.info(f"Saving audio map to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(audio_map, f, indent=2)


def start_server(visualizer_dir, replay_path, audio_map_path):
    """Starts the Vite dev server with the custom replay and audio map."""
    visualizer_dir = os.path.abspath(visualizer_dir)
    replay_path = os.path.abspath(replay_path)
    audio_map_path = os.path.abspath(audio_map_path)

    logger.info(f"\nStarting Vite server from '{visualizer_dir}'...")
    logger.info(f"Replay File: {replay_path}")
    logger.info(f"Audio Map File: {audio_map_path}")

    # Attempt to calculate relative paths for cleaner Vite usage
    try:
        rel_replay = os.path.relpath(replay_path, visualizer_dir)
        rel_audio_map = os.path.relpath(audio_map_path, visualizer_dir)
    except ValueError:
        # Fallback to absolute if on different drives (mostly Windows issue)
        rel_replay = replay_path
        rel_audio_map = audio_map_path

    env = os.environ.copy()

    # Construct the command
    # We use npx to ensure we use the local project dependencies
    cmd = [
        "npx", "cross-env",
        f"VITE_REPLAY_FILE={rel_replay}",
        f"VITE_AUDIO_MAP_FILE={rel_audio_map}",
        "vite"
    ]

    print("\nRunning command:", " ".join(cmd))
    print("Press Ctrl+C to stop the server.")

    try:
        subprocess.run(cmd, cwd=visualizer_dir, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    """Main function to add audio to a Werewolf replay."""
    parser = argparse.ArgumentParser(description="Add audio to a Werewolf game replay.")
    parser.add_argument(
        "-i", "--input_path", type=str, required=True, help="Path to the replay JSON file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory for the audio-enabled replay. Defaults to 'werewolf_replay_audio' in the current directory.",
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
        "--serve", action="store_true", help="Start a local HTTP server (Vite) to view the replay after generation."
    )
    parser.add_argument(
        "--tts-provider",
        type=str,
        default="vertex_ai",
        choices=["vertex_ai", "google_genai"],
        help="The TTS provider to use for audio synthesis.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs/audio/theatrical_prompt.txt"),
        help="Path to the system instruction prompt file for LLM enhancement.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to the LLM rewrite cache JSON file. Defaults to 'llm_cache.json' in the output directory.",
    )
    parser.add_argument(
        "--disable_llm_enhancement",
        action="store_true",
        help="Disable LLM-based text enhancement (theatrical rewriting).",
    )
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = "werewolf_replay_audio"
        
    if not args.cache_path:
        args.cache_path = os.path.join(args.output_dir, "llm_cache.json")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(output_dir=args.output_dir, base_name="add_audio")

    logger.info(f"Loading audio config from: {args.config_path}")
    audio_config = load_config(args.config_path)

    logger.info(f"Loading game replay from: {args.input_path}")
    if not os.path.exists(args.input_path):
        logger.error(f"Replay file not found: {args.input_path}")
        return
    with open(args.input_path, "r") as f:
        replay_data = json.load(f)

    game_config = replay_data["configuration"]
    player_voices = audio_config["voices"]["players"]
    player_voice_map = {
        agent_config["id"]: player_voices.get(agent_config["id"]) for agent_config in game_config["agents"]
    }

    # Load .env from REPO ROOT (PROJECT_ROOT is kaggle_environments package dir, so go one up)
    from kaggle_environments import PROJECT_ROOT
    env_path = os.path.join(PROJECT_ROOT, os.pardir, ".env")
    if os.path.exists(env_path):
        logger.info(f"Loading .env from: {env_path}")
        load_dotenv(env_path)
    else:
        logger.info(f".env not found at {env_path}, relying on system environment variables.")

    client = None
    llm_client = None
    
    # Initialize Clients
    
    # Only init LLM client if enhancement is NOT disabled
    if not args.disable_llm_enhancement:
        if os.getenv("GEMINI_API_KEY"):
             try:
                 llm_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
             except Exception as e:
                 logger.warning(f"Failed to init GenAI client for LLM enhancement: {e}")

    if args.tts_provider == "vertex_ai":
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            raise RuntimeError("Error: GOOGLE_CLOUD_PROJECT environment variable not found. It is required for Vertex AI.")
        try:
            client = texttospeech.TextToSpeechClient()
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client: {e}")
            logger.error("Please ensure you have authenticated with 'gcloud auth application-default login'")
    else:  # google_genai
        if not os.getenv("GEMINI_API_KEY"):
             raise RuntimeError(
                "Error: GEMINI_API_KEY environment variable not found. Audio generation with google.genai requires it."
            )
        # Verify client reuse or create new? 
        if llm_client:
            client = llm_client
        else:
            client = genai.Client()
            # If we created a client here but enhancement is disabled, we don't set llm_client
            if not args.disable_llm_enhancement:
                llm_client = client

    unique_speaker_messages, dynamic_moderator_messages = extract_game_data_from_json(replay_data)

    paths = audio_config["paths"]
    audio_dir = os.path.join(args.output_dir, paths["audio_dir_name"])
    os.makedirs(audio_dir, exist_ok=True)

    if args.debug_audio:
        # Debug audio doesn't use LLM enhancement in this refactor yet, or we could add it.
        # But for brevity, I'll leave it simple.
        audio_map = generate_debug_audio_files(
            args.output_dir,
            client,
            args.tts_provider,
            unique_speaker_messages,
            dynamic_moderator_messages,
            audio_config,
        )
    else:
        # Load resources for enhancement
        prompt_template = load_prompt(args.prompt_path)
        cache = load_cache(args.cache_path)
        
        # Decide if we pass llm_client based on flag
        active_llm_client = llm_client if not args.disable_llm_enhancement else None

        audio_map = generate_audio_files(
            client,
            args.tts_provider,
            unique_speaker_messages,
            dynamic_moderator_messages,
            player_voice_map,
            audio_config,
            args.output_dir,
            llm_client=active_llm_client,
            prompt_template=prompt_template,
            cache=cache,
            cache_path=args.cache_path
        )

    save_audio_map(audio_map, args.output_dir)

    if args.serve:
        visualizer_dir = os.path.join(os.path.dirname(__file__), "../visualizer/default")
        audio_map_path = os.path.join(args.output_dir, "audio_map.json")
        start_server(visualizer_dir, args.input_path, audio_map_path)


if __name__ == "__main__":
    main()
