import base64
import hashlib
import json
import os
import random
import http.server
import socketserver
import wave
from kaggle_environments import make
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Configuration ---
PORT = 8000
AUDIO_DIR = "audio"
OUTPUT_HTML_FILE = "game_replay_audio.html"
MODERATOR_VOICE = "enceladus"  # A distinct voice for the moderator

# --- Setup ---
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create the audio directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Main Script ---
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    client = genai.Client()

    # --- Helper Functions ---
    def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
       """Saves PCM audio data to a WAV file."""
       with wave.open(filename, "wb") as wf:
          wf.setnchannels(channels)
          wf.setsampwidth(sample_width)
          wf.setframerate(rate)
          wf.writeframes(pcm)

    def get_tts_audio(text: str, voice_name: str) -> bytes | None:
        """Fetches TTS audio from Gemini API and returns the audio content as bytes."""
        if not text:
            return None
        try:
            response = client.models.generate_content(
               model="gemini-2.5-flash-preview-tts",
               contents=text,
               config=types.GenerateContentConfig(
                  response_modalities=["AUDIO"],
                  speech_config=types.SpeechConfig(
                     voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                           voice_name=voice_name,
                        )
                     )
                  ),
               )
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            print(f"  - Error generating audio for '{text[:30]}...': {e}")
            return None

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
    # names = ["gemini-2.5-flash", "deepseek-r1", "kimi-k2", "qwen3", "gpt-4.1", "o4-mini", "claude-4-sonnet", "grok-4"]
    names = ["gemini-2.5-flash", "deepseek-r1", "kimi-k2", "qwen3", "gpt-4.1", "o4-mini", "gemini-2.5-pro", "grok-4"]

    # models = [
    #     "gemini/gemini-2.5-flash",
    #     "together_ai/deepseek-ai/DeepSeek-R1",
    #     "together_ai/moonshotai/Kimi-K2-Instruct",
    #     "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    #     "gpt-4.1",
    #     "o4-mini",
    #     "claude-4-sonnet-20250514",
    #     "xai/grok-4-latest",
    # ]
    models = [
        "gemini/gemini-2.5-flash",
        "together_ai/deepseek-ai/DeepSeek-R1",
        "together_ai/moonshotai/Kimi-K2-Instruct",
        "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "gpt-4.1",
        "o4-mini",
        "gemini/gemini-2.5-pro",
        "xai/grok-4-latest",
    ]

    # brands = ['gemini', 'deepseek', 'kimi', 'qwen', 'openai', 'openai', 'claude', 'grok']
    brands = ['gemini', 'deepseek', 'kimi', 'qwen', 'openai', 'openai', 'gemini', 'grok']

    voices = ['Kore', 'Charon', 'Leda', 'Despina', 'Erinome', 'Gacrux', 'Achird', 'Puck']
    # random.shuffle(voices)

    agents_config = [
        {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
         "thumbnail": URLS[brand], "display_name": model,
         "agent_harness_name": "llm_harness",
         "llms": [{"model_name": model, "parameters": parameter_dict.get(model, {})}]}
        for role, name, brand, model in zip(roles, names, brands, models)
    ]

    # Assign voices to agents
    for agent, voice in zip(agents_config, voices):
        agent['voice'] = voice

    player_voice_map = {agent["id"]: agent["voice"] for agent in agents_config}

    env = make(
        'werewolf',
        debug=True,
        configuration={
            "actTimeout": 300,
            "runTimeout": 3600,
            "agents": agents_config,
            "discussion_protocol": {
                "name": "RoundRobinDiscussion",
                "params": {
                    "max_rounds": 1
                }
            }
        }
    )

    print("2. Running a full game episode...")
    # agents = [f"llm/{model}" for model in models]
    agents = ['random'] * 8
    env.run(agents)

    print("3. Extracting dialogue and generating audio files...")
    unique_speaker_messages = set()
    dynamic_moderator_messages = set()
    audio_map = {}

    # Define static moderator messages
    static_moderator_messages = {
        "night_begins": '(rate="slow", volume="soft", voice="mysterious")[As darkness descends, the village falls silent.] (break="1s") (rate="slower", pitch="-2st")[Everyone, close your eyes.]',
        "day_begins": '(rate="fast", volume="loud")[Wake up, villagers!] (rate="medium", voice="neutral")[The sun rises on a new day.] (break="750ms") (rate="slow", voice="somber")[Let\'s see who survived the night.]',
        "discussion_begins": '(voice="authoritative")[The town meeting now begins.] (voice="neutral")[You have a few minutes to discuss and find the werewolves among you.] (voice="authoritative")[Begin.]',
        "voting_begins": '(rate="slow", voice="serious")[The time for talk is over.] (break="500ms") (rate="slower", volume="loud", voice="dramatic")[Now, you must cast your votes!]',
    }

    # Generate and cache static moderator audio
    for key, message in static_moderator_messages.items():
        map_key = f"moderator:{key}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"  - Generating moderator audio for: \"{message}\" ")
            audio_content = get_tts_audio(message, voice_name=MODERATOR_VOICE)
            if audio_content:
                wave_file(audio_path, audio_content)
                audio_map[map_key] = audio_path
        else:
            audio_map[map_key] = audio_path

    # Collect player chat and dynamic moderator announcements
    moderator_log_steps = env.info.get("MODERATOR_OBSERVATION", [])
    for step_log in moderator_log_steps:
        for data_entry in step_log:
            data_type = data_entry.get("data_type")
            json_str = data_entry.get('json_str')
            if not json_str:
                continue
            try:
                history_event = json.loads(json_str)
                data = history_event.get('data', {})

                if data_type == "ChatDataEntry":
                    speaker_id = data.get("actor_id")
                    message = data.get("message")
                    if speaker_id and message:
                        unique_speaker_messages.add((speaker_id, message))
                elif data_type == "DayExileElectedDataEntry":
                    player_id = data.get('elected_player_id')
                    role = data.get('elected_player_role_name')
                    if player_id and role:
                        dynamic_moderator_messages.add(f"Player {player_id} was exiled by vote. Their role was a {role}.")
                elif data_type == "WerewolfNightEliminationDataEntry":
                    player_id = data.get('eliminated_player_id')
                    role = data.get('eliminated_player_role_name')
                    if player_id and role:
                        dynamic_moderator_messages.add(f"Player {player_id} was eliminated. Their role was a {role}.")
                elif data_type == "DoctorSaveDataEntry":
                    player_id = data.get('saved_player_id')
                    if player_id:
                        dynamic_moderator_messages.add(f"Player {player_id} was attacked but saved by a Doctor!")
                elif data_type == "GameEndResultsDataEntry":
                    winner_team = data.get('winner_team')
                    if winner_team:
                        dynamic_moderator_messages.add(f"The game is over. The {winner_team} team has won!")

            except json.JSONDecodeError:
                print(f"  - Warning: Could not decode JSON: {json_str}")

    # Generate and cache dynamic moderator audio
    for message in dynamic_moderator_messages:
        map_key = f"moderator:{message}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"  - Generating dynamic moderator audio for: \"{message}\" ")
            audio_content = get_tts_audio(message, voice_name=MODERATOR_VOICE)
            if audio_content:
                wave_file(audio_path, audio_content)
                audio_map[map_key] = audio_path
        else:
            audio_map[map_key] = audio_path

    # Generate and save audio for each unique message from each speaker
    for speaker_id, message in unique_speaker_messages:
        voice = player_voice_map.get(speaker_id)
        if not voice:
            print(f"  - Warning: No voice found for speaker: {speaker_id}")
            continue

        map_key = f"{speaker_id}:{message}"
        filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"  - Generating audio for {speaker_id} ({voice}): \"{message[:40]}...\" ")
            audio_content = get_tts_audio(message, voice_name=voice)
            if audio_content:
                wave_file(audio_path, audio_content)
                audio_map[map_key] = audio_path
        else:
            audio_map[map_key] = audio_path

    print("4. Rendering the game to an HTML file...")
    html_content = env.render(mode="html")

    print("5. Injecting the local audio map into the HTML...")
    audio_map_json = json.dumps(audio_map)
    injection_script = f"<script>window.AUDIO_MAP = {audio_map_json};</script>"
    html_content = html_content.replace('</head>', f'{injection_script}</head>')

    with open(OUTPUT_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n6. Starting local server to serve '{OUTPUT_HTML_FILE}' and audio files...")
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=".", **kwargs)

    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        print(f"\nServing replay at: http://localhost:{PORT}/{OUTPUT_HTML_FILE}")
        print("Open this URL in your web browser.")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")