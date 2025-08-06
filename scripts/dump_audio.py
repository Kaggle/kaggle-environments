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

# --- Setup ---
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)

# Create the audio directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

client = genai.Client()

# --- Helper Functions ---
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   """Saves PCM audio data to a WAV file."""
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

def get_tts_audio(text: str) -> bytes | None:
    """Fetches TTS audio from Gemini API and returns the audio content as bytes."""
    response = client.models.generate_content(
       model="gemini-2.5-flash-preview-tts",
       contents=text,
       config=types.GenerateContentConfig(
          response_modalities=["AUDIO"],
          speech_config=types.SpeechConfig(
             voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                   voice_name='Kore',
                )
             )
          ),
       )
    )
    data = response.candidates[0].content.parts[0].inline_data.data
    return data

# --- Main Script ---
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
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

    # TODO: vertex AI model still has issues

    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager", "Villager"]
    random.shuffle(roles)
    names = ["gemini-2.5-flash", "deepseek-r1", "kimi-k2", "qwen3", "gpt-4.1", "o4-mini", "claude-4-sonnet", "grok-4"]

    parameter_dict = {
        "together_ai/deepseek-ai/DeepSeek-R1": {"max_tokens": 163839},
        "claude-4-sonnet-20250514": {"max_tokens": 64000}
    }

    models = [
        "gemini/gemini-2.5-flash",
        "together_ai/deepseek-ai/DeepSeek-R1",
        "together_ai/moonshotai/Kimi-K2-Instruct",
        "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "gpt-4.1",
        "o4-mini",
        "claude-4-sonnet-20250514",
        "xai/grok-4-latest",
    ]

    brands = ['gemini', 'deepseek', 'kimi', 'qwen', 'openai', 'openai', 'claude', 'grok']

    agents_config = [
        {"role": role, "id": name, "agent_id": f"llm_harness/{model}",
         "thumbnail": URLS[brand], "display_name": model,
         "agent_harness_name": "llm_harness",
         "llms": [{"model_name": model, "parameters": parameter_dict.get(model, {})}]}
        for role, name, brand, model in zip(roles, names, brands, models)
    ]

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
                    "max_rounds": 2
                }
            }
        }
    )

    print("2. Running a full game episode...")
    env.run(["random"] * 8)

    print("3. Extracting dialogue and generating audio files...")
    unique_messages = set()
    audio_map = {}

    # Collect all unique chat messages from the game history
    for step_data in env.steps:
        for agent_obs in step_data:
            if "new_history_entries_json" in agent_obs["observation"]:
                for entry_json in agent_obs["observation"]["new_history_entries_json"]:
                    entry = entry_json
                    if entry.get("data") and entry["entry_type"] == "discussion":
                        unique_messages.add(entry["data"]["message"])

    # Generate and save audio for each unique message
    for message in unique_messages:
        filename = hashlib.md5(message.encode()).hexdigest() + ".wav"
        audio_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"  - Generating audio for: \"{message[:50]}...\"")
            audio_content = get_tts_audio(message)
            if audio_content:
                wave_file(audio_path, audio_content)
                audio_map[message] = audio_path
        else:
            audio_map[message] = audio_path

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
