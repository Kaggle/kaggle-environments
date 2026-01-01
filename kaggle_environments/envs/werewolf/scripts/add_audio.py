import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import wave
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import yaml
from dotenv import load_dotenv
from google import genai
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech
from google.genai import types

from kaggle_environments.envs.werewolf.game.consts import EventName
from kaggle_environments.envs.werewolf.game.roles import shuffle_ids
from kaggle_environments.envs.werewolf.runner import setup_logger

# Initialize logger
logger = logging.getLogger(__name__)


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
            logger.info(f"Mapped ID '{agent_id}' -> '{unique_name}'")

    def get_name(self, agent_id: str) -> str:
        """Returns the disambiguated display name for an agent ID."""
        return self.id_to_display.get(agent_id, agent_id)

    def replace_names(self, text: str) -> str:
        """Replaces all occurrences of known agent IDs in text with display names."""
        if not text:
            return text

        # Sort by length descending to prevent partial matches (though IDs are usually distinct)
        # Using word boundaries to avoid replacing substrings
        sorted_ids = sorted(self.id_to_display.keys(), key=len, reverse=True)

        for agent_id in sorted_ids:
            display_name = self.id_to_display[agent_id]
            if agent_id != display_name:
                # Use regex for whole word matching
                pattern = r'\b' + re.escape(agent_id) + r'\b'
                text = re.sub(pattern, display_name, text)

        return text


class AudioConfig:
    """Handles loading and accessing audio configuration."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self) -> Dict:
        """Loads YAML configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def paths(self) -> Dict:
        return self.data.get("paths", {})

    @property
    def voices(self) -> Dict:
        return self.data.get("voices", {})

    @property
    def static_moderator_messages(self) -> Dict:
        return self.data.get("audio", {}).get("static_moderator_messages", {})

    def get_vertex_model(self) -> str:
        return self.data.get("vertex_ai_model", "gemini-2.5-flash-tts")


class ReplayParser:
    """Handles extracting game event data from the replay JSON."""

    def __init__(self, replay_data: Dict):
        self.replay_data = replay_data

    def extract_chronological_script(self) -> List[Dict]:
        """Extracts a chronological list of events for full context."""
        script = []
        info = self.replay_data.get("info", {})
        steps = self.replay_data.get("steps", [])

        # We need to interleave moderator events and player actions
        # Moderator events are in info['MODERATOR_OBSERVATION'] (steps)
        mod_logs = info.get("MODERATOR_OBSERVATION", [])

        for step_idx, step in enumerate(steps):
            # 1. Moderator Events for this step
            if step_idx < len(mod_logs):
                for data_entry in mod_logs[step_idx]:
                    json_str = data_entry.get("json_str")
                    try:
                        event = json.loads(json_str)
                        description = event.get("description", "")
                        data = event.get("data", {})
                        event_name = event.get("event_name", "")

                        if description:
                            # Enrich description with dynamic data if needed
                            # For now, just use description as the "text"
                            script.append({
                                "type": "moderator",
                                "text": description,
                                "day": event.get("day")
                            })
                    except:
                        pass

            # 2. Player Actions in this step
            for agent_state in step:
                action = agent_state.get("action")
                if action and isinstance(action, dict):
                    kwargs = action.get("kwargs", {})
                    raw_completion = kwargs.get("raw_completion")

                    if raw_completion:
                        # Try to parse the message
                        try:
                            # format_json_string logic?
                            # Assuming structure { "thought": ..., "response": ... }
                            if "```json" in raw_completion:
                                raw_completion = raw_completion.split("```json")[1].split("```")[0].strip()
                            elif "```" in raw_completion:
                                raw_completion = raw_completion.split("```")[1].split("```")[0].strip()

                            content = json.loads(raw_completion)
                            message = content.get("response") or content.get("message")

                            # Get speaker name
                            obs = agent_state.get("observation", {})
                            raw_obs = obs.get("raw_observation", {})
                            speaker_id = raw_obs.get("player_id", obs.get("player_id"))

                            if message and speaker_id:
                                script.append({
                                    "type": "player",
                                    "speaker": speaker_id,
                                    "text": message
                                })
                        except:
                            pass
        return script

    def extract_messages(self) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """Extracts unique speaker messages and dynamic moderator messages."""
        logger.info("Extracting game data from replay...")
        unique_speaker_messages = set()
        dynamic_moderator_messages = set()

        info = self.replay_data.get("info", {})
        moderator_log_steps = info.get("MODERATOR_OBSERVATION", [])

        for step_log in moderator_log_steps:
            for data_entry in step_log:
                json_str = data_entry.get("json_str")
                data_type = data_entry.get("data_type")

                try:
                    event = json.loads(json_str)
                    data = event.get("data", {})
                    event_name = event.get("event_name")
                    description = event.get("description", "")
                    day_count = event.get("day")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"  - Skipping log entry, failed to parse json_str: {e}")
                    continue

                self._process_entry(
                    data_type,
                    data,
                    event_name,
                    description,
                    day_count,
                    unique_speaker_messages,
                    dynamic_moderator_messages
                )

        logger.info(f"Found {len(unique_speaker_messages)} unique player messages.")
        logger.info(f"Found {len(dynamic_moderator_messages)} dynamic moderator messages.")
        return unique_speaker_messages, dynamic_moderator_messages

    def _process_entry(self, data_type, data, event_name, description, day_count,
                       unique_speaker_messages, dynamic_moderator_messages):
        """Processes a single event entry."""
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


class LLMEnhancer:
    """Handles text enhancement using LLM."""

    def __init__(self, api_key: Optional[str], prompt_path: str, cache_path: str, disabled: bool = False):
        self.disabled = disabled
        self.client = None
        self.prompt_template = self._load_prompt(prompt_path)
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.cache_updated = False

        if not disabled and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"Failed to init GenAI client for LLM enhancement: {e}")

    def _load_prompt(self, path: str) -> str:
        if not os.path.exists(path):
            logger.warning(f"Prompt file not found: {path}. Using default.")
            return "Rewrite the following text to be theatrical and expressive for TTS: '{text}'"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode cache file: {self.cache_path}. Starting fresh.")
        return {}

    def save_cache(self):
        """Saves cache to disk if updated."""
        if self.cache_updated:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved LLM cache to {self.cache_path}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"Failed to save cache: {e}")

    def enhance_script(self, script: List[Dict], name_manager: NameManager) -> Dict[str, str]:
        """Enhances the full game script in one go (or chunks)."""
        if self.disabled or not self.client:
            return {}

        # 1. Format script for LLM
        transcript_lines = []
        # We also need a way to map back: "Speaker:OriginalText" -> EnhancedText

        for entry in script:
            if entry["type"] == "moderator":
                text = name_manager.replace_names(entry["text"])
                transcript_lines.append(f"[Moderator]: {text}")
            elif entry["type"] == "player":
                speaker_display = name_manager.get_name(entry["speaker"])
                # The script provided to LLM should use *Display Names*
                text = name_manager.replace_names(entry["text"])
                transcript_lines.append(f"[{speaker_display}]: {text}")

        transcript = "\n".join(transcript_lines)

        # Check cache logic could be improved, but for now assumption is unique scripts

        prompt = self.prompt_template.replace("{transcript}", transcript)

        logger.info(f"Sending full script ({len(script)} events) to Gemini for enhancement...")

        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            if response.text:
                try:
                    enhanced_map = json.loads(response.text)
                    if isinstance(enhanced_map, list):
                        logger.warning("LLM returned a list instead of a dict. Attempting to merge if possible.")
                        # If list of dicts, merge them? Or just empty?
                        # If it's a list, it might be [ { "Key": "Val" }, ... ]
                        merged = {}
                        for item in enhanced_map:
                            if isinstance(item, dict):
                                merged.update(item)
                        if merged:
                            enhanced_map = merged
                            logger.info(f"Merged {len(merged)} entries from list.")
                        else:
                            logger.warning("Could not extract map from list. Using empty map.")
                            enhanced_map = {}

                    if not isinstance(enhanced_map, dict):
                        logger.warning(f"LLM returned unexpected type: {type(enhanced_map)}. Using empty map.")
                        enhanced_map = {}

                    logger.info(f"Received {len(enhanced_map)} enhanced entries.")
                    return enhanced_map
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON response from LLM.")
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")

        return {}

    def enhance(self, text: str, speaker: str) -> str:
        """Enhances text if client is available and text is not cached."""
        if self.disabled or not self.client:
            return text

        cache_key = f"{speaker}:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        logger.info(f"  - Enhancing text for {speaker}...")
        prompt = self.prompt_template.format(text=text, speaker=speaker)

        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            enhanced = response.text.strip()
            if enhanced:
                self.cache[cache_key] = enhanced
                self.cache_updated = True
                return enhanced
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"  - LLM enhancement failed: {e}")

        return text


class TTSGenerator(ABC):
    """Abstract base class for TTS generators."""

    @abstractmethod
    def generate(self, text: str, voice: str, **kwargs) -> Optional[bytes]:
        """Generates audio for the given text."""
        pass


class VertexTTSGenerator(TTSGenerator):
    """Generates audio using Vertex AI TTS."""

    def __init__(self, model_name: str):
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable required for Vertex AI.")
        self.client = texttospeech.TextToSpeechClient()
        self.model_name = model_name

    def generate(self, text: str, voice: str, **kwargs) -> Optional[bytes]:
        if not text:
            return None

        # Vertex TTS (Standard) doesn't support 'style_prompt' natively in SynthesisInput yet
        # unless using a specific generative endpoint which this client might not target by default.
        # But if the user provides markup tags (e.g. SSML), we could use it.
        # For now, we will verify if markup_tags indicates SSML.

        markup_tags = kwargs.get("markup_tags")
        is_ssml = False
        input_text = text
        if markup_tags:
            # Basic check if it looks like XML/SSML
            if "<" in markup_tags and ">" in markup_tags:
                # Wrap text: <tag>text</tag>
                # But we don't know the closing tag from the opening tag easily without parsing.
                # The prompt asked for "tags to wrap the text".
                # If LLM returns "<speak>", we assume it's full SSML?
                # If LLM returns '(rate="fast")', it's not SSML.
                # Let's treat it as text for now to avoid breaking standard TTS.
                pass

        try:
            synthesis_input = texttospeech.SynthesisInput(text=input_text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code="en-US", name=voice, model_name=self.model_name
            )
            audio_conf = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3, sample_rate_hertz=24000
            )
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_conf
            )
            return response.audio_content
        except (GoogleAPICallError, ValueError) as e:
            logger.error(f"  - Error generating audio (Vertex): {e}")
            return None


class GeminiTTSGenerator(TTSGenerator):
    """Generates audio using Google GenAI TTS."""

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY required for Google GenAI.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, text: str, voice: str, **kwargs) -> Optional[bytes]:
        if not text:
            return None

        style_prompt = kwargs.get("style_prompt")

        # Combine instructions and text
        # Format based on user example: "{Style Prompt}: {Text}"
        # text now contains inline tags if provided by LLM.

        final_content = text
        if style_prompt:
            final_content = f"{style_prompt}: {text}"

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-tts",
                contents=final_content,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
            return response.candidates[0].content.parts[0].inline_data.data
        except (GoogleAPICallError, ValueError) as e:
            logger.error(f"  - Error generating audio (GenAI): {e}")
            return None


class AudioManager:
    """Orchestrates the audio generation process."""

    def __init__(self, config: AudioConfig, enhancer: LLMEnhancer, tts: TTSGenerator, output_dir: str):
        self.config = config
        self.enhancer = enhancer
        self.tts = tts
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, config.paths.get("audio_dir_name", "audio"))
        os.makedirs(self.audio_dir, exist_ok=True)
        self.audio_map = {}
        self.name_manager = None

    def process_replay(self, replay_data: Dict):
        """Runs the full processing pipeline on the replay data."""
        self.name_manager = NameManager(replay_data)

        parser = ReplayParser(replay_data)

        # 1. Extract Full Context Script & Enhance
        chronological_script = parser.extract_chronological_script()
        enhanced_map = self.enhancer.enhance_script(chronological_script, self.name_manager)

        # 2. Extract Unique Messages for Audio Generation (as before)
        unique_msgs, dyn_mod_msgs = parser.extract_messages()

        messages = self._prepare_messages(unique_msgs, dyn_mod_msgs, replay_data, enhanced_map)
        self._generate_audio_batch(messages)
        self._save_audio_map()

        return messages

    def _prepare_messages(self, unique_msgs, dyn_mod_msgs, replay_data, enhanced_map) -> List[Dict]:
        """Prepares a list of message objects for processing."""
        messages = []

        # Helper to add
        def add(speaker_id, key, text, voice):
            # Key remains raw for lookup compatibility
            # Text is updated with display names for better TTS
            tts_text = self.name_manager.replace_names(text)

            # Lookup enhancement
            speaker_display = self.name_manager.get_name(speaker_id) if speaker_id != "moderator" else "Moderator"
            signature = f"{speaker_display}: {tts_text}"

            enhancement = enhanced_map.get(signature)
            final_text = tts_text
            style_prompt = None
            # markup_tags = None # No longer separate

            if enhancement:
                if isinstance(enhancement, dict):
                    # New structure: {"style_prompt": "...", "text_content": "..."}
                    style_prompt = enhancement.get("style_prompt")
                    # The LLM *returns* the text with tags inserted inline.
                    # We trust the LLM to preserve the words (as instructed) and only add tags.
                    enhanced_text = enhancement.get("text_content")
                    if enhanced_text:
                        final_text = enhanced_text
                else:
                    # Legacy fallback
                    final_text = enhancement

            messages.append({
                "speaker": speaker_id,
                "key": key,
                "original_text": text,
                "final_text": final_text,
                "style_prompt": style_prompt,
                "voice": voice
            })

        # 1. Static Moderator Messages
        moderator_voice = self.config.voices["moderator"]
        static_msgs = self.config.static_moderator_messages
        key_aliases = {
            "discussion_begins": "Discussion begins!",
            "voting_begins": "Exile voting begins!"
        }

        for key, text in static_msgs.items():
            add("moderator", key, text, moderator_voice)
            if key in key_aliases:
                add("moderator", key_aliases[key], text, moderator_voice)

        # 2. Dynamic Moderator Messages
        for msg in dyn_mod_msgs:
            # For dynamic messages, the key is the exact text
            add("moderator", msg, msg, moderator_voice)

        # 3. Player Messages
        game_config = replay_data.get("configuration", {})
        player_voices = self.config.voices.get("players", {})
        player_voice_map = {
            a["id"]: player_voices.get(a["id"]) for a in game_config.get("agents", [])
        }

        for speaker_id, text in unique_msgs:
            voice = player_voice_map.get(speaker_id)
            if voice:
                add(speaker_id, text, text, voice)
            else:
                logger.warning(f"  - Warning: No voice found for speaker: {speaker_id}")

        return messages

    def _generate_audio_batch(self, messages: List[Dict]):
        """Generates audio for a batch of messages."""
        logger.info(f"Processing {len(messages)} messages...")

        for msg in messages:
            speaker_id = msg["speaker"]
            final_text = msg["final_text"]
            voice = msg["voice"]
            key = msg["key"]

            # Extract style/tags
            style_prompt = msg.get("style_prompt")
            markup_tags = msg.get("markup_tags")

            # Generate Keys & Path
            # IMPORTANT: map_key uses raw speaker_id and raw key/text to match visualizer logic
            map_key = f"{speaker_id}:{key}"
            filename = hashlib.md5(map_key.encode()).hexdigest() + ".wav"
            audio_path_html = os.path.join(self.config.paths.get("audio_dir_name", "audio"), filename)

            # Use abspath for local check, but audio_path_on_disk should be inside output_dir
            audio_path_disk = os.path.join(self.audio_dir, filename)

            if not os.path.exists(audio_path_disk):
                # print(f'Generating audio: "{final_text[:60]}..."') # Reduce noise
                logger.debug(f'Generating audio for {speaker_id}: "{final_text[:40]}..." (Style: {style_prompt})')

                audio_content = self.tts.generate(
                    final_text,
                    voice,
                    style_prompt=style_prompt,
                    markup_tags=markup_tags
                )

                if audio_content:
                    self._save_wave(audio_path_disk, audio_content)
                    self.audio_map[map_key] = audio_path_html
            else:
                self.audio_map[map_key] = audio_path_html

    def _save_wave(self, filename, pcm, channels=1, rate=24000, sample_width=2):
        """Saves PCM audio bytes to WAV."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)

    def _save_audio_map(self):
        """Saves the audio map JSON."""
        path = os.path.join(self.output_dir, "audio_map.json")
        logger.info(f"Saving audio map to: {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.audio_map, f, indent=2)

    def generate_debug_audio(self):
        """Generates a single debug file."""
        logger.info("Generating debug audio...")
        debug_dir_name = self.config.paths.get("debug_audio_dir_name", "audio_debug")
        debug_dir = os.path.join(self.output_dir, debug_dir_name)
        os.makedirs(debug_dir, exist_ok=True)

        filename = "debug_audio.wav"
        path = os.path.join(debug_dir, filename)

        if not os.path.exists(path):
            content = self.tts.generate("Testing start, testing end.", "Charon")
            if content:
                self._save_wave(path, content)

        # Map everything to this file? (Not implemented fully in minimal functionality check)
        logger.info(f"Debug audio generated at {path}")


class VisualizerServer:
    """Manages the Vite dev server."""

    @staticmethod
    def start(visualizer_dir: str, replay_path: str, audio_map_path: str):
        """Starts Vite."""
        visualizer_dir = os.path.abspath(visualizer_dir)
        replay_path = os.path.abspath(replay_path)
        audio_map_path = os.path.abspath(audio_map_path)

        logger.info(f"\nStarting Vite server from '{visualizer_dir}'...")

        # Relativize for cleaner env vars
        try:
            rel_replay = os.path.relpath(replay_path, visualizer_dir)
            rel_audio_map = os.path.relpath(audio_map_path, visualizer_dir)
        except ValueError:
            rel_replay = replay_path
            rel_audio_map = audio_map_path

        env = os.environ.copy()
        env["VITE_REPLAY_FILE"] = rel_replay
        env["VITE_AUDIO_MAP_FILE"] = rel_audio_map
        cmd = [
            "npx",
            "vite",
        ]

        print("\nRunning command:", " ".join(cmd))
        print("Press Ctrl+C to stop the server.")

        try:
            subprocess.run(cmd, cwd=visualizer_dir, env=env, check=False)
        except KeyboardInterrupt:
            print("\nServer stopped.")


def load_env_modules():
    """Loads environment variables from repo root."""
    try:
        from kaggle_environments import PROJECT_ROOT
        env_path = os.path.join(PROJECT_ROOT, os.pardir, ".env")
        if os.path.exists(env_path):
            logger.info(f"Loading .env from: {env_path}")
            load_dotenv(env_path)
        else:
            logger.info(f".env not found at {env_path}, relying on system vars.")
    except ImportError:
        logger.warning("Could not import kaggle_environments.PROJECT_ROOT")


def main():
    parser = argparse.ArgumentParser(description="Add audio to a Werewolf game replay.")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to replay JSON.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory.")
    parser.add_argument("-c", "--config_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs/audio/standard.yaml"))
    parser.add_argument("--debug-audio", action="store_true", help="Generate debug audio only.")
    parser.add_argument("--serve", action="store_true", help="Start Vite server.")
    parser.add_argument("--tts-provider", type=str, default="vertex_ai", choices=["vertex_ai", "google_genai"])
    parser.add_argument("--prompt_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs/audio/theatrical_prompt.txt"))
    parser.add_argument("--cache_path", type=str, help="LLM cache file path.")
    parser.add_argument("--disable_llm_enhancement", action="store_true", help="Disable LLM enhancement.")

    args = parser.parse_args()

    # Defaults
    if not args.output_dir:
        args.output_dir = "werewolf_replay_audio"
    if not args.cache_path:
        args.cache_path = os.path.join(args.output_dir, "llm_cache.json")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(output_dir=args.output_dir, base_name="add_audio")
    load_env_modules()

    # Config
    try:
        config = AudioConfig(args.config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Replay
    if not os.path.exists(args.input_path):
        logger.error(f"Replay not found: {args.input_path}")
        return
    with open(args.input_path, "r", encoding="utf-8") as f:
        replay_data = json.load(f)

    # Components
    gemini_key = os.getenv("GEMINI_API_KEY")
    enhancer = LLMEnhancer(gemini_key, args.prompt_path, args.cache_path, args.disable_llm_enhancement)

    if args.tts_provider == "vertex_ai":
        tts = VertexTTSGenerator(config.get_vertex_model())
    else:
        tts = GeminiTTSGenerator(gemini_key)

    manager = AudioManager(config, enhancer, tts, args.output_dir)

    if args.debug_audio:
        manager.generate_debug_audio()
    else:
        manager.process_replay(replay_data)
        enhancer.save_cache()

    if args.serve:
        vis_dir = os.path.join(os.path.dirname(__file__), "../visualizer/default")
        audio_map_path = os.path.join(args.output_dir, "audio_map.json")
        VisualizerServer.start(vis_dir, args.input_path, audio_map_path)


if __name__ == "__main__":
    main()
