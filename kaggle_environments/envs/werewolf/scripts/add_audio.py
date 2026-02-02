import argparse
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import wave
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import yaml
from dotenv import load_dotenv
from google import genai
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from google.cloud import texttospeech
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from kaggle_environments.envs.werewolf.game.consts import EventName
from kaggle_environments.envs.werewolf.game.roles import shuffle_ids
from kaggle_environments.envs.werewolf.runner import setup_logger

# Initialize logger
logger = logging.getLogger(__name__)


class NameManager:
    """Manages player display names and disambiguation."""

    def __init__(self, replay_data: Dict, simplification_rules: List[Dict] = None,
                 simplification_map: Dict[str, str] = None):
        self.id_to_display = {}
        self.simplification_rules = simplification_rules or []
        self.simplification_map = simplification_map or {}
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

        # 2.5 Apply Name Simplification (Map first, then Rules)
        for agent in agents:
            name = agent.get("display_name") or agent.get("name") or agent.get("id")
            original_name = name

            # A. Explicit Map
            if name in self.simplification_map:
                name = self.simplification_map[name]

            # B. Regex Rules (apply if map didn't change it, OR apply on top? User said "simplify names in the map", implying map is primary)
            # Let's apply rules AFTER map, in case map output needs cleaning? Or maybe map is final?
            # Usually map is "exact override". If map hits, we might skip rules.
            # But let's allow rules to run just in case, unless map changed it?
            # Actually, if I map "A" -> "B", I probably don't want regex to change "B". 
            # But if I map "A-v1" -> "A-v1-clean", maybe I do?
            # Safest is: If map hit, use map value. Checks rules only if map didn't hit?
            # OR: Apply map, THEN apply rules to the result.
            # Let's apply map, then rules.

            if self.simplification_rules:
                for rule in self.simplification_rules:
                    pattern = rule.get("pattern")
                    replacement = rule.get("replacement", "")
                    if pattern:
                        try:
                            name = re.sub(pattern, replacement, name)
                        except re.error as e:
                            logger.warning(f"Invalid regex pattern '{pattern}': {e}")

            if name != original_name:
                agent["display_name"] = name.strip()

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
                # Use regex for whole word matching, optionally consuming "Player" prefix
                pattern = r'\b(?:Player\s*)?' + re.escape(agent_id) + r'\b'
                text = re.sub(pattern, display_name, text)

        return text

        return text


class TranscriptManager:
    """Handles moderator message overrides with normalization and substring replacement."""

    def __init__(self, overrides: Dict[str, str]):
        self.overrides = overrides or {}

    def normalize(self, text: str) -> str:
        """Collapses extra spaces before terminal punctuation introduced by legacy cleaning logic."""
        if not text:
            return ""
        return re.sub(r"\s+([.?!])", r"\1", text.strip())

    def apply_overrides(self, text: str) -> str:
        """Applies transcript overrides to consistent moderator messages."""
        if not text:
            return text

        result = self.normalize(text)

        # Sort keys by length descending to prevent partial match collisions
        sorted_keys = sorted(self.overrides.keys(), key=len, reverse=True)

        for key in sorted_keys:
            normalized_key = self.normalize(key)
            replacement_value = self.overrides[key]

            if normalized_key and normalized_key in result:
                # Global replacement of the normalized fragment within the normalized text
                result = result.replace(normalized_key, replacement_value)

        return result


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

    @property
    def name_simplification_rules(self) -> List[Dict]:
        return self.data.get("audio", {}).get("name_simplification_rules", [])

    @property
    def name_simplification_map(self) -> Dict[str, str]:
        return self.data.get("audio", {}).get("name_simplification_map", {})

    @property
    def speech_intro_template(self) -> str:
        return self.data.get("audio", {}).get("speech_intro_template", "")

    @property
    def transcript_overrides(self) -> Dict[str, str]:
        return self.data.get("audio", {}).get("transcript_overrides", {})

    def get_vertex_model(self) -> str:
        return self.data.get("vertex_ai_model", "gemini-2.5-flash-tts")

    @property
    def vertex_ai_regions(self) -> List[str]:
        return self.data.get("vertex_ai_regions", [])


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

                        if event_name == EventName.VOTE_REQUEST:
                            script.append(
                                {"type": "moderator", "text": "Wake up Werewolves, who would you like to eliminate?",
                                 "day": event.get("day")})
                        elif event_name == EventName.HEAL_REQUEST:
                            script.append({"type": "moderator", "text": "Wake up Doctor, who would you like to save?",
                                           "day": event.get("day")})
                        elif event_name == EventName.INSPECT_REQUEST:
                            script.append({"type": "moderator", "text": "Wake up Seer, who would you like to inspect?",
                                           "day": event.get("day")})
                        elif description:
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

    def extract_messages(self) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
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
                text = f"{data['actor_id']} votes to exile {data['target_id']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "WerewolfNightVoteDataEntry":
            if data.get("actor_id") and data.get("target_id"):
                text = f"{data['actor_id']} votes to eliminate {data['target_id']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "SeerInspectActionDataEntry":
            if data.get("actor_id") and data.get("target_id"):
                text = f"{data['actor_id']} inspects {data['target_id']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "DoctorHealActionDataEntry":
            if data.get("actor_id") and data.get("target_id"):
                text = f"{data['actor_id']} heals {data['target_id']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "DayExileElectedDataEntry":
            if all(k in data for k in ["elected_player_id", "elected_player_role_name"]):
                text = f"{data['elected_player_id']} was exiled by vote. Their role was a {data['elected_player_role_name']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "WerewolfNightEliminationDataEntry":
            if all(k in data for k in ["eliminated_player_id", "eliminated_player_role_name"]):
                text = f"{data['eliminated_player_id']} was eliminated. Their role was a {data['eliminated_player_role_name']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "DoctorSaveDataEntry":
            if "saved_player_id" in data:
                text = f"{data['saved_player_id']} was attacked but saved by a Doctor!"
                dynamic_moderator_messages.add((text, text))
        elif data_type == "SeerInspectResultDataEntry":
            if data.get("role"):
                text = f"{data['actor_id']} saw {data['target_id']}'s role is {data['role']}."
                dynamic_moderator_messages.add((text, text))
            elif data.get("team"):
                text = f"{data['actor_id']} saw {data['target_id']}'s team is {data['team']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "GameEndResultsDataEntry":
            if "winner_team" in data:
                text = f"The game is over. The {data['winner_team']} team has won!"
                dynamic_moderator_messages.add((text, text))
        elif data_type == "WerewolfNightEliminationElectedDataEntry":
            if "elected_target_player_id" in data:
                text = f"The werewolves have chosen to eliminate {data['elected_target_player_id']}."
                dynamic_moderator_messages.add((text, text))
        elif data_type == "RequestWerewolfVotingDataEntry":
            text = "Wake up Werewolves, who would you like to eliminate?"
            key = description if description else text
            dynamic_moderator_messages.add((key, text))
        elif event_name == EventName.DAY_START:
            text = f"Day {day_count} begins!"
            dynamic_moderator_messages.add((text, text))
        elif event_name == EventName.NIGHT_START:
            text = f"Night {day_count} begins!"
            key = description if description else text
            dynamic_moderator_messages.add((key, text))
        elif event_name == EventName.MODERATOR_ANNOUNCEMENT:
            if "discussion rule is" in description:
                text = "Discussion begins!"
                key = description if description else text
                dynamic_moderator_messages.add((key, text))
            elif "Voting phase begins" in description:
                text = "Exile voting begins!"
                key = description if description else text
                dynamic_moderator_messages.add((key, text))
        elif event_name == EventName.VOTE_REQUEST:
            text = "Wake up Werewolves, who would you like to eliminate?"
            key = description if description else text
            dynamic_moderator_messages.add((key, text))
        elif event_name == EventName.HEAL_REQUEST:
            text = "Wake up Doctor, who would you like to save?"
            key = description if description else text
            dynamic_moderator_messages.add((key, text))
        elif event_name == EventName.INSPECT_REQUEST:
            text = "Wake up Seer, who would you like to inspect?"
            key = description if description else text
            dynamic_moderator_messages.add((key, text))


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

    def enhance_script(self, script: List[Dict], name_manager: NameManager, tqdm_kwargs: Dict = None) -> Dict[str, str]:
        """Enhances the full game script in chunks for better context and progress reporting."""
        if self.disabled or not self.client:
            return {}

        tqdm_kwargs = tqdm_kwargs or {}
        tqdm_kwargs = tqdm_kwargs.copy()
        tqdm_kwargs["desc"] = "LLM Enhancement"
        tqdm_kwargs["unit"] = "chunk"

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

        # 2. Chunk and Process
        chunk_size = 20
        all_enhanced = {}

        chunks = [transcript_lines[i:i + chunk_size] for i in range(0, len(transcript_lines), chunk_size)]

        for chunk in tqdm(chunks, **tqdm_kwargs):
            chunk_transcript = "\n".join(chunk)
            prompt = self.prompt_template.replace("{transcript}", chunk_transcript)

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
                            merged = {}
                            for item in enhanced_map:
                                if isinstance(item, dict):
                                    merged.update(item)
                            enhanced_map = merged

                        if isinstance(enhanced_map, dict):
                            all_enhanced.update(enhanced_map)
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode JSON response from LLM for a chunk.")
            except Exception as e:
                logger.warning(f"LLM enhancement failed for a chunk: {e}")

        logger.info(f"Received {len(all_enhanced)} enhanced entries across {len(chunks)} chunks.")
        return all_enhanced

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

    def __init__(self, model_name: str, regions: List[str] = None):
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable required for Vertex AI.")

        self.model_name = model_name
        self.client = None
        self.region = None

        if not regions:
            # Default client (implicitly uses default region)
            self.client = texttospeech.TextToSpeechClient()
            logger.info("Initialized Vertex AI client with default region.")
        else:
            # Select ONE region for this entire game session
            selected_region = random.choice(regions)
            self.region = selected_region

            api_endpoint = f"{selected_region}-texttospeech.googleapis.com"
            client_options = ClientOptions(api_endpoint=api_endpoint)
            try:
                self.client = texttospeech.TextToSpeechClient(client_options=client_options)
                logger.info(f"Initialized Vertex AI client for region: {selected_region}")
            except Exception as e:
                logger.error(f"Failed to init client for region {selected_region}: {e}")
                raise e

    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        before_sleep=lambda retry_state: logger.warning(
            f"Quota exceeded, retrying... (Attempt {retry_state.attempt_number})")
    )
    def generate(self, text: str, voice: str, **kwargs) -> Optional[bytes]:
        if not text:
            return None

        # Clean custom markup: (params)[content] -> content
        # Example: (rate="fast")[Hello] -> Hello
        input_text = re.sub(r"\([^)]+\)\[([^\]]+)\]", r"\1", text)

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
                language_code="en-US", name=voice
            )
            if self.model_name:
                voice_params.model_name = self.model_name
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
    """Generates audio using Google GenAI TTS (via Vertex AI TextToSpeechClient)."""

    def __init__(self, api_key: str = None, regions: List[str] = None):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable required for Vertex AI Gemini.")

        # Use default client (ADC) or api_key if provided (ClientOptions not shown in example but Standard pattern)
        # The example uses ADC. We'll stick to ADC/Environment unless an explicit key is needed for some reason.
        # But we previously used genai.Client with key or vertexai=True. 
        # TextToSpeechClient typically uses ADC.
        self.client = texttospeech.TextToSpeechClient()

    def generate(self, text: str, voice: str, **kwargs) -> Optional[bytes]:
        if not text:
            return None

        style_prompt = kwargs.get("style_prompt")
        # Default style for moderator if not provided
        final_prompt = style_prompt if style_prompt else "Speak naturally and clearly."

        try:
            # Construct SynthesisInput with prompt
            # Note: prompt argument requires google-cloud-texttospeech >= 2.29.0
            synthesis_input = texttospeech.SynthesisInput(text=text, prompt=final_prompt)

            voice_params = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice,
                model_name="gemini-2.5-flash-tts"
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # WAV
                sample_rate_hertz=24000
            )

            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )

            return response.audio_content

        except Exception as e:
            logger.error(f"Gemini/Vertex TTS failed: {e}")
            return None


class RandomVoiceAssigner:
    """Deterministically assigns voices to players from a pool."""

    def __init__(self, voice_pool: List[str], fixed_map: Dict[str, str], seed: int):
        self.voice_pool = sorted(voice_pool)  # Sort for stability
        self.fixed_map = fixed_map
        self.seed = seed
        self.assigned_voices = {}
        self.rng = random.Random(seed)

    def assign_voices(self, players: List[str]) -> Dict[str, str]:
        """Assigns voices to a list of players."""
        # 1. Assign fixed voices first
        remaining_players = []
        for p in players:
            if p in self.fixed_map:
                self.assigned_voices[p] = self.fixed_map[p]
            else:
                remaining_players.append(p)

        # 2. Assign from pool for remaining
        # Shuffle pool deterministically
        pool = list(self.voice_pool)
        self.rng.shuffle(pool)

        # Assign round-robin if pool is smaller than players (unlikely with 30 voices but safe)
        for i, p in enumerate(remaining_players):
            voice = pool[i % len(pool)]
            self.assigned_voices[p] = voice

        return self.assigned_voices

    def get_voice(self, player_name: str) -> str:
        return self.assigned_voices.get(player_name, self.voice_pool[0] if self.voice_pool else "en-US-Journey-D")


class AudioManager:
    """Orchestrates the audio generation process."""

    def __init__(self, config: AudioConfig, enhancer: LLMEnhancer, tts: TTSGenerator, output_dir: str,
                 tqdm_kwargs: Dict = None):
        self.config = config
        self.enhancer = enhancer
        self.tts = tts
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, config.paths.get("audio_dir_name", "audio"))
        os.makedirs(self.audio_dir, exist_ok=True)
        self.audio_map = {}
        self.name_manager = None
        self.transcript_manager = TranscriptManager(config.transcript_overrides)
        self.tqdm_kwargs = tqdm_kwargs or {}

    def process_replay(self, replay_data: Dict):
        """Runs the full processing pipeline on the replay data."""
        self.name_manager = NameManager(
            replay_data,
            self.config.name_simplification_rules,
            self.config.name_simplification_map
        )

        parser = ReplayParser(replay_data)

        # 1. Extract Full Context Script & Enhance
        chronological_script = parser.extract_chronological_script()
        enhanced_map = self.enhancer.enhance_script(chronological_script, self.name_manager,
                                                    tqdm_kwargs=self.tqdm_kwargs)

        # 2. Extract Unique Messages for Audio Generation (as before)
        unique_msgs, dyn_mod_msgs = parser.extract_messages()

        messages = self._prepare_messages(unique_msgs, dyn_mod_msgs, replay_data, enhanced_map)
        self._generate_audio_batch(messages)
        self._save_audio_map()

        return messages

    def _prepare_messages(self, unique_msgs, dyn_mod_msgs, replay_data, enhanced_map) -> List[Dict]:
        """Prepares a list of message objects for processing."""
        messages = []
        intro_template = self.config.speech_intro_template

        # Helper to add
        def add(speaker_id, key, text, voice, is_player=False, force_style=None):
            # Key remains raw for lookup compatibility
            # Text is updated with display names for better TTS
            tts_text = self.name_manager.replace_names(text)

            # Apply transcript overrides for moderator messages
            if speaker_id == "moderator":
                tts_text = self.transcript_manager.apply_overrides(tts_text)

            # Lookup enhancement
            speaker_display = self.name_manager.get_name(speaker_id) if speaker_id != "moderator" else "Moderator"
            signature = f"{speaker_display}: {tts_text}"

            enhancement = enhanced_map.get(signature)
            final_text = tts_text
            style_prompt = None

            if enhancement:
                if isinstance(enhancement, dict):
                    style_prompt = enhancement.get("style_prompt")
                    enhanced_text = enhancement.get("text_content")
                    if enhanced_text:
                        final_text = enhanced_text
                else:
                    final_text = enhancement

            # Override style if forced (e.g. for moderator)
            if force_style:
                style_prompt = force_style

            # Prepend intro if it's a player and template exists
            if is_player and intro_template:
                # We insert the intro BEFORE the final text
                # Note: final_text might contain style tags or be pure text.
                # If we rely on Gemini to handle tags, we can just prepend.
                intro = intro_template.format(player=speaker_display)
                final_text = f"{intro} {final_text}"

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
        moderator_style = self.config.data.get("audio", {}).get("moderator_style", "Speak naturally and clearly.")
        key_aliases = {
            "discussion_begins": "Discussion begins!",
            "voting_begins": "Exile voting begins!"
        }

        for key, text in static_msgs.items():
            # Pass the static moderator style
            add("moderator", key, text, moderator_voice, is_player=False, force_style=moderator_style)
            if key in key_aliases:
                add("moderator", key_aliases[key], text, moderator_voice, is_player=False, force_style=moderator_style)

        # 2. Dynamic Moderator Messages
        for msg_item in dyn_mod_msgs:
            # msg_item should be a tuple (key, text) or string
            if isinstance(msg_item, tuple):
                key, text = msg_item
            else:
                key = text = msg_item

            add("moderator", key, text, moderator_voice, is_player=False, force_style=moderator_style)

        # 3. Player Messages
        game_config = replay_data.get("configuration", {})
        seed = game_config.get("seed", 0)  # Default to 0 if no seed, but usually present

        # Initialize Voice Assigner
        # Fixed map comes from config (keys are Original names usually, but here we deal with IDs or Names)
        # config.voices['players'] maps Name -> Voice. 
        # But here we have Agent IDs. 
        # We need to map Agent ID -> Name -> Voice if possible? 
        # Or just use the assigner to map Agent ID -> Voice directly?
        # The assigner logic: fixed_map keys should match whatever we pass to assign_voices.
        # We pass Agent IDs (e.g. "Cedric", "0", etc).

        # Original logic: 
        # player_voices = self.config.voices.get("players", {})
        # player_voice_map = { a["id"]: player_voices.get(a["id"]) ... }
        # This implies standard.yaml has IDs as keys (Kai, Jordan etc).
        # In this game, IDs are likely Names if not randomized? or strings.

        # Let's get all agent IDs first.
        agents = game_config.get("agents", [])
        agent_ids = [a["id"] for a in agents]

        # Retrieve voice pool and players map from config
        voice_pool = self.config.data.get("voices", {}).get("voice_pool", [])
        fixed_player_voices = self.config.voices.get("players", {})

        assigner = RandomVoiceAssigner(voice_pool, fixed_player_voices, seed)
        assigner.assign_voices(agent_ids)

        for speaker_id, text in unique_msgs:
            voice = assigner.get_voice(speaker_id)
            if voice:
                add(speaker_id, text, text, voice, is_player=True)
            else:
                logger.warning(f"  - Warning: No voice found for speaker: {speaker_id}")

        return messages

    def _generate_audio_batch(self, messages: List[Dict]):
        """Generates audio for a batch of messages."""
        logger.info(f"Processing {len(messages)} messages...")

        # Use simple progress bar writing to stdout to avoid logging conflicts
        # Merge defaults with custom kwargs
        pbar_kwargs = {"desc": "Generating Audio", "unit": "msg"}
        pbar_kwargs.update(self.tqdm_kwargs)

        for msg in tqdm(messages, **pbar_kwargs):
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

            # Use CONTENT hash for filename so that if we change the text (e.g. add intro), we regenerate audio.
            # We include voice and style in hash so changing voice/style also forces regen.
            content_signature = f"{final_text}|{voice}|{style_prompt}"
            filename = hashlib.md5(content_signature.encode()).hexdigest() + ".wav"
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

        # Use /@fs/ prefix for absolute paths to allow serving external files
        env = os.environ.copy()
        env["VITE_REPLAY_FILE"] = f"/@fs/{replay_path}"
        env["VITE_AUDIO_MAP_FILE"] = f"/@fs/{audio_map_path}"
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


def process_replay_file(input_path, output_dir, config_path, tts_provider, prompt_path, cache_path, disable_llm,
                        debug_audio=False, tqdm_kwargs=None):
    """Helper to process a single replay file programmatically."""
    tqdm_kwargs = tqdm_kwargs or {}

    with open(input_path, "r", encoding="utf-8") as f:
        replay_data = json.load(f)

    # Setup Components
    config = AudioConfig(config_path)

    # LLM Enhancer
    api_key = os.getenv("GEMINI_API_KEY")
    enhancer = LLMEnhancer(api_key, prompt_path, cache_path, disabled=disable_llm)

    # TTS Generator
    if tts_provider == "gemini":
        tts = GeminiTTSGenerator(api_key, regions=config.vertex_ai_regions)
    else:
        model_name = config.get_vertex_model()
        regions = config.vertex_ai_regions
        tts = VertexTTSGenerator(model_name, regions=regions)

    manager = AudioManager(config, enhancer, tts, output_dir, tqdm_kwargs=tqdm_kwargs)

    setup_logger(output_dir=output_dir, base_name="add_audio")  # Ensure logger is setup for this process?
    # Actually, running in thread might share logger. 
    # But AudioManager uses logger.

    if debug_audio:
        manager.generate_debug_audio()
    else:
        manager.process_replay(replay_data)

    # Save cache if needed
    enhancer.save_cache()


def main():
    parser = argparse.ArgumentParser(description="Add audio to a Werewolf game replay.")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to replay JSON.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory.")
    parser.add_argument("-c", "--config_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs/audio/standard.yaml"))
    parser.add_argument("--debug-audio", action="store_true", help="Generate debug audio only.")
    parser.add_argument("--serve", action="store_true", help="Start Vite server.")
    parser.add_argument("--voice", choices=["chirp", "gemini"], default="gemini",
                        help="Voice model to use (chirp/gemini)")
    parser.add_argument("--prompt_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs/audio/theatrical_prompt.txt"))
    parser.add_argument("--cache_path", type=str, help="LLM cache file path.")
    parser.add_argument("--enable_llm_enhancement", action="store_true",
                        help="Enable LLM enhancement (theatrical rewrites).")
    parser.add_argument("--disable_llm_enhancement", action="store_true",
                        help="Disable LLM enhancement (theatrical rewrites).")

    args = parser.parse_args()

    # Determine LLM status
    # Default to False if not specified, unless enable flag is set.
    # But wait, logic below says disable_llm = args.disable...
    # Let's keep existing logic structure but fixing the args for voice.

    disable_llm = args.disable_llm_enhancement
    if args.enable_llm_enhancement:
        disable_llm = False

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
    enhancer = LLMEnhancer(gemini_key, args.prompt_path, args.cache_path, not args.enable_llm_enhancement)

    if args.voice == "chirp":
        tts = VertexTTSGenerator(config.get_vertex_model(), regions=config.vertex_ai_regions)
    else:
        tts = GeminiTTSGenerator(gemini_key, regions=config.vertex_ai_regions)

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
