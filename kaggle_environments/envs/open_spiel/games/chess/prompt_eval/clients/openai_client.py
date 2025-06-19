# openai_client.py
import time
from typing import Optional, Any, Dict

import openai # Main import
from openai import OpenAIError # Specific error import

try:
    from ..config import get_openai_api_key
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import get_openai_api_key
import logging
from .base_client import LlmClient
from .llm_response import LlmResponse

class OpenAIClient(LlmClient):
    # Map standard names to OpenAI specific names if they differ
    _KWARG_MAP = {
        "temperature": "temperature",
        "max_tokens": "max_completion_tokens",  # o3 models use max_completion_tokens instead of max_tokens
        "stop_sequences": "stop", # OpenAI uses 'stop' for stop sequences
        "top_p": "top_p",
    }

    def __init__(self, logger: logging.Logger, default_model_id: str = "o3-2025-04-16"):
        super().__init__(logger, default_model_id)
        self.client = None # Initialize client to None
        try:
            api_key = get_openai_api_key() # Fetches from llm.config
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not found or is empty.")
            # Instantiate the client (v1.x style)
            self.client = openai.OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully.")
        except EnvironmentError as e:
            self.logger.error(f"Failed to configure OpenAI client: {e}", exc_info=True)
            # Allow continuation, but calls will fail if client is None
        except Exception as e:
            self.logger.error(f"Unexpected error configuring OpenAI client: {e}", exc_info=True)
            # Allow continuation

    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_id: str,
        **kwargs: Any
    ) -> LlmResponse:
        start_time = time.perf_counter()
        llm_response = LlmResponse(model_id=model_id)

        if not self.client:
            llm_response.error = "OpenAI client was not initialized (missing API key?)."
            self.logger.error(llm_response.error)
            llm_response.duration_ms = int(round((time.perf_counter() - start_time) * 1000))
            return llm_response

        try:
            # Map standard kwargs to OpenAI specific ones using the map
            api_params = {}
            for standard_key, value in kwargs.items():
                openai_key = self._KWARG_MAP.get(standard_key)
                if openai_key:
                    # o3 models have restrictions on certain parameters
                    if model_id.startswith("o3") and standard_key == "temperature" and value != 1:
                        self.logger.debug(f"Skipping temperature={value} for o3 model (only default temperature=1 supported)")
                        continue
                    api_params[openai_key] = value
                # else: # Optional: log or handle unknown kwargs if needed
                #     self.logger.warn(f"Ignoring unknown kwarg for OpenAI: {standard_key}")

            # Construct messages list for ChatCompletion
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            self.logger.debug(f"Calling OpenAI model {model_id} with messages and params: {api_params}")

            # --- Use the new client method ---
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                **api_params
            )
            # --- End new client method ---

            self.logger.debug("OpenAI response received.")
            llm_response.raw_response = response.dict() # Store raw response if needed (as dict)

            choice = response.choices[0] if response.choices else None

            if not choice:
                # Check for potential content filter issues (though finish_reason is usually better)
                finish_reason = getattr(response, "finish_reason", "unknown") # Fallback
                llm_response.error = f"No choices returned from OpenAI. Finish reason: {finish_reason}"
                llm_response.stop_reason = finish_reason
                self.logger.warn(f"OpenAI API call failed: {llm_response.error}")
                return llm_response # Early return on no choice

            llm_response.response_text = choice.message.content or "" # Content is nested in message, handle None
            llm_response.stop_reason = choice.finish_reason

            # Extract token usage information (v1.x structure)
            usage = response.usage
            if usage:
                llm_response.prompt_tokens = usage.prompt_tokens
                llm_response.completion_tokens = usage.completion_tokens
                # llm_response.total_tokens = usage.total_tokens # Optional: Add if needed in LlmResponse
            else:
                self.logger.warn("Missing usage metadata in OpenAI response.")

            # Check for specific finish reasons like content filtering
            if choice.finish_reason == "content_filter":
                 llm_response.error = "Response stopped due to content filter."
                 self.logger.warn(llm_response.error)
            elif choice.finish_reason == "length":
                 self.logger.warn(f"Response stopped due to max_tokens limit for model {model_id}.")
            # Note: OpenAIError exceptions handle API-level issues (rate limits, auth etc.)

        except OpenAIError as e: # Catch specific OpenAI errors
            llm_response.error = f"OpenAI API Error: {type(e).__name__} - {e}"
            # Log specific details if available, e.g., status code
            if hasattr(e, 'status_code'):
                llm_response.error += f" (Status: {e.status_code})"
            if hasattr(e, 'code'):
                 llm_response.error += f" (Code: {e.code})"
            self.logger.error(llm_response.error, exc_info=True)
        except Exception as e: # Catch other unexpected errors
            llm_response.error = f"Unexpected error during OpenAI call: {type(e).__name__} - {e}"
            self.logger.error(llm_response.error, exc_info=True)

        end_time = time.perf_counter()
        llm_response.duration_ms = int(round((end_time - start_time) * 1000))

        if llm_response.error:
            self.logger.info(f"OpenAI call finished in {llm_response.duration_ms}ms with error: {llm_response.error}")
        else:
            self.logger.info(f"OpenAI call succeeded in {llm_response.duration_ms}ms. Stop Reason: {llm_response.stop_reason}. Tokens(P/C): {llm_response.prompt_tokens or '?'}/{llm_response.completion_tokens or '?'}")

        return llm_response