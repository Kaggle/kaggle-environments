import time
import random
from typing import Optional, Any, Dict, List

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

try:
    from ..config import get_gemini_api_key
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import get_gemini_api_key
import logging
from .base_client import LlmClient
from .llm_response import LlmResponse


class GeminiClient(LlmClient):
    _KWARG_MAP = {
        "temperature": "temperature",
        "max_tokens": "max_output_tokens",
        "stop_sequences": "stop_sequences",
        "top_p": "top_p",
        "top_k": "top_k",
    }

    def __init__(self, logger: logging.Logger, default_model_id: str = "gemini-2.5-pro"):
        super().__init__(logger, default_model_id)
        try:
            api_key = get_gemini_api_key()
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY environment variable not found or is empty.")
            genai.configure(api_key=api_key)
            self.logger.info("Gemini API configured successfully.")
        except EnvironmentError as e:
            self.logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error configuring Gemini API: {e}", exc_info=True)

    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_id: str,
        **kwargs: Any
    ) -> LlmResponse:
        start_time = time.perf_counter()
        llm_response = LlmResponse(model_id=model_id)
        
        # Define retryable error types
        retryable_exceptions = (
            google_exceptions.ResourceExhausted,  # Rate limits, quota issues (429)
            google_exceptions.ServiceUnavailable,  # Server unavailable (503)
            google_exceptions.DeadlineExceeded,   # Request timeout
            google_exceptions.InternalServerError, # Server errors (500)
            google_exceptions.GatewayTimeout,     # Gateway timeout (504)
            google_exceptions.TooManyRequests,    # Explicit too many requests
            ConnectionError,                      # Network connectivity issues
            TimeoutError                          # General timeouts
        )
        
        # Define retry parameters
        max_retries = 5
        base_delay = 1.0  # Base delay in seconds
        
        # Generate content configuration (outside retry loop to avoid rebuilding each time)
        generation_config_dict: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in self._KWARG_MAP:
                generation_config_dict[self._KWARG_MAP[key]] = value
        generation_config = genai.GenerationConfig(**generation_config_dict)
        
        contents = []
        contents.append({"role": "user", "parts": [prompt]})
        
        # Initialize model with system instruction (outside retry loop)
        try:
            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=system_prompt if system_prompt else None
            )
        except Exception as e:
            llm_response.error = f"Failed to initialize Gemini model: {e}"
            self.logger.error(llm_response.error, exc_info=True)
            return llm_response
            
        # Retry loop
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Only log first attempt and retries
                if retry_count == 0:
                    self.logger.debug(f"Calling Gemini model {model_id} with prompt and config...")
                else:
                    self.logger.info(f"Retry {retry_count}/{max_retries} for Gemini model {model_id}...")
                
                # Make the API call
                response = model.generate_content(contents=contents, generation_config=generation_config)
                self.logger.debug("Gemini response received.")
                
                # Process successful response
                self.logger.debug(f"Gemini response received for model: {model_id}")
                llm_response.raw_response = response
                candidate = response.candidates[0] if response.candidates else None
                
                if not candidate:
                    reason = getattr(response.prompt_feedback.block_reason, "name", "UNKNOWN")
                    llm_response.error = f"No candidates returned. Block reason: {reason}"
                    llm_response.stop_reason = reason
                    self.logger.warn(f"Gemini API call failed: {llm_response.error}")
                    # Content blocks are not retryable
                    break
                
                # Extract text content more safely
                text = ""
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text += part.text
                llm_response.response_text = text
                llm_response.thinking = None
                
                finish_reason = getattr(candidate.finish_reason, "name", "UNKNOWN")
                llm_response.stop_reason = finish_reason
                
                if hasattr(response, "usage_metadata"):
                    llm_response.prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
                    llm_response.completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
                    # Calculate completion tokens from total if candidates_token_count is not available
                    if llm_response.completion_tokens is None:
                        total_tokens = getattr(response.usage_metadata, "total_token_count", None)
                        if total_tokens and llm_response.prompt_tokens:
                            llm_response.completion_tokens = total_tokens - llm_response.prompt_tokens
                else:
                    self.logger.warn("Missing usage metadata in Gemini response.")
                
                if finish_reason == "SAFETY":
                    blocked = [r.category.name for r in candidate.safety_ratings if r.blocked]
                    llm_response.error = f"Response blocked due to SAFETY. Categories: {blocked}"
                    self.logger.warn(llm_response.error)
                    # Safety blocks are not retryable
                    break
                
                # If we got here, the call was successful
                break
                
            except retryable_exceptions as e:
                retry_count += 1
                error_msg = str(e)
                
                # If we've reached max retries, record the error and break
                if retry_count > max_retries:
                    if isinstance(e, google_exceptions.ResourceExhausted):
                        llm_response.error = f"Quota Exceeded after {max_retries} retries: {error_msg}"
                    else:
                        llm_response.error = f"Retryable error persisted after {max_retries} retries: {error_msg}"
                    self.logger.error(llm_response.error, exc_info=True)
                    break
                
                # Calculate backoff delay with jitter (randomization)
                delay = min(base_delay * (2 ** (retry_count - 1)), 60)  # Cap at 60 seconds
                jitter = random.uniform(0, 0.5 * delay)  # Add up to 50% jitter
                sleep_time = delay + jitter
                
                self.logger.warn(f"Retryable error occurred (retry {retry_count}/{max_retries}): {error_msg}")
                self.logger.info(f"Waiting {sleep_time:.2f}s before retry...")
                time.sleep(sleep_time)
                continue
                
            except google_exceptions.PermissionDenied as e:
                # Auth/permission errors not retryable
                llm_response.error = f"Permission Denied: {e}"
                self.logger.error(llm_response.error, exc_info=True)
                break
                
            except google_exceptions.GoogleAPIError as e:
                # Other API errors are not retryable by default
                llm_response.error = f"Google API Error: {e}"
                self.logger.error(llm_response.error, exc_info=True)
                break
                
            except Exception as e:
                # Unexpected errors not retryable
                llm_response.error = f"Unexpected error: {e}"
                self.logger.error(llm_response.error, exc_info=True)
                break
        
        # Capture total duration including all retries
        end_time = time.perf_counter()
        llm_response.duration_ms = int(round((end_time - start_time) * 1000))
        
        if llm_response.error:
            self.logger.info(f"Gemini call failed in {llm_response.duration_ms}ms: {llm_response.error}")
        else:
            retries_msg = f" after {retry_count} retries" if retry_count > 0 else ""
            self.logger.info(f"Gemini call succeeded{retries_msg} in {llm_response.duration_ms}ms. Stop Reason: {llm_response.stop_reason}")
        
        return llm_response