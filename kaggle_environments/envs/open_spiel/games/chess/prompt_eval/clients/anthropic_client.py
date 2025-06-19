import time
from typing import Optional, Any, Dict

import anthropic
from anthropic import AnthropicError

try:
    from ..config import get_anthropic_api_key
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import get_anthropic_api_key
import logging
from .base_client import LlmClient
from .llm_response import LlmResponse


class AnthropicClient(LlmClient):
    _KWARG_MAP = {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequences": "stop_sequences",
        "thinking_tokens": "thinking_budget_tokens"  # Special parameter for thinking budget
    }

    def __init__(self, logger: logging.Logger, default_model_id: str = "claude-sonnet-4-20250514"):
        super().__init__(logger, default_model_id)
        self.client = None
        try:
            api_key = get_anthropic_api_key()
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY environment variable not found or is empty.")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.logger.info("Anthropic client initialized successfully.")
        except EnvironmentError as e:
            self.logger.error(f"Failed to configure Anthropic client: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error configuring Anthropic client: {e}", exc_info=True)

    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_id: str,
        max_tokens: int = 2048,  # Set a default value
        **kwargs: Any
    ) -> LlmResponse:
        start_time = time.perf_counter()
        llm_response = LlmResponse(model_id=model_id)

        if not self.client:
            llm_response.error = "Anthropic client was not initialized (missing API key?)."
            self.logger.error(llm_response.error)
            llm_response.duration_ms = int(round((time.perf_counter() - start_time) * 1000))
            return llm_response

        try:
            # Map standard kwargs to Anthropic specific ones
            api_params = {}
            thinking_enabled = False
            thinking_budget_tokens = None
            
            # Debug: log all kwargs received
            self.logger.debug(f"Anthropic client received kwargs: {kwargs}")
            
            for standard_key, value in kwargs.items():
                anthropic_key = self._KWARG_MAP.get(standard_key)
                if anthropic_key == "thinking_budget_tokens" and value:
                    thinking_enabled = True
                    thinking_budget_tokens = value
                elif anthropic_key:
                    api_params[anthropic_key] = value
                else:
                    self.logger.debug(f"Ignoring unknown parameter: {standard_key}={value}")

            # Construct messages for Anthropic API
            messages = []
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            })

            # Configure thinking if enabled
            if thinking_enabled and thinking_budget_tokens:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens
                }

            # Add max_tokens to parameters if not already present
            if 'max_tokens' not in api_params:
                api_params['max_tokens'] = max_tokens  # Use the parameter default value
                
            self.logger.debug(f"Calling Anthropic model {model_id} with parameters: {api_params}")
                
            # Handle system prompt properly - Anthropic expects a string, not None or list
            params = {
                "model": model_id,
                "messages": messages,
                **api_params
            }
            
            # Only add system parameter if it's a non-empty string
            if system_prompt:
                params["system"] = system_prompt
                
            response = self.client.messages.create(**params)

            self.logger.debug("Anthropic response received.")
            llm_response.raw_response = response.dict() if hasattr(response, "dict") else response

            # Extract the text response - handle different content block types
            llm_response.response_text = ""
            for content_block in response.content:
                # Different content blocks might have different attributes
                if hasattr(content_block, 'text'):
                    llm_response.response_text += content_block.text
                elif hasattr(content_block, 'value'):
                    llm_response.response_text += content_block.value
                # Skip other content blocks we don't understand
            
            # Extract thinking if available
            if hasattr(response, "thinking") and response.thinking:
                llm_response.thinking = response.thinking.value

            # Extract token usage information
            if hasattr(response, "usage"):
                usage = response.usage
                llm_response.prompt_tokens = usage.input_tokens
                llm_response.completion_tokens = usage.output_tokens
            else:
                self.logger.warn("Missing usage metadata in Anthropic response.")

            # Set stop reason
            if hasattr(response, "stop_reason"):
                llm_response.stop_reason = response.stop_reason
            else:
                llm_response.stop_reason = "unknown"

            # Check for content filtering
            if llm_response.stop_reason == "content_filtered":
                llm_response.error = "Response stopped due to content filtering."
                self.logger.warn(llm_response.error)
            elif llm_response.stop_reason == "max_tokens":
                self.logger.warn(f"Response stopped due to max_tokens limit for model {model_id}.")

        except AnthropicError as e:
            llm_response.error = f"Anthropic API Error: {type(e).__name__} - {e}"
            self.logger.error(llm_response.error, exc_info=True)
        except Exception as e:
            llm_response.error = f"Unexpected error during Anthropic call: {type(e).__name__} - {e}"
            self.logger.error(llm_response.error, exc_info=True)

        end_time = time.perf_counter()
        llm_response.duration_ms = int(round((end_time - start_time) * 1000))

        if llm_response.error:
            self.logger.info(f"Anthropic call finished in {llm_response.duration_ms}ms with error: {llm_response.error}")
        else:
            self.logger.info(f"Anthropic call succeeded in {llm_response.duration_ms}ms. Stop Reason: {llm_response.stop_reason}. Tokens(P/C): {llm_response.prompt_tokens or '?'}/{llm_response.completion_tokens or '?'}")

        return llm_response