import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

import logging
from .llm_response import LlmResponse

class LlmClient(ABC):
    """
    Abstract Base Class for Large Language Model clients.

    Provides a standard interface for sending prompts to different LLM providers
    and receiving structured responses containing text, metadata, and error info.

    Subclasses must implement the `_make_api_call` method.
    """

    def __init__(self, logger: logging.Logger, default_model_id: Optional[str] = None):
        """
        Initializes the base client.

        Args:
            logger: A Python logging.Logger instance for logging interactions.
            default_model_id: The default model identifier to use if not specified in send_message.
        """
        self.logger = logger
        self.default_model_id = default_model_id

    def send_message(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs: Any # For parameters like temperature, max_tokens, etc.
    ) -> LlmResponse:
        """
        Sends a prompt to the LLM and returns a structured response.

        Handles timing and basic logging around the API call. Subclasses
        implement the actual API interaction logic in `_make_api_call`.

        Args:
            prompt: The main user prompt/message.
            system_prompt: Optional system-level instructions for the LLM.
            model_id: Specific model identifier to use (overrides default).
            **kwargs: Additional parameters specific to the LLM provider's API
                      (e.g., temperature, max_tokens, top_p).

        Returns:
            An LlmResponse object containing the result of the API call.
        """
        start_time = time.perf_counter()
        target_model_id = model_id or self.default_model_id

        if not target_model_id:
             # Log and return an error response immediately if no model is specified
             error_msg = "No model_id specified (and no default set) for LLMClient."
             self.logger.error(f"LLM API Call FAILED: {error_msg}")
             return LlmResponse(error=error_msg)

        self.logger.info(f"Sending prompt to LLM (Model: {target_model_id})...")
        # self.logger.debug(f"System Prompt: {system_prompt}")
        # self.logger.debug(f"User Prompt: {prompt}")

        try:
            response = self._make_api_call(
                prompt=prompt,
                system_prompt=system_prompt,
                model_id=target_model_id,
                **kwargs
            )
            # Ensure duration is set even if _make_api_call forgot
            if response.duration_ms is None:
                 end_time = time.perf_counter()
                 response.duration_ms = int(round((end_time - start_time) * 1000))

            # Ensure model_id is set
            if response.model_id is None:
                response.model_id = target_model_id

            if response.is_success:
                self.logger.info(f"LLM response received: {response}")
            else:
                self.logger.error(f"LLM API Call FAILED: {response}")

            return response

        except Exception as e:
            # Catch unexpected errors during the API call itself
            end_time = time.perf_counter()
            duration_ms = int(round((end_time - start_time) * 1000))
            error_msg = f"Unexpected error during LLM API call: {e}"
            self.logger.error(error_msg, exc_info=True) # Log traceback
            return LlmResponse(
                error=error_msg,
                duration_ms=duration_ms,
                model_id=target_model_id
            )


    @abstractmethod
    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_id: str,
        **kwargs: Any
    ) -> LlmResponse:
        """
        Protected abstract method for making the actual API call.

        Subclasses must implement this method to interact with their specific
        LLM provider's API. They are responsible for:
        1. Formatting the request according to the provider's requirements.
        2. Making the network request.
        3. Handling provider-specific errors (e.g., rate limits, content filters)
           and populating the `error` field of LlmResponse if necessary.
        4. Parsing the successful response to extract:
           - `response_text`
           - `thinking` (if available)
           - `prompt_tokens`
           - `completion_tokens`
           - `stop_reason`
           - `raw_response` (optional)
        5. Calculating and setting `duration_ms` for the API call.
        6. Returning a populated `LlmResponse` object.

        Args:
            prompt: The main user prompt.
            system_prompt: Optional system prompt.
            model_id: The specific model identifier to use.
            **kwargs: Additional parameters for the API call.

        Returns:
            An LlmResponse object.
        """
        pass