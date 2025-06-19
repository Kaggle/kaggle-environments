from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LlmResponse:
    """
    Represents the structured response from an LLM API call.
    """
    # Core content
    response_text: Optional[str] = None  # The primary text response from the LLM
    thinking: Optional[str] = None      # Optional thinking/reasoning steps provided by the LLM

    # Metadata for logging and analysis
    model_id: Optional[str] = None      # Specific model/version used (e.g., "gemini-1.5-pro-latest")
    prompt_tokens: Optional[int] = None # Number of tokens in the input prompt
    completion_tokens: Optional[int] = None # Number of tokens in the generated response
    duration_ms: Optional[int] = None   # Time taken for the API call in milliseconds
    stop_reason: Optional[str] = None   # Reason the LLM stopped generating (e.g., 'stop', 'max_tokens', 'error')

    # Error handling
    error: Optional[str] = None         # Error message if the API call failed

    # Additional raw data if needed for debugging
    raw_response: Optional[dict] = field(default=None, repr=False) # The full, raw response object (optional)

    @property
    def is_success(self) -> bool:
        """Returns True if the API call was successful and produced text."""
        return self.error is None and self.response_text is not None

    def __str__(self) -> str:
        """Simple string representation for logging."""
        if self.is_success:
            tok_info = f"P:{self.prompt_tokens or '?'} C:{self.completion_tokens or '?'}"
            duration_info = f"{self.duration_ms or '?'}ms"
            thinking_info = f" Thinking:{self.thinking[:50]}..." if self.thinking else ""
            return (f"LLMResponse(Success | Model:{self.model_id or '?'} | Tokens:{tok_info} | "
                    f"Duration:{duration_info} | Stop:{self.stop_reason or '?'} | "
                    f"Response:{self.response_text[:100]}...{thinking_info})")
        else:
            return f"LLMResponse(Error: {self.error or 'Unknown API Error'})"