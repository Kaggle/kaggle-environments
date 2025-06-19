"""Configuration utilities for LLM clients."""
import os
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this config file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, just use environment variables
    pass

def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment variables."""
    return os.getenv("ANTHROPIC_API_KEY")

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variables.""" 
    return os.getenv("OPENAI_API_KEY")

def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")