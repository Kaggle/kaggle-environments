#!/usr/bin/env python3
"""
Example usage of the LLM clients for chess prompt evaluation.

This script demonstrates how to use the cleaned up LLM clients.
Set the appropriate API keys as environment variables before running.
"""

import os
from logger import get_logger
from llm import AnthropicClient, OpenAIClient, GeminiClient, LlmResponse

def main():
    """Demonstrate usage of the LLM clients."""
    logger = get_logger("chess_llm_example")
    
    # Example chess prompt
    chess_prompt = """
    Analyze this chess position and suggest the best move:
    
    Position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
    
    Please provide your analysis and recommended move.
    """
    
    system_prompt = "You are a chess grandmaster providing move analysis."
    
    print("Chess Prompt Evaluation Example")
    print("=" * 40)
    
    # Check which clients can be initialized based on available API keys
    clients = []
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            anthropic_client = AnthropicClient(logger)
            clients.append(("Anthropic", anthropic_client))
            print("✓ Anthropic client initialized")
        except Exception as e:
            print(f"✗ Anthropic client failed: {e}")
    else:
        print("- Anthropic API key not found (set ANTHROPIC_API_KEY)")
        
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_client = OpenAIClient(logger)
            clients.append(("OpenAI", openai_client))
            print("✓ OpenAI client initialized")
        except Exception as e:
            print(f"✗ OpenAI client failed: {e}")
    else:
        print("- OpenAI API key not found (set OPENAI_API_KEY)")
        
    if os.getenv("GEMINI_API_KEY"):
        try:
            gemini_client = GeminiClient(logger)
            clients.append(("Gemini", gemini_client))
            print("✓ Gemini client initialized")
        except Exception as e:
            print(f"✗ Gemini client failed: {e}")
    else:
        print("- Gemini API key not found (set GEMINI_API_KEY)")
    
    if not clients:
        print("\nNo API keys found. Set at least one of:")
        print("- ANTHROPIC_API_KEY")
        print("- OPENAI_API_KEY") 
        print("- GEMINI_API_KEY")
        return
    
    print(f"\nTesting {len(clients)} available client(s)...")
    print()
    
    # Test each available client
    for name, client in clients:
        print(f"Testing {name} client:")
        print("-" * 20)
        
        try:
            response: LlmResponse = client.send_message(
                prompt=chess_prompt,
                system_prompt=system_prompt,
                max_tokens=2000,  # Higher limit for reasoning models (o3, Gemini 2.5 Pro)
                temperature=0.7
            )
            
            if response.is_success:
                print(f"✓ Success! Duration: {response.duration_ms}ms")
                print(f"Model: {response.model_id}")
                print(f"Tokens - Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens}")
                print(f"Stop reason: {response.stop_reason}")
                print(f"Response preview: {response.response_text[:200]}...")
                if response.thinking:
                    print(f"Thinking preview: {response.thinking[:100]}...")
            else:
                print(f"✗ Failed: {response.error}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            
        print()

if __name__ == "__main__":
    main()