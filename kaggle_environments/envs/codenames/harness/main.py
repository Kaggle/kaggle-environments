import json
import os
import litellm

# Provide clean raw responses
litellm.drop_params = True
litellm.set_verbose = True

class LLMCodenamesAgent:
    def __init__(self, model_name=None, litellm_kwargs=None):
        self.model_name = model_name
        self.litellm_kwargs = litellm_kwargs or {}

    def __call__(self, obs, config):
        turn = obs.current_turn
        
        # 0: Red Spymaster, 2: Blue Spymaster
        if turn in [0, 2]:
            return self.spymaster_turn(obs, config)
        # 1: Red Guesser, 3: Blue Guesser
        else:
            return self.guesser_turn(obs, config)

    def _inject_memory_context(self, prompt, obs, config):
        # Inject history if present
        if hasattr(obs, "history") and obs.history:
            prompt += "\nHere is the history of past games in this session:\n"
            window_size = config.get("memory_window_size", 0)
            prompt += json.dumps(obs.history[-window_size:], indent=2)
            prompt += "\n\n"
            
        # Inject the running log of turns for the current game.
        # This provides the LLM with context on previous clues given, guesses made,
        # and their results, allowing it to see the progression of the match.
        if hasattr(obs, "current_game_turns") and obs.current_game_turns:
            prompt += "Clues and guesses in this game so far:\n"
            prompt += json.dumps(obs.current_game_turns, indent=2)
            prompt += "\n\n"
            
        return prompt

    def spymaster_turn(self, obs, config):
        roles = obs.roles
        words = obs.words
        revealed = obs.revealed
        turn = obs.current_turn
        team = "red" if turn == 0 else "blue"
        
        prompt = f"You are the {team.upper()} Spymaster in Codenames.\n\n"
        prompt += f"Your goal is to get your team to guess all your {team.upper()} words while avoiding the opposite team's words and the assassin.\n"
        
        # Inject memory context (past games and current turns)
        prompt = self._inject_memory_context(prompt, obs, config)
            
        prompt += "Here is the board state:\n"
        
        for i in range(25):
            status = "Revealed" if revealed[i] else "Hidden"
            prompt += f"- {words[i]}: {roles[i].upper()} ({status})\n"
            
        prompt += "\nThink step-by-step about which unrevealed words you can connect with a single-word clue. Provide your reasoning in a 'thinking' key.\n"
        prompt += "VALIDITY RULES FOR CLUES:\n"
        prompt += "- The clue must be a SINGLE WORD. It CANNOT contain spaces or hyphens.\n"
        prompt += "- The clue CANNOT contain or be contained within any unrevealed word currently hidden on the board (e.g., if 'DOG' is hidden, your clue cannot be 'DOGS' or 'HOTDOG').\n"
        prompt += "Note: A clue number of 0 means 'unlimited guesses, but 0 words relate to this clue' (often used to help guessers avoid the assassin or opponent words). A clue number of -1 means 'infinity' (unlimited guesses, for when you want them to guess remaining words from previous clues).\n"
        prompt += 'You MUST format your response as valid JSON like this:\n'
        prompt += '{"thinking": "I see CAT and DOG, so ANIMAL connects 2 words...", "clue": "ANIMAL", "number": 2}\n'
        prompt += "Do not include any other text or markdown formatting outside of the JSON block."
        
        messages = [{"role": "user", "content": prompt}]
        last_error = None
        for attempt in range(3):
            try:
                print(f"[{team.upper()} SPYMASTER] Calling model {self.model_name} (Attempt {attempt+1}/3)...")
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort="high",
                    **self.litellm_kwargs
                )
                print(f"[{team.upper()} SPYMASTER] Received response.")
                content = response.choices[0].message.content.strip()
                
                # Clean possible markdown format
                clean_content = content
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                    
                action = json.loads(clean_content.strip(), strict=False)
                
                if "thinking" in action:
                    print(f"Reasoning: {action['thinking']}")
                    
                if "clue" not in action or "number" not in action:
                    raise ValueError("JSON missing 'clue' or 'number' keys")
                    
                result = {"clue": action["clue"], "number": action["number"]}
                if "thinking" in action:
                    result["thinking"] = action["thinking"]
                result["prompt"] = prompt
                return result
            except Exception as e:
                last_error = e
                print(f"[{team.upper()} SPYMASTER] Parse failed on attempt {attempt+1}: {e}")
                err_msg = f"Your previous response failed to parse or was invalid. Error: {e}.\nRaw response:\n{content if 'content' in locals() else 'None'}\nPlease correct your response and format it as valid JSON strictly adhering to the original instructions."
                messages.append({"role": "assistant", "content": content if 'content' in locals() else ""})
                messages.append({"role": "user", "content": err_msg})
                
        # We must fail loudly per the harness rules if it fails after retries
        raise ValueError(f"Failed to parse Spymaster response from model after retries. Last Error: {last_error}. Raw response: {content if 'content' in locals() else 'None'}")

    def guesser_turn(self, obs, config):
        words = obs.words
        revealed = obs.revealed
        clue = obs.clue
        remaining = obs.guesses_remaining
        turn = obs.current_turn
        team = "red" if turn == 1 else "blue"
        
        clue_number = obs.clue_number
        prompt = f"You are the {team.upper()} Guesser in Codenames.\n\n"
        prompt += f"Your goal is to correctly guess your team's words based on the Spymaster's clues while avoiding the opposite team's words and the assassin.\n"
        
        # Inject memory context (past games and current turns)
        prompt = self._inject_memory_context(prompt, obs, config)
        
        # Add note to clarify the last entry in current_game_turns
        if hasattr(obs, "current_game_turns") and obs.current_game_turns:
            prompt += "Note: The last entry in the 'Clues and guesses in this game so far' list above represents your current turn, showing the guesses you have already made for the current clue.\n\n"
        
        prompt += f"The clue from your Spymaster is: '{clue}' for {clue_number} words. (You have {remaining} guesses remaining this turn.)\n\n"
        prompt += f"If you correctly guess {clue_number} words based on this clue, you may make a bonus guess based on all information you've received so far.\n\n"
        
        if clue_number == 0:
            prompt += "A clue number of 0 means NONE of your remaining words relate to this clue (often used to point out the assassin). You get unlimited guesses, but you MUST still make at least one guess.\n\n"
        elif clue_number == -1:
            prompt += "A clue number of -1 means 'Infinity'. You get unlimited guesses based on this clue and previous clues. You must make at least one guess.\n\n"
            
        prompt += "Here are the unrevealed words on the board you can choose from:\n"
        
        for i in range(25):
            if not revealed[i]:
                prompt += f"{i}: {words[i]}\n"
                
        prompt += "\nThink step-by-step about which unrevealed word matches the clue best. Provide your reasoning in a 'thinking' key.\n"
        prompt += "Then provide the integer index of the ONE word you want to guess right now in a 'guess' key.\n"
        prompt += "If you want to end your turn without guessing, set 'guess' to -1.\n"
        prompt += "You MUST format your response as valid JSON like this:\n"
        prompt += '{"thinking": "The clue is ANIMAL. Cat is at index 4, so I will guess 4...", "guess": 4}\n'
        prompt += "Do not include any other text or markdown formatting outside of the JSON block."
        
        messages = [{"role": "user", "content": prompt}]
        last_error = None
        for attempt in range(3):
            try:
                print(f"[{team.upper()} GUESSER] Calling model {self.model_name} (Attempt {attempt+1}/3)...")
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort="high",
                    **self.litellm_kwargs
                )
                print(f"[{team.upper()} GUESSER] Received response.")
                content = response.choices[0].message.content.strip()
                
                # Clean possible markdown format
                clean_content = content
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                    
                action = json.loads(clean_content.strip(), strict=False)
                
                if "thinking" in action:
                    print(f"Reasoning: {action['thinking']}")
                    
                if "guess" not in action:
                    raise ValueError("JSON missing 'guess' key")
                    
                result = {"guess": int(action["guess"])}
                if "thinking" in action:
                    result["thinking"] = action["thinking"]
                result["prompt"] = prompt
                return result
            except Exception as e:
                last_error = e
                print(f"[{team.upper()} GUESSER] Parse failed on attempt {attempt+1}: {e}")
                err_msg = f"Your previous response failed to parse or was invalid. Error: {e}.\nRaw response:\n{content if 'content' in locals() else 'None'}\nPlease correct your response and format it as valid JSON strictly adhering to the original instructions."
                messages.append({"role": "assistant", "content": content if 'content' in locals() else ""})
                messages.append({"role": "user", "content": err_msg})
                
        raise ValueError(f"Failed to parse Guesser response as JSON/integer after retries. Last Error: {last_error}. Raw response: {content if 'content' in locals() else 'None'}")

_AGENT_OBJECT = None
_SETUP_COMPLETE = False

def agent_fn(obs, config):
    global _AGENT_OBJECT, _SETUP_COMPLETE
    
    if not _SETUP_COMPLETE:
        if "MODEL_NAME" not in os.environ:
            raise ValueError("MODEL_NAME was not specified as an environment variable. Agent cannot be configured.")
            
        if "MODEL_PROXY_KEY" not in os.environ:
            raise ValueError(
                "MODEL_PROXY_KEY was not specified as an environment variable. Model proxy cannot function correctly."
            )
            
        if "MODEL_PROXY_URL" not in os.environ:
            raise ValueError("MODEL_PROXY_URL was not injected. Agent cannot run.")
            
        litellm_kwargs = {}
        if os.environ["MODEL_PROXY_URL"] != "dummy_url":
            litellm_kwargs = {
                "api_base": f"{os.environ['MODEL_PROXY_URL']}/openapi",
                "api_key": os.environ["MODEL_PROXY_KEY"],
            }
            
        model_name = os.environ["MODEL_NAME"]
        
        # When using Kaggle's proxy, all models route through OpenAI interface
        if os.environ.get("MODEL_PROXY_URL") != "dummy_url":
            # The proxy expects openai/ prefixes, e.g., google/gemini-3-pro-preview -> openai/google/gemini-3-pro-preview
            model_name = f"openai/{model_name}"
        else:
            # Special handling for local testing (Google AI Studio)
            if "gemini" in model_name.lower() and not model_name.startswith("gemini/"):
                model_name = f"gemini/{model_name}"
            
        _AGENT_OBJECT = LLMCodenamesAgent(model_name=model_name, litellm_kwargs=litellm_kwargs)
        _SETUP_COMPLETE = True

    return _AGENT_OBJECT(obs, config)
