import json
import os
import litellm

# Provide clean raw responses
litellm.drop_params = True
litellm.set_verbose = True

class LLMCodenamesAgent:
    def __init__(self, model_name=None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        self.model_name = model_name or os.environ.get("MODEL_NAME", "gemini/gemini-2.5-flash")

    def __call__(self, obs, config):
        turn = obs.current_turn
        
        # 0: Red Spymaster, 2: Blue Spymaster
        if turn in [0, 2]:
            return self.spymaster_turn(obs, config)
        # 1: Red Guesser, 3: Blue Guesser
        else:
            return self.guesser_turn(obs, config)

    def spymaster_turn(self, obs, config):
        roles = obs.roles
        words = obs.words
        revealed = obs.revealed
        turn = obs.current_turn
        team = "red" if turn == 0 else "blue"
        
        prompt = f"You are the {team.upper()} Spymaster in Codenames.\n\n"
        prompt += f"Your goal is to get your team to guess all your {team.upper()} words while avoiding the opposite team's words and the assassin.\n"
        prompt += "Here is the board state:\n"
        
        for i in range(25):
            status = "Revealed" if revealed[i] else "Hidden"
            prompt += f"- {words[i]}: {roles[i].upper()} ({status})\n"
            
        prompt += "\nProvide a single-word clue that connects multiple of your unrevealed words, and the number of words it connects.\n"
        prompt += 'You MUST format your response as valid JSON like this:\n'
        prompt += '{"clue": "ANIMAL", "number": 2}\n'
        prompt += "Do not include any other text or markdown formatting outside of the JSON block."
        
        try:
            print(f"[{team.upper()} SPYMASTER] Calling model {self.model_name}...")
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            print(f"[{team.upper()} SPYMASTER] Received response.")
            content = response.choices[0].message.content.strip()
            
            # Clean possible markdown format
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            action = json.loads(content.strip())
            if "clue" not in action or "number" not in action:
                raise ValueError("JSON missing 'clue' or 'number' keys")
                
            return action
        except Exception as e:
            # We must fail loudly per the harness rules
            raise ValueError(f"Failed to parse Spymaster response from model. Error: {e}. Raw response: {content if 'content' in locals() else 'None'}")

    def guesser_turn(self, obs, config):
        words = obs.words
        revealed = obs.revealed
        clue = obs.clue
        remaining = obs.guesses_remaining
        turn = obs.current_turn
        team = "red" if turn == 1 else "blue"
        
        prompt = f"You are the {team.upper()} Guesser in Codenames.\n"
        prompt += f"The clue from your Spymaster is: '{clue}' for {remaining} words.\n\n"
        prompt += "Here are the unrevealed words on the board you can choose from:\n"
        
        for i in range(25):
            if not revealed[i]:
                prompt += f"{i}: {words[i]}\n"
                
        prompt += "\nRespond ONLY with the integer index of the ONE word you want to guess right now.\n"
        prompt += "If you want to end your turn without guessing, respond with -1.\n"
        prompt += "Output just the number and nothing else."
        
        try:
            print(f"[{team.upper()} GUESSER] Calling model {self.model_name}...")
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            print(f"[{team.upper()} GUESSER] Received response.")
            content = response.choices[0].message.content.strip()
            return int(content)
        except Exception as e:
            raise ValueError(f"Failed to parse Guesser response as integer. Error: {e}. Raw response: {content if 'content' in locals() else 'None'}")

agent = LLMCodenamesAgent()

def agent_fn(obs, config):
    return agent(obs, config)
