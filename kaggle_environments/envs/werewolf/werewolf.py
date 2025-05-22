from os import path
import json
import random # Added for random.choice

# my_kaggle_env.py
# Import your AECEnv game class
from .env import WerewolfEnv, ActionType, Role, Phase, WerewolfObservationModel # Added ActionType, Role, Phase


def random_agent(obs):
    raw_aec_obs = obs.get('raw_aec_observation')

    # Default to NO_OP if observation is missing or agent cannot act
    if not raw_aec_obs:
        return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

    current_phase = Phase(raw_aec_obs['phase'])
    my_role = Role(raw_aec_obs['role'])

    all_player_names = json.loads(raw_aec_obs['all_player_unique_names'])
    my_unique_name = raw_aec_obs['my_unique_name']

    my_idx = all_player_names.index(my_unique_name)

    alive_player_indices = [i for i, status in enumerate(raw_aec_obs['alive_players']) if status == 1]

    action_to_take = {"action_type": ActionType.NO_OP.value} # Default action

    if current_phase == Phase.NIGHT_WEREWOLF_VOTE:
        if my_role == Role.WEREWOLF:
            known_ww_status = raw_aec_obs['known_werewolves']
            # Werewolves target alive non-werewolf players
            potential_targets = [
                idx for idx in alive_player_indices
                if idx < len(known_ww_status) and known_ww_status[idx] == 0
            ]
            if potential_targets:
                target_idx = random.choice(potential_targets)
                action_to_take = {"action_type": ActionType.NIGHT_KILL_VOTE.value, "target_idx": target_idx}
    
    elif current_phase == Phase.NIGHT_DOCTOR_SAVE:
        if my_role == Role.DOCTOR:
            # Doctors can save any alive player (including themselves)
            if alive_player_indices:
                target_idx = random.choice(alive_player_indices)
                action_to_take = {"action_type": ActionType.NIGHT_SAVE_TARGET.value, "target_idx": target_idx}

    elif current_phase == Phase.NIGHT_SEER_INSPECT:
        if my_role == Role.SEER:
            # Seers can inspect any alive player
            if alive_player_indices:
                target_idx = random.choice(alive_player_indices)
                action_to_take = {"action_type": ActionType.NIGHT_INSPECT_TARGET.value, "target_idx": target_idx}

    elif current_phase == Phase.DAY_DISCUSSION:
        if my_idx in alive_player_indices: # Only alive players can discuss
            messages = [
                "Hello everyone!", 
                "I have a strong feeling about someone.", 
                "Any information to share?", 
                "I am a simple Villager just trying to survive.", 
                "Let's think carefully before voting."
            ]
            
            if len(alive_player_indices) > 0:
                rand_player_for_msg_idx = random.choice(alive_player_indices)
                messages[1] = f"I think {all_player_names[rand_player_for_msg_idx]} is acting suspiciously."
                
                votable_for_message = [p_idx for p_idx in alive_player_indices if p_idx != rand_player_for_msg_idx]
                if votable_for_message:
                    rand_player_for_vote_msg_idx = random.choice(votable_for_message)
                    messages[4] = f"We should consider voting for {all_player_names[rand_player_for_vote_msg_idx]} today."
                elif len(alive_player_indices) == 1: 
                     messages[4] = "It seems I'm the only one left to talk to."

            action_to_take = {"action_type": ActionType.DAY_DISCUSS.value, "message": random.choice(messages)}

    elif current_phase == Phase.DAY_VOTING:
        if my_idx in alive_player_indices: # Only alive players can vote
            votable_targets = [p_idx for p_idx in alive_player_indices if p_idx != my_idx]
            if votable_targets:
                target_idx = random.choice(votable_targets)
                action_to_take = {"action_type": ActionType.DAY_LYNCH_VOTE.value, "target_idx": target_idx}
    
    elif current_phase == Phase.GAME_OVER:
        action_to_take = {"action_type": ActionType.NO_OP.value}
        
    if "target_idx" not in action_to_take:
        action_to_take["target_idx"] = None
    if "message" not in action_to_take:
        action_to_take["message"] = None
        
    return action_to_take


# This function is part of the skeleton and retained as a placeholder.
def dummy_inference_endpoint(prompt):
    # In a real scenario, this would query an LLM.
    # For testing, we can make it return a valid JSON action string.
    # Example: return '{"action_type": "NO_OP"}'
    return '{"action_type": "NO_OP", "message": "dummy action"}'


endpoints = {'dummy_llm': dummy_inference_endpoint}


class LLMAgent:
    def __init__(self, model_name="dummy_llm", system_prompt="You are a helpful assistant playing Werewolf."):
        """
        Initializes the LLMAgent.
        Args:
            model_name (str): Identifier for the LLM model (currently conceptual).
            system_prompt (str): A system prompt to guide the LLM's behavior.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.memory = [] # Stores a history of observations or processed information
        self.inferencer = endpoints[model_name]
    
    def parse_llm_response_to_action(self, llm_response_str: str) -> dict:
        """
        Parses a JSON string from an LLM into a valid game action dictionary.

        Args:
            llm_response_str: The JSON string response from the LLM.
                            Expected format: {"action_type": "ACTION_NAME_STR", "target_idx": int_or_null, "message": "str_or_null"}

        Returns:
            A dictionary representing the game action, or a NO_OP action if parsing fails.
        """
        try:
            action_data = json.loads(llm_response_str)
            if not isinstance(action_data, dict):
                raise ValueError("LLM response is not a JSON object.")

            action_type_str = action_data.get("action_type")
            if not action_type_str or not hasattr(ActionType, action_type_str):
                raise ValueError(f"Invalid or missing 'action_type': {action_type_str}")

            action_type_enum_val = ActionType[action_type_str].value
            target_idx = action_data.get("target_idx") # Can be None
            message = action_data.get("message")     # Can be None

            # Basic type check for target_idx if present
            if target_idx is not None and not isinstance(target_idx, int):
                target_idx = None # Or raise error, but defaulting to None is safer for NO_OP fallback

            return {"action_type": action_type_enum_val, "target_idx": target_idx, "message": message}

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Error parsing LLM response '{llm_response_str}': {e}. Defaulting to NO_OP.")
            return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}
    
    def __call__(self, obs):
        """
        Processes an observation, updates memory, and decides on an action.
        Currently, it only stores the observation and returns a NO_OP action.
        """
        raw_aec_obs = obs.get('raw_aec_observation')

        if not raw_aec_obs:
            # Default action if no observation is available
            return {"action_type": ActionType.NO_OP.value, "target_idx": None, "message": None}

        # Convert raw observation to a more readable format.
        # If WerewolfObservationModel instantiation or get_human_readable fails,
        # the error will propagate as per the "no try-except" constraint.
        pydantic_obs = WerewolfObservationModel(**raw_aec_obs)
        human_readable_obs = pydantic_obs.get_human_readable()
        
        # Update memory
        self.memory.append(human_readable_obs)

        # --- Placeholder for actual LLM interaction (conceptual) ---
        current_prompt = f"{self.system_prompt}\n\nObservation History:\n{json.dumps(self.memory, indent=2)}\n\nWhat is your action?"
        llm_response_action_str = self.inferencer(current_prompt)
        action_to_take = self.parse_llm_response_to_action(llm_response_action_str)
        
        return action_to_take
    

agents = {"random": random_agent, "dummy_llm": LLMAgent('dummy_llm')}


def interpreter(state, env):
    """
    state: list of dictionaries, one for each agent.
           Each dict has: {observation, action, reward, status, info}
    env:   the kaggle_environments.Environment object itself.
           We can use this to store our AECEnv instance across steps.
    """
    
    # --- Initialize AECEnv instance if it's the start of an episode ---
    # `env.steps` is a list of (state, actions) tuples from previous steps.
    # `len(env.steps) == 1` indicates the first time interpreter is called for actions in an episode,
    # as `env.steps` initially contains just the reset state.
    # Or, if using `env.run()`, the state list might be empty on the very first internal call.
    # A more robust check is if our specific game instance attribute is not set on `env`.

    is_new_episode = not hasattr(env, 'my_aec_game_instance') or env.done
    if is_new_episode:
        print("Interpreter: Initializing new AECEnv game instance.")
        
        # Prepare parameters for WerewolfEnv constructor (excluding num_players)
        aec_params = {}
        num_players_for_aec = len(state) # Number of players from Kaggle's state

        # Copy other relevant params from env.configuration if they exist
        # and are expected by WerewolfEnv (e.g., num_doctors, num_seers).
        # These should be defined in the werewolf.json or specification dict
        # if they are to be configurable via make(configuration={...}).
        if hasattr(env.configuration, "num_doctors"):
            aec_params["num_doctors"] = env.configuration.num_doctors
        if hasattr(env.configuration, "num_seers"):
            aec_params["num_seers"] = env.configuration.num_seers
        if hasattr(env.configuration, "max_days"):
            aec_params["max_days"] = env.configuration.max_days

        # render_mode is part of WerewolfEnv's signature but usually handled differently by Kaggle
        # if hasattr(env.configuration, "render_mode"):
        #     aec_params["render_mode"] = env.configuration.render_mode
        env.my_aec_game_instance = WerewolfEnv(**aec_params)
        
        # Assign to 'game' here so it's available within the loop below
        game = env.my_aec_game_instance
        
        # Pass num_players to reset via options
        reset_options = {"num_players": num_players_for_aec}
        env.my_aec_game_instance.reset(options=reset_options) # Initialize the AEC game state
        
        # Set initial observations for all Kaggle agents from the AECEnv
        for i, agent_id_str in enumerate(env.my_aec_game_instance.agents):
            # The 'state' here is the Kaggle state object we need to populate
            # The first agent in AEC might have an observation ready after reset
            # For other agents, observe() might be needed or they wait their turn.
            # This initial population might also be partly handled by an on_reset hook
            # in your environment's JSON if you define one.
            # Initialize info for each agent in the Kaggle state
            initial_aec_info = game.infos.get(env.my_aec_game_instance.agents[i], {})
            state[i].info = initial_aec_info
            if state[i].status == "ACTIVE": # or INACTIVE if waiting for first turn
                # This mapping assumes Kaggle agent index maps to PettingZoo agent index
                raw_obs = env.my_aec_game_instance.observe(agent_id_str)
                state[i].observation = {"raw_aec_observation": raw_obs} # Structure as needed
                state[i].reward = 0 # Initial reward
                # state[i].status will be updated by Kaggle core based on AECEnv's termination

    game = env.my_aec_game_instance
    processed_actor_idx_for_this_step = None

    # --- Process actions for the current agent in AECEnv ---
    if game.agent_selection and not game.terminations.get(game.agent_selection, False) and not game.truncations.get(game.agent_selection, False):
        current_aec_agent_id = game.agent_selection
        kaggle_agent_idx = game.agent_id_to_index[current_aec_agent_id]

        if state[kaggle_agent_idx].status == "ACTIVE":
            action_from_kaggle_agent = state[kaggle_agent_idx].action
            game.step(action_from_kaggle_agent) # AECEnv processes the action
            # After game.step(), game.active_player_indices_history is updated.
            # The last element is the index of the agent who just acted.
            if game.active_player_indices_history and game.active_player_indices_history[-1] is not None:
                processed_actor_idx_for_this_step = game.active_player_indices_history[-1]
    elif game.agent_selection and (game.terminations.get(game.agent_selection, False) or game.truncations.get(game.agent_selection, False)):
        # If the selected agent is already done, env.step might just advance the selector
        # We still need to record who was supposed to act if it's relevant for history
        # However, game.step() for a dead agent usually just cycles.
        # The important actor is the one whose action changes state.
        # If no one acts, processed_actor_idx_for_this_step remains None or reflects last actor.
        # Let's ensure it's set if an action was processed.
        # If game.step() is called for a dead agent, it might not update active_player_indices_history meaningfully for *this* step.
        # This case might need refinement based on how game.step() handles dead agent turns.
        # For now, we assume active_player_indices_history[-1] is the key after a state-changing step.
        pass


    # --- Update Kaggle state from AECEnv ---
    for i, agent_state in enumerate(state):
        # Get the corresponding PettingZoo agent_id string
        if i < len(game.agent_ids):
            aec_agent_id_str = game.agent_ids[i]

            raw_obs = game.observe(aec_agent_id_str)
            # Ensure observation is a dict, not the Pydantic model instance
            agent_state.observation["raw_aec_observation"] = raw_obs

            agent_state.reward = game.rewards.get(aec_agent_id_str, 0)
            current_info = game.infos.get(aec_agent_id_str, {})
            agent_state.info = current_info # game.infos contains last_action_feedback etc.

            # Add extra info for the JS renderer
            agent_state.info["actor_for_this_kaggle_step"] = processed_actor_idx_for_this_step

            if game.current_phase == Phase.GAME_OVER and game.game_winner_team:
                agent_state.info["game_winner_team"] = game.game_winner_team

            if game.terminations.get(aec_agent_id_str, False) or game.truncations.get(aec_agent_id_str, False):
                agent_state.status = "DONE"
            elif game.agent_selection == aec_agent_id_str:
                agent_state.status = "ACTIVE"
            else:
                agent_state.status = "INACTIVE"
        else:
            agent_state.status = "DONE"
            agent_state.reward = agent_state.reward if agent_state.reward is not None else 0


    all_aec_done = all(game.terminations.get(ag, False) or game.truncations.get(ag, False) for ag in game.agents if ag in game.terminations)
    if not game.agents or all_aec_done :
        print("Interpreter: All AEC agents are done. Marking Kaggle episode as DONE.")
        for i_done in range(len(state)):
            state[i_done].status = "DONE"
            state[i_done].reward = state[i_done].reward if isinstance(state[i_done].reward, (int, float)) else 0
            # Ensure winner team info is in the final state for JS
            if hasattr(game, 'game_winner_team') and game.game_winner_team:
                 state[i_done].info["game_winner_team"] = game.game_winner_team
            # Ensure actor info is present even if it was None for the last phase transition
            if "actor_for_this_kaggle_step" not in state[i_done].info:
                state[i_done].info["actor_for_this_kaggle_step"] = processed_actor_idx_for_this_step

        if hasattr(env, 'my_aec_game_instance'):
             env.my_aec_game_instance.close()

    return state


def renderer(state, env):
    if not hasattr(env, 'my_aec_game_instance'):
        return "Werewolf game instance not initialized yet."

    game = env.my_aec_game_instance
    output_lines = []

    current_kaggle_step_index = env.render_step_ind

    if current_kaggle_step_index == 0:
        return "*** Werewolf Game Initialized ***"

    # acting_agent_kaggle_idx is the Kaggle index of the agent that took an action
    # which resulted in the state env.steps[current_kaggle_step_index].
    acting_agent_kaggle_idx = game.active_player_indices_history[current_kaggle_step_index]

    if acting_agent_kaggle_idx is None:
        return f"Error: No acting agent found in history for step index {current_kaggle_step_index}."

    acting_agent_id_str = game.agent_ids[acting_agent_kaggle_idx]

    # current_k_step_state_list is env.steps[current_kaggle_step_index] (passed as 'state' to renderer)
    # This is the state *after* the acting_agent_kaggle_idx took their action.
    current_k_step_state_list = state 
    action_agent_took = current_k_step_state_list[acting_agent_kaggle_idx].action
    
    # previous_k_step_state_list is env.steps[current_kaggle_step_index - 1]
    # This contains the observation the agent received *before* acting.
    previous_k_step_state_list = env.steps[current_kaggle_step_index - 1]
    
    # Observation the agent received to make its decision
    obs_agent_received_full = previous_k_step_state_list[acting_agent_kaggle_idx].observation
    obs_agent_received_raw_aec = obs_agent_received_full.get("raw_aec_observation")

    # The status of the agent *when it was called to make the action*
    status_when_acting = previous_k_step_state_list[acting_agent_kaggle_idx].status

    output_lines.append(f"--- Werewolf Game State ---")
    
    current_phase_val = obs_agent_received_raw_aec.get('phase') if obs_agent_received_raw_aec else None
    current_phase_str = Phase(current_phase_val).name if current_phase_val is not None else 'N/A'
    output_lines.append(f"Current Phase (when agent acted): {current_phase_str}")
    
    output_lines.append(f"Active Agent ID: {acting_agent_id_str}")    
    output_lines.append(f"  Kaggle Agent Index: {acting_agent_kaggle_idx}")
    output_lines.append(f"  Kaggle Agent Status (when agent acted): {status_when_acting}")

    if obs_agent_received_raw_aec:
        role_val = obs_agent_received_raw_aec.get('role')
        role_str = Role(role_val).name if role_val is not None else 'N/A'
        output_lines.append(f"  Observation for {acting_agent_id_str} (Role: {role_str}):")
        obs = WerewolfObservationModel(**obs_agent_received_raw_aec)
        for key, value in obs.get_human_readable().items():
            output_lines.append(f"    {key}: {value}")
    else:
        output_lines.append(f"  No raw_aec_observation found for {acting_agent_id_str}.")

    # Use action_description_for_log from the agent's info in the *current* step's state
    agent_info_after_action = current_k_step_state_list[acting_agent_kaggle_idx].info
    action_description = agent_info_after_action.get("action_description_for_log", str(action_agent_took))
    output_lines.append(f"Action Processed by Env: {action_description}")

    return "\n".join(output_lines)


def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "werewolf.js"))
    with open(jspath, encoding="utf-8") as f:
        return f.read()


jsonpath = path.abspath(path.join(path.dirname(__file__), "werewolf.json"))
with open(jsonpath) as f:
    specification = json.load(f)