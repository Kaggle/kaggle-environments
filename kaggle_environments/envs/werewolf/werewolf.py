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


agents = {"random": random_agent}


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

    # --- Process actions for the current agent in AECEnv ---
    # The `state` passed to the interpreter contains the actions chosen by Kaggle agents
    # We need to find which agent's turn it is in AEC and apply its action.
    
    if not game.terminations[game.agent_selection] and not game.truncations[game.agent_selection]:
        current_aec_agent_id = game.agent_selection
        kaggle_agent_idx = game.agent_id_to_index[current_aec_agent_id]

        if state[kaggle_agent_idx].status == "ACTIVE":
            action_from_kaggle_agent = state[kaggle_agent_idx].action
            
            # Validate/convert action_from_kaggle_agent if necessary to fit AECEnv's action space
            # For example, Kaggle might give a single integer, AEC might expect a more complex action.
            game.step(action_from_kaggle_agent) # AECEnv processes the action
    
    # --- Update Kaggle state from AECEnv ---
    for i, agent_state in enumerate(state):
        # Get the corresponding PettingZoo agent_id string (e.g., "player_0", "player_1")
        # This assumes your Kaggle environment's JSON configures two agents,
        # and they correspond to game.possible_agents in order.
        if i < len(game.agent_ids):
            aec_agent_id_str = game.agent_ids[i]
            
            raw_obs = game.observe(aec_agent_id_str)
            agent_state.observation.raw_aec_observation = raw_obs
            
            # Rewards in AECEnv are typically for the action just taken by an agent.
            # Kaggle's `state[i].reward` is the reward attributed to agent `i` at this step.
            agent_state.reward = game.rewards.get(aec_agent_id_str, 0)
            
            # Update info for the agent in the Kaggle state
            agent_state.info = game.infos.get(aec_agent_id_str, {})
            
            if game.terminations.get(aec_agent_id_str, False) or game.truncations.get(aec_agent_id_str, False):
                agent_state.status = "DONE"
            elif game.agent_selection == aec_agent_id_str: # If it's this agent's turn next
                agent_state.status = "ACTIVE"
            else: # Other agents are waiting
                agent_state.status = "INACTIVE"
        else: # Should not happen if agent numbers match
            agent_state.status = "DONE"
            agent_state.reward = agent_state.reward if agent_state.reward is not None else 0


    # Check if all agents in AEC are done
    all_aec_done = all(game.terminations.get(ag, False) or game.truncations.get(ag, False) for ag in game.agents if ag in game.terminations) # ensure agent is in dicts
    if not game.agents or all_aec_done : # if no agents left or all are done
        print("Interpreter: All AEC agents are done. Marking Kaggle episode as DONE.")
        for i in range(len(state)):
            state[i].status = "DONE"
            # Ensure rewards are numbers before returning
            state[i].reward = state[i].reward if isinstance(state[i].reward, (int, float)) else 0
        if hasattr(env, 'my_aec_game_instance'):
             env.my_aec_game_instance.close() # Clean up AECEnv
            #  delattr(env, 'my_aec_game_instance') # Remove from env to allow re-init on next episode

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
    pass


jsonpath = path.abspath(path.join(path.dirname(__file__), "werewolf.json"))
with open(jsonpath) as f:
    specification = json.load(f)