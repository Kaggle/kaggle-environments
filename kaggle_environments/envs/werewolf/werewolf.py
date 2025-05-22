from os import path
from random import choice
import json

# my_kaggle_env.py
# Import your AECEnv game class
from .env import WerewolfEnv


def random_agent(obs):
    raise Exception(f"obs: {obs}")
    return


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
        if i < len(game.possible_agents):
            aec_agent_id_str = game.possible_agents[i]
            
            raw_obs = game.observe(aec_agent_id_str)
            agent_state.observation.raw_aec_observation = raw_obs
            
            # Rewards in AECEnv are typically for the action just taken by an agent.
            # Kaggle's `state[i].reward` is the reward attributed to agent `i` at this step.
            agent_state.reward = game.rewards.get(aec_agent_id_str, 0)
            
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
             delattr(env, 'my_aec_game_instance') # Remove from env to allow re-init on next episode

    return state



def renderer(state, env):
    # This is your JS renderer. For the Python side, you might just pass
    # a JSON-serializable representation of the game state.
    # If you have a render method in your AECEnv that produces a string or serializable output:
    # if hasattr(env, 'my_aec_game_instance'):
    #     return env.my_aec_game_instance.render(mode="ansi_string_or_json") 
    # else:
    #     return "Game not initialized."
    # For Kaggle's default HTML/JS rendering, this function usually returns
    # content that the associated .js file can understand.
    # The core.py passes this to a to_json helper.
    
    # A common pattern is to ensure the observation (which should be serializable)
    # contains enough info for the JS renderer.
    # For this example, we'll just return the raw first agent's observation for simplicity.
    return state[0].observation.get("raw_aec_observation", {})


def html_renderer():
    pass


jsonpath = path.abspath(path.join(path.dirname(__file__), "werewolf.json"))
with open(jsonpath) as f:
    specification = json.load(f)