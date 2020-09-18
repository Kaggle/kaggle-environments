from functools import wraps
from gfootball.env import football_action_set

def agent_wrapper(agent):
    """
    Decorator allowing agent code to return native GRF actions.
    @agent_wrapper
    def my_agent(obs):
        ...
        return football_action_set.action_right
    """
    core_action_to_int = { action : nr for nr, action in enumerate(football_action_set.action_set_v1) }

    @wraps(agent)
    def agent_wrapper(obs):
        action = agent(obs)
        if isinstance(action, football_action_set.CoreAction):
            return [core_action_to_int[action]]
        return [action]

    return agent_wrapper
