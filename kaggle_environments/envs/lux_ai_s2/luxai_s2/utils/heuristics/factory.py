from luxai_s2.state import ObservationStateDict


def build_single_heavy(agent, obs: ObservationStateDict):
    actions = dict()
    if agent == "player_0":
        factories = obs["factories"][agent]
        units = obs["units"][agent]
        if len(units) == 0:
            for unit_id in factories:
                factory = factories[unit_id]
                # if factory["cargo"]["metal"] >= 100:
                actions[unit_id] = 1  # build a heavy

    return actions
