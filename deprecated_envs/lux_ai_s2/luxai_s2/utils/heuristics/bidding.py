from luxai_s2.unit import BidActionType
from luxai_s2.state import ObservationStateDict
def zero_bid(player, obs: ObservationStateDict) -> BidActionType:
    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"
    return dict(bid=0, faction=faction)