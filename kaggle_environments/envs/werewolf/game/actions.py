from __future__ import annotations
from enum import Enum
from typing import Literal, Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


from .consts import PerceivedThreatLevel


# ------------------------------------------------------------------ #
class Action(BaseModel):
    """Root of the discriminated-union tree."""
    day: int
    phase: str
    actor_id: str
    reasoning: Optional[str] = Field(
        default=None, max_length=4096,
        description="The self monologue that illustrate how you arrived at the action. "
                    "It will be invisible to other players.")

    perceived_threat_level: PerceivedThreatLevel = Field(
        default=PerceivedThreatLevel.SAFE,
        description="The self perceived threat level you are currently experiencing from other players. "
                    "The assessment will be invisible to other players."
    )

    def serialize(self):
        return {'action_type': self.__class__.__name__, 'kwargs': self.model_dump()}


# ——— Mix-in for actions that need a target ------------------------ #
class TargetedAction(Action):
    target_id: str = Field(description="The target player's id.")


# ——— Concrete leaf classes --------------------------------------- #
class HealAction(TargetedAction):
    pass


class InspectAction(TargetedAction):
    pass


class VoteAction(TargetedAction):
    pass


class EliminateProposalAction(VoteAction):
    pass


class ChatAction(Action):
    message: str = Field(default="", max_length=4096)


class NoOpAction(Action):
    pass


# ------------------------------------------------------------ #
class BidAction(Action):
    """
    An amount the actor is willing to pay this round.
    Currency unit can be generic 'chips' or role-specific.
    """
    amount: int = Field(ge=0)


ACTIONS = [
    EliminateProposalAction,
    HealAction,
    InspectAction,
    VoteAction,
    ChatAction,
    BidAction,
    NoOpAction
]

ACTION_REGISTRY = {
    action.__name__: action for action in ACTIONS
}

def create_action(serialized):
    return ACTION_REGISTRY[serialized['action_type']](**serialized.get('kwargs', {}))
