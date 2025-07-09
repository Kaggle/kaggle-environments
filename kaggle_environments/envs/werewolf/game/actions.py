from __future__ import annotations
from enum import Enum
from typing import Literal, Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ------------------------------------------------------------------ #
class Action(BaseModel):
    """Root of the discriminated-union tree."""
    actor_id: str
    reasoning: Optional[str] = Field(default=None, max_length=4096)

    def serialize(self):
        return {'action_type': self.__class__.__name__, 'kwargs': self.model_dump()}


# ——— Mix-in for actions that need a target ------------------------ #
class TargetedAction(Action):
    target_id: str


# ——— Concrete leaf classes --------------------------------------- #
class EliminateProposalAction(TargetedAction):
    pass


class HealAction(TargetedAction):
    pass


class InspectAction(TargetedAction):
    pass


class VoteAction(TargetedAction):
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
