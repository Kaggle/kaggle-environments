from __future__ import annotations
from enum import Enum
from typing import Literal, Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ------------------------------------------------------------------ #
class Action(BaseModel):
    """Root of the discriminated-union tree."""
    actor_id: str
    reasoning: Optional[str] = Field(default=None, max_length=4096)


# ——— Mix-in for actions that need a target ------------------------ #
class TargetedAction(Action):
    target_id: str


# ——— Concrete leaf classes --------------------------------------- #
class EliminateAction(TargetedAction):
    pass


class EliminateProposalAction(TargetedAction):
    """
    A werewolf's *suggestion* for tonight’s victim to be eliminated.
    These actions never reach the public log.
    """
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


ACTION_REGISTRY = {
    ActionKind.ELIMINATE: EliminateAction,
    ActionKind.HEAL: HealAction,
    ActionKind.INSPECT: InspectAction,
    ActionKind.VOTE: VoteAction,
    ActionKind.CHAT: ChatAction,
    ActionKind.NOOP: NoOpAction,
}


def create_action(kind: ActionKind, **kwargs) -> Action:
    return ACTION_REGISTRY[kind](**kwargs)
