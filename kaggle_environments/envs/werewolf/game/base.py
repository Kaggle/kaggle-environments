from abc import ABC, abstractmethod
from typing import Type, Dict, Protocol, Any, Annotated, Optional, List

from pydantic import BaseModel, StringConstraints

from .consts import EVENT_HANDLER_FOR_ATTR_NAME, EventName, MODERATOR_ID


# The ID regex supports Unicode letters (\p{L}), numbers (\p{N}) and common symbol for ID.
ROBUST_ID_REGEX = r'^[\p{L}\p{N} _.-]+$'

PlayerID = Annotated[str, StringConstraints(pattern=ROBUST_ID_REGEX, min_length=1, max_length=128)]


class BasePlayer(BaseModel):
    id: PlayerID
    """The unique id of the player. Also, how the player is referred to in the game."""


class BaseAction(BaseModel):
    pass


class BaseState(BaseModel):
    @abstractmethod
    def push_event(self,
                   description: str,
                   event_name: EventName,
                   public: bool,
                   visible_to: Optional[List[PlayerID]] = None,
                   data: Any = None, source=MODERATOR_ID):
        """Publish an event."""


class BaseHistoryEntry(BaseModel):
    event_name: EventName
    

class BaseModerator(ABC):
    def advance(self, player_actions: Dict[PlayerID, BaseAction]):
        """"""

    def request_action(
            self, action_cls: Type[BaseAction], player_id: PlayerID, prompt: str, data=None,
            event_name=EventName.MODERATOR_ANNOUNCEMENT
    ):
        """"""

    @abstractmethod
    def record_night_save(self, doctor_id: str, target_id: str):
        pass

    @property
    @abstractmethod
    def state(self) -> BaseState:
        pass


def on_event(event_type: EventName):
    def decorator(func):
        setattr(func, EVENT_HANDLER_FOR_ATTR_NAME, event_type)
        return func
    return decorator


class EventHandler(Protocol):
    def __call__(self, event: BaseHistoryEntry) -> Any:
        pass


class RoleEventHandler(Protocol):
    def __call__(self, me: BasePlayer, moderator: BaseModerator, event: BaseHistoryEntry) -> Any:
        pass



class BaseRole(BaseModel, ABC):
    def get_event_handlers(self) -> Dict[EventName, RoleEventHandler]:
        """Inspects the role instance and collects all methods decorated with @on_event"""
        handlers = {}
        for attr_name in dir(self):
            if not attr_name.startswith('__'):
                attr = getattr(self, attr_name)
                if callable(attr) and hasattr(attr, EVENT_HANDLER_FOR_ATTR_NAME):
                    event_type = getattr(attr, EVENT_HANDLER_FOR_ATTR_NAME)
                    handlers[event_type] = attr
        return handlers
