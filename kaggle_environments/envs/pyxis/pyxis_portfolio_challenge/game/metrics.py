import logging
import uuid
from datetime import datetime
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

from pyxis_portfolio_challenge.game.game_state import GameState

logger = logging.getLogger(__name__)


class GameMetricsData(BaseModel):
    """
    Data model for game analytics metrics.

    Also used as a model for a single entry in the level leaderboard.
    """

    model_config = ConfigDict(extra="forbid")
    user_id: str
    level_id: int
    game_id: str
    final_enpv: float
    final_eroi: float
    realised_roi: float
    final_capital: float
    eroi_over_time: list[float]
    enpv_over_time: list[float]
    av_enpv: float
    timestamp: datetime


def prepare_game_metrics_data(
    user_name: str,
    level_idx: int,
    previous_game_state: GameState,
    game_state: GameState,
    actions: dict[uuid.UUID, Optional[Literal["invest"]]],
) -> GameMetricsData:
    """
    Prepare game metrics data after a game completion.

    Parameters
    ----------
    user_name : str
        The name of the user (or agent).
    level_idx : int
        The index of the level played.
    previous_game_state : GameState
        The state of the game before the final actions were taken.
    game_state : GameState
        The final state of the game after all actions and evolutions.
    actions : dict[uuid.UUID, Optional[Literal["invest"]]]
        The actions taken by the user during the game.

    Returns
    -------
    GameMetricsData
        The prepared game metrics data ready for insertion into the database.

    """
    game_metrics = GameMetricsData(
        user_id=str(user_name),
        level_id=level_idx,
        game_id=str(game_state.id),
        final_enpv=previous_game_state.enpv(),
        final_eroi=previous_game_state.eroi(),
        realised_roi=game_state.realised_roi(),
        final_capital=game_state.cash,
        eroi_over_time=game_state.eroi_over_time,
        enpv_over_time=game_state.enpv_over_time,
        av_enpv=np.mean(game_state.enpv_over_time),
        timestamp=datetime.now(),
    )
    return game_metrics
