from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from pyxis_portfolio_challenge.game.asset import AssetState
    from pyxis_portfolio_challenge.game.trial import Trial


class TrialsJSON(BaseModel):
    """Variant of the Trials model for JSON Structured inputs."""

    phase_1: Trial
    phase_2: Trial
    phase_3: Trial


class DrugAssetJSON(BaseModel):
    """Variant of the DrugAsset model for JSON Structured inputs."""

    name: str
    therapeutic_area: Literal[
        "oncology", "respiratory and immunology", "vaccines and infectious disease"
    ]
    type: Literal["internal", "BD"] = "internal"
    description: str
    max_revenue: float  # M
    time_until_max_revenue: int  # H
    time_until_patent_expiry: int  # T
    trials: TrialsJSON
    state: "AssetState"
    pending_trial_phase: Optional[Literal["Phase 1", "Phase 2", "Phase 3"]]
    time_on_market: int = 0


class DrugAssetListJSON(BaseModel):
    """List of DrugAssetJSON instances."""

    assets: list[DrugAssetJSON]
