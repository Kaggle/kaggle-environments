"""
Observation space layout computed from feature config flags.

ObsLayout is a frozen dataclass that computes all feature counts and offsets
once at environment init time. Both single-agent and multi-agent envs construct
one and use it instead of module-level constants.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyxis_portfolio_challenge.game.trial import TrialPhase

NUM_TRIAL_PHASES = len(TrialPhase)
NUM_TAS = 3  # oncology, respiratory and immunology, vaccines and infectious disease

TA_ORDER = [
    "oncology",
    "respiratory and immunology",
    "vaccines and infectious disease",
]

TA_INDEX = {ta: i for i, ta in enumerate(TA_ORDER)}


@dataclass(frozen=True)
class ObsLayout:
    """
    Observation space layout computed from feature config flags.

    All dimension counts and per-asset/per-trial offsets are computed once
    at construction. The env's hot-path observation methods read these
    integers directly — no boolean checks at observation time.
    """

    # --- Feature flags (inputs) ---
    clinical_sites_enabled: bool

    # --- Global feature counts ---
    num_site_features: int  # 4 or 0 (operational, free, in_dev, auction_active)
    offset_clinical_sites: int  # absolute index of first site feature, or -1
    global_features: int  # sum of cash(1) + above; multi-agent adds time(1)

    # --- Per-asset feature counts ---
    asset_scalar_features: int  # 10 base + ta_index(1) + indication(multi) + brand
    extra_asset_scalars: int  # number of extra scalars beyond ta_index (indication)

    # --- Per-trial feature counts ---
    num_ptrs_readings_features: int  # 1 (ptrs_sample_count_norm) or 0
    trial_features: int  # 3 base + ptrs_readings
    asset_total_features: int  # asset_scalar + NUM_TRIAL_PHASES * trial_features

    # --- Feature flags (marketing) ---
    marketing_enabled: bool
    offset_brand_score: int  # -1 when disabled

    # --- Per-asset scalar offsets (relative to asset block start) ---
    # Base 10 scalars always present at offsets 0-9:
    #   0: max_revenue, 1: time_until_max_revenue, 2: time_until_patent_expiry,
    #   3: pending_trial_phase, 4: time_on_market, 5: cost_this_step,
    #   6: revenue_this_step, 7: enpv, 8: eroi, 9: state
    offset_ta_index: int
    offset_indication: int  # -1 if no indication feature (single-agent)

    # --- Per-trial offsets (relative to trial block start) ---
    # Base 3 always present: 0: cost_remaining, 1: time_remaining, 2: ptrs
    offset_ptrs_count: int  # -1 when disabled (ptrs_readings feature)

    @classmethod
    def from_config(
        cls,
        marketing_config=None,
        ptrs_readings_config=None,
        clinical_sites_config=None,
        *,
        has_time_feature: bool = False,
        has_indication_feature: bool = False,
    ) -> ObsLayout:
        """
        Build layout from config objects.

        Parameters
        ----------
        marketing_config
            Configuration for the marketing spend feature. None if disabled.
        ptrs_readings_config
            Configuration for the PTRS readings feature. None if disabled.
        clinical_sites_config
            Configuration for the clinical sites feature. None if disabled.
        has_time_feature
            True for multi-agent env (adds time to global features).
        has_indication_feature
            True for multi-agent env (adds indication per asset).

        """
        marketing_on = marketing_config is not None and marketing_config.enabled
        sites_on = (
            clinical_sites_config is not None and clinical_sites_config.enabled
        )

        # Global features
        # Clinical-site globals: operational_sites, free_sites,
        # num_sites_in_development, site_auction_active.
        num_sites = 4 if sites_on else 0
        base_global = 2 if has_time_feature else 1  # cash [+ time]
        # Site features are appended after all other global blocks.
        offset_clinical_sites = base_global if sites_on else -1
        global_features = base_global + num_sites

        # Per-asset scalars
        has_indication = 1 if has_indication_feature else 0

        # Base 10 scalars at fixed offsets 0-9
        base_scalar = 10
        cur = base_scalar

        offset_ta_index = cur
        cur += 1  # ta_index always present

        offset_indication = cur if has_indication_feature else -1
        cur += has_indication

        num_brand_score = 1 if marketing_on else 0
        offset_brand_score = cur if marketing_on else -1
        cur += num_brand_score

        asset_scalar_features = cur
        extra = has_indication

        # Per-trial features
        ptrs_readings_on = (
            ptrs_readings_config is not None and ptrs_readings_config.enabled
        )
        num_ptrs_readings = 1 if ptrs_readings_on else 0
        trial_features = 3 + num_ptrs_readings

        offset_ptrs_count = 3 if ptrs_readings_on else -1

        asset_total_features = asset_scalar_features + NUM_TRIAL_PHASES * trial_features

        return cls(
            marketing_enabled=marketing_on,
            offset_brand_score=offset_brand_score,
            clinical_sites_enabled=sites_on,
            num_site_features=num_sites,
            offset_clinical_sites=offset_clinical_sites,
            global_features=global_features,
            asset_scalar_features=asset_scalar_features,
            extra_asset_scalars=extra,
            num_ptrs_readings_features=num_ptrs_readings,
            trial_features=trial_features,
            asset_total_features=asset_total_features,
            offset_ta_index=offset_ta_index,
            offset_indication=offset_indication,
            offset_ptrs_count=offset_ptrs_count,
        )
