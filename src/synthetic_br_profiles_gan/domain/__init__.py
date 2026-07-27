"""Tabelas de domínio brasileiras usadas por geradores e validadores."""

from synthetic_br_profiles_gan.domain.brazil import (
    REGION_STATES,
    REGIONS,
    STATE_DDDS,
    STATE_MUNICIPALITIES,
    STATE_REGION,
    all_ddds,
    ddds_for_state,
    municipalities_for_state,
    region_for_state,
    states_for_region,
)

__all__ = [
    "REGION_STATES",
    "REGIONS",
    "STATE_DDDS",
    "STATE_MUNICIPALITIES",
    "STATE_REGION",
    "all_ddds",
    "ddds_for_state",
    "municipalities_for_state",
    "region_for_state",
    "states_for_region",
]
